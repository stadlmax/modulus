# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.distributed as dist

from typing import List, Optional


def _all_gather_idx_first_dim_fwd(
    tensor: torch.Tensor,
    sizes: List[int],
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert tensor.dim() == 2
    dim = tensor.size(1)
    output = [
        torch.empty((size, dim), dtype=tensor.dtype, device=tensor.device)
        for size in sizes
    ]
    dist.all_gather(output, tensor, group=process_group)
    return torch.cat(output, dim=0)


def _all_gather_idx_first_dim_bwd(
    grad_output: torch.Tensor,
    sizes: List[int],
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert grad_output.dim() == 2
    dim = grad_output.size(1)
    local_size = sizes[dist.get_rank(group=process_group)]

    grad_tensor = [
        torch.empty(
            (local_size, dim), dtype=grad_output.dtype, device=grad_output.device
        )
        for _ in sizes
    ]
    scatter_grad_output = list(torch.split(grad_output, sizes, dim=0))
    scatter_grad_output = [t.contiguous() for t in scatter_grad_output]
    dist.all_to_all(grad_tensor, scatter_grad_output, group=process_group)
    grad_tensor = torch.stack(grad_tensor, dim=2)
    grad_tensor = grad_tensor.sum(dim=2)
    return grad_tensor


class AllGatherIdxFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        process_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.sizes = sizes
        ctx.process_group = process_group
        return _all_gather_idx_first_dim_fwd(tensor, sizes, process_group)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_tensor = _all_gather_idx_first_dim_bwd(
            grad_output, ctx.sizes, ctx.process_group
        )
        return grad_tensor, None, None


def all_gather_idx_first_dim(
    tensor: torch.Tensor, sizes: List[int], process_group: dist.ProcessGroup = None
) -> torch.Tensor:
    return AllGatherIdxFirstDimAutograd.apply(tensor, sizes, process_group)


class DistributedGraph:
    def __init__(
        self,
        global_offsets: torch.Tensor,
        global_indices: torch.Tensor,
        partition_size: int,
        partition_groups: List[dist.ProcessGroup],
    ):
        self.local_offsets = None
        self.local_indices = None
        self.dst_nodes_per_partition = None
        self.src_nodes_per_partition = None
        self.num_local_src_nodes = None
        self.num_local_dst_nodes = None
        self.num_local_edges = None
        self.num_global_src_nodes = global_indices.max().item() + 1
        self.num_global_dst_nodes = global_offsets.size(0) - 1
        self.num_global_edges = global_indices.size(0)

        self.gather_sizes = None
        self.scatter_sizes = None
        self.scatter_indices = None

        self.local_indices_to_global = None
        self.local_src_node_ids_to_global = None
        self.local_dst_node_ids_to_global = None

        self.num_src_nodes_per_partition = None
        self.num_dst_nodes_per_partition = None
        self.num_indices_per_partition = [None] * partition_size

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = self.rank % self.world_size
        self.partition_rank = self.rank % partition_size
        self.partition_size = partition_size
        self.partition_id = self.rank // partition_size

        self.partition_groups = partition_groups
        self.local_partition_group = partition_groups[self.partition_id]

        # this partitions offsets and indices on each rank in the same fashion
        # it could be rewritten to do it on one rank and exchange the partitions
        # however, as we expect the global graphs not to be too large for one CPU
        # we do it once and then can get rid of it afterwards without going through
        # tedious gather/scatter routines for communicating the partitions

        # get distribution of destination IDs
        dst_nodes_per_partition = (
            self.num_global_dst_nodes + self.partition_size - 1
        ) // self.partition_size
        dst_offsets_per_partition = [
            rank * dst_nodes_per_partition for rank in range(self.partition_size + 1)
        ]
        dst_offsets_per_partition[-1] = min(
            self.num_global_dst_nodes, dst_offsets_per_partition[-1]
        )

        src_nodes_per_partition = (
            self.num_global_src_nodes + self.partition_size - 1
        ) // self.partition_size
        src_offsets_per_partition = [
            rank * src_nodes_per_partition for rank in range(self.partition_size + 1)
        ]
        src_offsets_per_partition[-1] = min(
            self.num_global_src_nodes, src_offsets_per_partition[-1]
        )

        scatter_indices = [None] * self.partition_size
        gather_sizes = [None] * self.partition_size

        for rank in range(self.partition_size):
            offset_start = dst_offsets_per_partition[rank]
            offset_end = dst_offsets_per_partition[rank + 1]
            offsets = global_offsets[offset_start : offset_end + 1].detach().clone()
            partition_indices = (
                global_indices[offsets[0] : offsets[-1]].detach().clone()
            )
            offsets -= offsets[0].item()

            global_src_ids_per_rank, inverse_mapping = partition_indices.unique(
                sorted=True, return_inverse=True
            )
            local_src_ids_per_rank = torch.arange(
                0, global_src_ids_per_rank.size(0), dtype=offsets.dtype
            )
            global_src_ids_to_gpu = global_src_ids_per_rank // src_nodes_per_partition
            remote_src_ids_per_rank = (
                global_src_ids_per_rank
                - global_src_ids_to_gpu * src_nodes_per_partition
            )

            indices = local_src_ids_per_rank[inverse_mapping]
            self.num_indices_per_partition[rank] = indices.size(0)

            if rank == self.partition_rank:
                self.num_local_edges = indices.size(0)
                self.num_local_dst_nodes = offsets.size(0) - 1
                self.num_dst_nodes_per_partition = [
                    dst_offsets_per_partition[rank + 1]
                    - dst_offsets_per_partition[rank]
                    for rank in range(self.partition_size)
                ]
                self.num_local_src_nodes = global_src_ids_per_rank.size(0)
                self.local_src_node_ids_to_global = range(
                    src_offsets_per_partition[rank], src_offsets_per_partition[rank + 1]
                )
                self.num_src_nodes_per_partition = [
                    src_offsets_per_partition[rank + 1]
                    - src_offsets_per_partition[rank]
                    for rank in range(self.partition_size)
                ]
                self.num_local_src_nodes_partition = self.num_src_nodes_per_partition[
                    rank
                ]
                self.local_dst_node_ids_to_global = range(
                    dst_offsets_per_partition[rank], dst_offsets_per_partition[rank + 1]
                )
                self.local_indices_to_global = range(
                    global_offsets[offset_start], global_offsets[offset_end]
                )

                self.local_offsets = offsets.detach().clone().to(device=self.device_id)
                self.local_indices = indices.detach().clone().to(device=self.device_id)

            for rank_offset in range(self.partition_size):
                mask = global_src_ids_to_gpu == rank_offset
                if self.partition_rank == rank:
                    # simply count number of nonzero elements in mask
                    gather_sizes[rank_offset] = mask.sum().item()

                if self.partition_rank == rank_offset:
                    # indices to scatter to this rank on rank_offset
                    scatter_indices[rank] = (
                        remote_src_ids_per_rank[mask]
                        .detach()
                        .clone()
                        .to(device=self.device_id, dtype=torch.int64)
                    )

        # concatenate indices and save sizes
        # this makes later communication easier
        scatter_sizes = [idx.size(0) for idx in scatter_indices]

        self.scatter_sizes = scatter_sizes
        self.gather_sizes = gather_sizes
        self.scatter_indices = torch.cat(scatter_indices, dim=0)

    def get_local_src_node_features(
        self, global_node_features: torch.Tensor
    ) -> torch.Tensor:
        # TODO backward
        return global_node_features[self.local_src_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_local_dst_node_features(
        self, global_node_features: torch.Tensor
    ) -> torch.Tensor:
        # TODO backward
        return global_node_features[self.local_dst_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_local_edge_features(
        self, global_edge_features: torch.Tensor
    ) -> torch.Tensor:
        # TODO backward
        return global_edge_features[self.local_indices_to_global, :].to(
            device=self.device_id
        )

    def get_global_src_node_features(
        self, local_node_features: torch.Tensor
    ) -> torch.Tensor:
        return all_gather_idx_first_dim(
            local_node_features,
            self.num_src_nodes_per_partition,
            self.local_partition_group,
        )

    def get_global_dst_node_features(
        self, local_node_features: torch.Tensor
    ) -> torch.Tensor:
        return all_gather_idx_first_dim(
            local_node_features,
            self.num_dst_nodes_per_partition,
            self.local_partition_group,
        )

    def get_global_edge_features(
        self, local_edge_features: torch.Tensor
    ) -> torch.Tensor:
        return all_gather_idx_first_dim(
            local_edge_features,
            self.num_indices_per_partition,
            self.local_partition_group,
        )


def _all_to_all_idx_first_dim_fwd(
    tensor: torch.Tensor,
    local_partition_group: dist.ProcessGroup,
    scatter_indices: torch.Tensor,
    gather_sizes: List[int],
    scatter_sizes: List[int],
    num_local_src_nodes: int,
) -> torch.Tensor:
    total_gathered_elem = sum(gather_sizes)
    local_rank = dist.get_rank(group=local_partition_group)
    global_rank = dist.get_rank()
    assert (
        total_gathered_elem == num_local_src_nodes
    ), f"Expected {num_local_src_nodes} to receive on rank {local_rank} (global rank {global_rank}), got {total_gathered_elem}."

    tensor_to_scatter = tensor[scatter_indices, :]
    tensor_to_gather = torch.empty(
        (total_gathered_elem, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    dist.all_to_all_single(
        tensor_to_gather,
        tensor_to_scatter,
        output_split_sizes=gather_sizes,
        input_split_sizes=scatter_sizes,
        group=local_partition_group,
    )

    return tensor_to_gather


def _all_to_all_idx_first_dim_bwd(
    tensor: torch.Tensor,
    local_partition_group: dist.ProcessGroup,
    scatter_indices: torch.Tensor,
    gather_sizes: List[int],
    scatter_sizes: List[int],
    num_local_nodes: int,
) -> torch.Tensor:
    # gather_indices, scatter_indices correspond to forward
    # in backward, the roles are reversed
    out = torch.empty(
        (num_local_nodes, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    total_gathered_elem = sum(scatter_sizes)

    tensor_to_scatter = tensor
    tensor_to_gather = torch.empty(
        (total_gathered_elem, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    dist.all_to_all_single(
        tensor_to_gather,
        tensor_to_scatter,
        output_split_sizes=scatter_sizes,
        input_split_sizes=gather_sizes,
        group=local_partition_group,
    )

    out.scatter_add_(
        src=tensor_to_gather,
        index=scatter_indices.view(-1, 1).expand(-1, tensor.size(1)),
        dim=0,
    )

    return out


class AllToAllIdxFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, graph: DistributedGraph) -> torch.Tensor:
        ctx.graph = graph
        local_rank = dist.get_rank(group=graph.local_partition_group)
        global_rank = dist.get_rank()
        expected_rows = graph.num_src_nodes_per_partition[local_rank]
        assert (
            tensor.size(0) == expected_rows
        ), f"Expected tensor.size(0) = {expected_rows} on rank {local_rank} (global rank {global_rank}), but got {tensor.size(0)} instead."

        src_tensor = _all_to_all_idx_first_dim_fwd(
            tensor,
            graph.local_partition_group,
            graph.scatter_indices,
            graph.gather_sizes,
            graph.scatter_sizes,
            graph.num_local_src_nodes,
        )
        ctx.src_shape = src_tensor.shape
        ctx.tensor_shape = tensor.shape
        return src_tensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_src_tensor: torch.Tensor):
        graph = ctx.graph
        need_grad_src_tensor, need_grad_graph = ctx.needs_input_grad
        assert need_grad_graph == False
        local_rank = dist.get_rank(group=graph.local_partition_group)
        global_rank = dist.get_rank()
        assert (
            grad_src_tensor.size(0) == graph.num_local_src_nodes
        ), f"Expected tensor.size(0) = {graph.num_local_src_nodes} on rank {local_rank} (global rank {global_rank}), but got {tensor.size(0)} instead."

        grad_dst_tensor = None
        if need_grad_src_tensor:
            grad_dst_tensor = _all_to_all_idx_first_dim_bwd(
                grad_src_tensor,
                graph.local_partition_group,
                graph.scatter_indices,
                graph.gather_sizes,
                graph.scatter_sizes,
                graph.num_local_src_nodes_partition,
            )
        return grad_dst_tensor, None


def all_to_all_idx_first_dim(tensor: torch.Tensor, graph: DistributedGraph):
    return AllToAllIdxFirstDimAutograd.apply(tensor, graph)
