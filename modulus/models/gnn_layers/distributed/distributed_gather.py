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

from typing import Tuple, List
from .distributed_graph import DistributedGraph


def gather_scatter_tensor_fwd(
        tensor: torch.Tensor, 
        local_partition_group: dist.ProcessGroup, 
        scatter_indices: torch.Tensor,
        gather_sizes: List[int],
        scatter_sizes: List[int],
        num_local_src_nodes: int
) -> torch.Tensor:
    scatter_idx, scatter_sizes = scatter_indices
    total_gathered_elem = sum(gather_sizes)
    local_rank = dist.get_rank(group=local_partition_group)
    global_rank = dist.get_rank()
    assert total_gathered_elem == num_local_src_nodes,\
        f"Expected {num_local_src_nodes} to receive on rank {local_rank} (global rank {global_rank}), got {total_gathered_elem}."

    tensor_to_scatter = tensor[scatter_idx, :]
    tensor_to_gather = torch.empty(
        (total_gathered_elem, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    dist.all_to_all_single(
        tensor_to_gather, 
        tensor_to_scatter, 
        output_split_sizes=gather_sizes,
        input_split_sizes=scatter_sizes,
        group=local_partition_group)

    return tensor_to_gather


def gather_scatter_tensor_bwd(
        tensor: torch.Tensor, 
        local_partition_group: dist.ProcessGroup, 
        scatter_indices: torch.Tensor,
        gather_sizes: List[int],
        scatter_sizes: List[int],
        num_local_dst_nodes: int
) -> torch.Tensor:
    # gather_indices, scatter_indices correspond to forward
    # in backward, the roles are reversed
    out = torch.zeros(
        (num_local_dst_nodes, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
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
        group=local_partition_group)

    out.scatter_add_(
        src=tensor_to_gather, 
        index=scatter_indices.view(-1, 1).expand(-1, tensor.size(1)),
        dim=0
    )

    return out


class GatherScatterAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, graph: DistributedGraph) -> torch.Tensor:
        assert tensor.size(0) == graph.num_local_dst_nodes
        ctx.graph = graph
        src_tensor = gather_scatter_tensor_fwd(tensor, graph.local_partition_group, graph.gather_sizes, graph.scatter_sizes,  graph.scatter_index, graph.num_local_src_nodes)
        return src_tensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_src_tensor: torch.Tensor):
        graph = ctx.graph
        assert grad_src_tensor.size(0) == graph.num_local_src_nodes
        need_grad_tensor, need_grad_graph = ctx.needs_input_grad
        assert need_grad_graph == False
        grad_dst_tensor = None
        if need_grad_tensor:
            grad_dst_tensor = gather_scatter_tensor_bwd(grad_src_tensor, graph.local_partition_group, graph.gather_sizes, graph.scatter_sizes,  graph.scatter_index, graph.num_local_dst_nodes)
        return grad_dst_tensor, None


def gather_scatter(tensor: torch.Tensor, graph: DistributedGraph):
    return GatherScatterAutograd.apply(tensor, graph)
