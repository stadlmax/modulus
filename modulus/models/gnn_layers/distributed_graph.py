# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .distributed_utils import (
    all_gatherv_first_dim,
    gatherv_first_dim,
    scatterv_first_dim,
    all_to_all_idx_first_dim,
)


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

        self.recv_sizes = None
        self.send_sizes = None
        self.send_indices = None

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

        send_indices = [None] * self.partition_size
        recv_sizes = [None] * self.partition_size

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
                    recv_sizes[rank_offset] = mask.sum().item()

                if self.partition_rank == rank_offset:
                    # indices to send to this rank from this rank
                    send_indices[rank] = (
                        remote_src_ids_per_rank[mask]
                        .detach()
                        .clone()
                        .to(device=self.device_id, dtype=torch.int64)
                    )

        # concatenate indices and save sizes
        # this makes later communication easier
        send_sizes = [idx.size(0) for idx in send_indices]

        self.send_sizes = send_sizes
        self.recv_sizes = recv_sizes
        self.send_indices = torch.cat(send_indices, dim=0)

    def get_partioned_local_src_node_features(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatterv_first_dim(
                global_node_features,
                self.num_src_nodes_per_partition,
                self.local_partition_group,
            )

        return global_node_features[self.local_src_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_all_local_src_node_features(
        self, partioned_src_node_features: torch.Tensor
    ) -> torch.Tensor:
        # main primitive to gather all necessary src features
        # which are required for a csc-based message passing step
        return all_to_all_idx_first_dim(
            partioned_src_node_features,
            self.send_indices,
            self.recv_sizes,
            self.send_sizes,
            self.local_partition_group,
            self.num_src_nodes_per_partition,
        )

    def get_local_dst_node_features(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatterv_first_dim(
                global_node_features,
                self.num_dst_nodes_per_partition,
                self.local_partition_group,
            )

        return global_node_features[self.local_dst_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_local_edge_features(
        self,
        global_edge_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatterv_first_dim(
                global_edge_features,
                self.num_indices_per_partition,
                self.local_partition_group,
            )
        return global_edge_features[self.local_indices_to_global, :].to(
            device=self.device_id
        )

    def get_global_src_node_features(
        self,
        local_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gatherv_first_dim(
                local_node_features,
                self.num_src_nodes_per_partition,
                self.local_partition_group,
            )

        return all_gatherv_first_dim(
            local_node_features,
            self.num_src_nodes_per_partition,
            self.local_partition_group,
        )

    def get_global_dst_node_features(
        self, local_node_features: torch.Tensor, get_on_all_ranks: bool = True
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gatherv_first_dim(
                local_node_features,
                self.num_dst_nodes_per_partition,
                self.local_partition_group,
            )

        return all_gatherv_first_dim(
            local_node_features,
            self.num_dst_nodes_per_partition,
            self.local_partition_group,
        )

    def get_global_edge_features(
        self, local_edge_features: torch.Tensor, get_on_all_ranks: bool = True
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gatherv_first_dim(
                local_edge_features,
                self.num_indices_per_partition,
                self.local_partition_group,
            )

        return all_gatherv_first_dim(
            local_edge_features,
            self.num_indices_per_partition,
            self.local_partition_group,
        )
