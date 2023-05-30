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

from typing import List


class DistributedGraph:
    def __init__(self, global_offsets: torch.Tensor, global_indices: torch.Tensor, partition_size: int, partition_groups: List[dist.ProcessGroup]):
        self.local_offsets = None
        self.local_indices = None
        self.num_local_src_nodes = None
        self.num_local_dst_nodes = None
        self.num_local_edges = None
        self.local_graph = None

        self.gather_sizes = None
        self.scatter_sizes = None
        self.scatter_indices = None

        self.local_edge_ids_to_global = None
        self.local_node_ids_to_global = None


        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = self.rank % self.world_size
        self.partition_rank = self.rank % partition_size
        self.partition_size = partition_size
        self.partition_id = self.rank // partition_size

        self.partition_groups = partition_groups
        self.local_partition_group = partition_groups[self.partition_id]
        
        self.partition_graph(global_offsets, global_indices)
    
    def partition_graph(self, global_offsets: torch.Tensor, global_indices: torch.Tensor):
        # this partitions offsets and indices on each rank in the same fashion
        # it could be rewritten to do it on one rank and exchange the partitions
        # however, as we expect the global graphs not to be too large for one CPU
        # we do it once and then can get rid of it afterwards without going through
        # tedious gather/scatter routines for communicating the partitions

        num_global_nodes = global_offsets.size(0) - 1
        nodes_per_partition = (num_global_nodes + self.partition_size - 1) // self.partition_size

        scatter_indices = [None] * self.partition_size
        gather_sizes = [None] * self.partition_size

        for rank in range(self.partition_size):
            offset_start = nodes_per_partition * rank
            offset_end   = min(num_global_nodes, nodes_per_partition * (rank + 1)) + 1
            offsets = global_offsets[offset_start:offset_end].detach().clone()
            partition_indices = global_indices[offsets[0]:offsets[-1]].detach().clone()
            offsets -= global_offsets[offset_start]

            global_src_ids_per_rank, inverse_mapping = partition_indices.unique(sorted=True, return_inverse=True)
            global_src_ids_to_gpu = global_src_ids_per_rank // nodes_per_partition
            
            local_src_ids_per_rank  = torch.arange(0, global_src_ids_per_rank.size(0))
            remote_src_ids_per_rank = global_src_ids_per_rank - global_src_ids_to_gpu * nodes_per_partition

            indices = local_src_ids_per_rank[inverse_mapping]

            if rank == self.partition_rank:
                self.num_local_edges = indices.size(0)
                self.num_local_dst_nodes = offsets.size(0) - 1
                self.num_local_src_nodes = indices.max().item() + 1
                self.local_node_ids_to_global = range(offset_start, offset_end - 1)
                self.local_edge_ids_to_global = range(global_offsets[offset_start], global_offsets[offset_end-1])
                self.local_offsets = offsets.detach().clone().to(device=self.device_id)
                self.local_indices = indices.detach().clone().to(device=self.device_id)

            for rank_offset in range(self.partition_size):
                mask = global_src_ids_to_gpu == rank_offset
                if rank == self.partition_rank:
                    # simply count number of nonzero elements in mask
                    gather_sizes[rank_offset] = mask.sum().detach()

                if rank_offset == self.partition_rank:        
                    # indices to scatter to this rank on rank_offset
                    scatter_indices[rank] = remote_src_ids_per_rank[mask].detach().clone().to(device=self.device_id)

        # concatenate indices and save sizes
        # this makes later communication easier
        scatter_sizes = [idx.size(0) for idx in scatter_indices]

        self.scatter_sizes = scatter_sizes
        self.gather_sizes = gather_sizes
        self.scatter_indices = torch.cat(scatter_indices, dim=0)

    def get_local_node_features(self, global_node_features: torch.Tensor) -> torch.Tensor:
        return global_node_features[self.local_node_ids_to_global, :].to(device=self.device_id)

    def get_local_edge_features(self, global_edge_features: torch.Tensor) -> torch.Tensor:
        return global_edge_features[self.local_edge_ids_to_global, :].to(device=self.device_id)
 