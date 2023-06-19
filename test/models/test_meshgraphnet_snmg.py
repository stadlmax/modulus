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

import sys, os

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import pytest
import torch
import numpy as np

from torch.nn.parallel import DistributedDataParallel

from modulus.models.gnn_layers.utils import CuGraphCSC
from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet
from modulus.distributed import DistributedManager
from modulus.distributed.utils import create_process_groups, custom_allreduce_fut


def test_distributed_meshgraphnet(
    partition_size: int, dtype: torch.dtype, do_concat_trick: bool = False
):
    dist = DistributedManager()

    model_kwds = {
        "input_dim_nodes": 3,
        "input_dim_edges": 4,
        "output_dim": 5,
        "processor_size": 10,
        "num_layers_node_processor": 2,
        "num_layers_edge_processor": 2,
        "hidden_dim_node_encoder": 256,
        "num_layers_node_encoder": 2,
        "hidden_dim_edge_encoder": 256,
        "num_layers_edge_encoder": 2,
        "hidden_dim_node_decoder": 256,
        "num_layers_node_decoder": 2,
        "do_concat_trick": do_concat_trick,
    }

    device = dist.local_rank
    partition = dist.local_rank // partition_size

    # initialize single GPU model for reference
    torch.cuda.manual_seed(partition)
    torch.manual_seed(partition)
    np.random.seed(partition)

    model_single_gpu = MeshGraphNet(**model_kwds).to(device=device, dtype=dtype)
    # initialze distributed model with the same seeds
    torch.cuda.manual_seed(partition)
    torch.manual_seed(partition)
    np.random.seed(partition)

    num_partitions = dist.world_size // partition_size
    partition_groups = create_process_groups(partition_size)
    custom_hook = lambda process_group, bucket: custom_allreduce_fut(
        process_group, bucket.buffer(), divisor=num_partitions
    )
    model_multi_gpu = MeshGraphNet(**model_kwds).to(device=device, dtype=dtype)

    model_multi_gpu = DistributedDataParallel(
        model_multi_gpu, process_group=partition_groups[partition]
    )
    model_multi_gpu.register_comm_hook(None, custom_hook)

    # initialize data
    min_degree, max_degree = 3, 6
    num_nodes = 1024
    offsets = torch.empty(num_nodes + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.randint(
        min_degree, max_degree + 1, (num_nodes,), dtype=torch.int64
    )
    offsets = offsets.cumsum(dim=0)
    num_indices = offsets[-1].item()
    indices = torch.randint(0, num_nodes, (num_indices,), dtype=torch.int64)

    graph_single_gpu = CuGraphCSC(
        offsets.to(device),
        indices.to(device),
        num_nodes,
        num_nodes,
    )
    graph_multi_gpu = CuGraphCSC(
        offsets,
        indices,
        num_nodes,
        num_nodes,
        partition_size=partition_size,
        partition_groups=partition_groups,
    )

    nfeat_single_gpu = torch.randn((num_nodes, model_kwds["input_dim_nodes"])).to(
        device=device, dtype=dtype
    )
    efeat_single_gpu = torch.randn((num_indices, model_kwds["input_dim_edges"])).to(
        device=device, dtype=dtype
    )
    nfeat_multi_gpu = nfeat_single_gpu.detach().clone()
    nfeat_multi_gpu = graph_multi_gpu.get_partioned_local_src_node_features(
        nfeat_multi_gpu
    )
    efeat_multi_gpu = efeat_single_gpu.detach().clone()
    efeat_multi_gpu = graph_multi_gpu.get_local_edge_features(efeat_multi_gpu)

    # forward + backward passes
    out_single_gpu = model_single_gpu(
        nfeat_single_gpu, efeat_single_gpu, graph_single_gpu
    )
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(nfeat_multi_gpu, efeat_multi_gpu, graph_multi_gpu)
    out_multi_gpu = graph_multi_gpu.get_global_dst_node_features(out_multi_gpu)
    loss = out_multi_gpu.sum()
    if dist.local_rank % partition_size != 0:
        loss = loss * 0

    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (1e-2, 1e-5),
        torch.bfloat16: (1e-1, 1e-3),
        torch.float16: (1e-1, 1e-3),
    }
    atol, rtol = tolerances[dtype]

    # compare forward
    diff = out_single_gpu - out_multi_gpu
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        out_single_gpu, out_multi_gpu, atol=atol, rtol=rtol
    ), f"{mask.sum()} elements have diff > {atol} \n {out_single_gpu[mask]} \n {out_multi_gpu[mask]}"

    # compare model gradients (ensure correctness of backward)
    model_multi_gpu_parameters = list(model_multi_gpu.parameters())
    for param_idx, param in enumerate(model_single_gpu.parameters()):
        diff = param - model_multi_gpu_parameters[param_idx]
        diff = torch.abs(diff)
        mask = diff > atol
        assert torch.allclose(
            param, model_multi_gpu_parameters[param_idx], atol=atol, rtol=rtol
        ), f"{mask.sum()} elements have diff > {atol} \n {param[mask]} \n {model_multi_gpu_parameters[param_idx][mask]}"

    if dist.local_rank == 0:
        print(
            f"PASSED (partition_size: {partition_size}, dtype: {dtype}, concat_trick: {do_concat_trick})"
        )

    torch.distributed.barrier()


if __name__ == "__main__":
    DistributedManager.initialize()
    test_distributed_meshgraphnet(8, torch.float32, False)
    test_distributed_meshgraphnet(8, torch.float32, True)
    test_distributed_meshgraphnet(8, torch.float16, False)
    test_distributed_meshgraphnet(8, torch.bfloat16, False)
    test_distributed_meshgraphnet(8, torch.bfloat16, True)
    test_distributed_meshgraphnet(4, torch.float32, False)
    test_distributed_meshgraphnet(2, torch.float32, False)
