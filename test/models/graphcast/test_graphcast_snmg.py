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

from torch.nn.parallel import DistributedDataParallel

from utils import fix_random_seeds, create_random_input
from utils import get_icosphere_path
from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.distributed import DistributedManager
from modulus.distributed.utils import create_process_groups, custom_allreduce_fut

icosphere_path = get_icosphere_path()


def test_distributed_graphcast(
    partition_size: int, dtype: torch.dtype, do_concat_trick: bool = True
):
    dist = DistributedManager()

    model_kwds = {
        "meshgraph_path": icosphere_path,
        "static_dataset_path": None,
        "input_dim_grid_nodes": 2,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 2,
        "processor_layers": 3,
        "hidden_dim": 4,
        "do_concat_trick": do_concat_trick,
        "use_cugraphops_encoder": True,
        "use_cugraphops_processor": True,
        "use_cugraphops_decoder": True,
    }

    device = dist.local_rank
    partition = dist.local_rank // partition_size

    # initialize single GPU model for reference
    fix_random_seeds(partition)
    model_single_gpu = GraphCastNet(partition_size=1, **model_kwds).to(
        device=device, dtype=dtype
    )
    # initialze distributed model with the same seeds
    fix_random_seeds(partition)
    num_partitions = dist.world_size // partition_size
    partition_groups = create_process_groups(partition_size)
    custom_hook = lambda process_group, bucket: custom_allreduce_fut(
        process_group, bucket.buffer(), divisor=num_partitions
    )
    model_multi_gpu = GraphCastNet(
        partition_size=partition_size, partition_groups=partition_groups, **model_kwds
    ).to(device=device, dtype=dtype)

    model_multi_gpu = DistributedDataParallel(
        model_multi_gpu, process_group=partition_groups[partition]
    )
    model_multi_gpu.register_comm_hook(None, custom_hook)

    # initialize data
    x_single_gpu = create_random_input(model_kwds["input_dim_grid_nodes"]).to(
        device=device, dtype=dtype
    )
    x_multi_gpu = x_single_gpu.detach().clone()

    # forward + backward passes
    out_single_gpu = model_single_gpu(x_single_gpu)
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(x_multi_gpu)
    loss = out_multi_gpu.sum()
    if dist.local_rank % partition_size != 0:
        loss = loss * 0

    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (2e-3, 1e-6),
        torch.bfloat16: (5e-2, 1e-3),
        torch.float16: (5e-2, 1e-3),
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
    test_distributed_graphcast(8, torch.float32, True)
    test_distributed_graphcast(8, torch.float32, False)
    test_distributed_graphcast(8, torch.float16, True)
    test_distributed_graphcast(8, torch.bfloat16, True)
    test_distributed_graphcast(8, torch.bfloat16, False)
    test_distributed_graphcast(4, torch.float32, True)
    test_distributed_graphcast(2, torch.float32, True)
