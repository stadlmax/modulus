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
import time
import torch

from torch.nn.parallel import DistributedDataParallel

from utils import fix_random_seeds, create_random_input
from utils import get_icosphere_path
from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.distributed import DistributedManager
from modulus.distributed.utils import custom_allreduce_fut

icosphere_path = get_icosphere_path()


def test_distributed_graphcast(
    partition_size: int, dtype: torch.dtype, do_concat_trick: bool = True
):
    DistributedManager.create_process_subgroup("graph_partition", partition_size)
    dist_manager = DistributedManager()

    model_kwds = {
        "meshgraph_path": icosphere_path,
        "static_dataset_path": None,
        "input_dim_grid_nodes": 34,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 34,
        "processor_layers": 8,
        "hidden_dim": 32,
        "do_concat_trick": do_concat_trick,
        "use_cugraphops_encoder": True,
        "use_cugraphops_processor": True,
        "use_cugraphops_decoder": True,
    }

    device = dist_manager.local_rank
    partition = dist_manager.group_id("graph_partition")

    # initialize single GPU model for reference
    fix_random_seeds(partition)
    model_single_gpu = GraphCastNet(partition_size=1, **model_kwds).to(
        device=device, dtype=dtype
    )
    # initialze distributed model with the same seeds
    fix_random_seeds(partition)
    custom_hook = lambda process_group, bucket: custom_allreduce_fut(
        process_group,
        bucket.buffer(),
        divisor=dist_manager.num_groups("graph_partition"),
    )
    model_multi_gpu = GraphCastNet(
        partition_size=partition_size,
        partition_group_name="graph_partition",
        dist_manager=dist_manager,
        expect_partitioned_input=True,
        produce_aggregated_output=False,
        **model_kwds,
    ).to(device=device, dtype=dtype)

    model_multi_gpu = DistributedDataParallel(
        model_multi_gpu, process_group=dist_manager.group("graph_partition")
    )
    model_multi_gpu.register_comm_hook(None, custom_hook)

    # initialize data
    x_single_gpu = create_random_input(model_kwds["input_dim_grid_nodes"]).to(
        device=device, dtype=dtype
    )
    x_multi_gpu = x_single_gpu.detach().clone()
    x_multi_gpu = (
        x_multi_gpu[0]
        .view(model_multi_gpu.module.input_dim_grid_nodes, -1)
        .permute(1, 0)
    )
    x_multi_gpu = model_multi_gpu.module.g2m_graph.get_src_node_features_in_partition(
        x_multi_gpu
    )

    # forward + backward passes
    out_single_gpu = model_single_gpu(x_single_gpu)
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(x_multi_gpu)
    loss = out_multi_gpu.sum()
    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (2e-3, 1e-6),
        torch.bfloat16: (5e-2, 1e-3),
        torch.float16: (5e-2, 1e-3),
    }
    atol, rtol = tolerances[dtype]

    # compare forward, now fully materialize out_multi_gpu to faciliate comparison
    _B, _C, _N = out_multi_gpu.shape
    out_multi_gpu = out_multi_gpu.view(_C, _N).permute(1, 0)
    out_multi_gpu = model_multi_gpu.module.m2g_graph.get_global_dst_node_features(
        out_multi_gpu
    )
    out_multi_gpu = out_multi_gpu.permute(1, 0).view(out_single_gpu.shape)
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

    start_single = torch.cuda.Event(enable_timing=True)
    end_single = torch.cuda.Event(enable_timing=True)
    start_multi = torch.cuda.Event(enable_timing=True)
    end_multi = torch.cuda.Event(enable_timing=True)

    optim_single = torch.optim.Adam(
        model_single_gpu.parameters(), fused=True, foreach=True
    )
    optim_multi = torch.optim.Adam(
        model_multi_gpu.parameters(), fused=True, foreach=True
    )

    dist_manager.barrier()

    start_single.record()
    torch.cuda.nvtx.range_push(
        f"SINGLE GPU: {dtype}, {partition_size}, {do_concat_trick}"
    )
    for _ in range(20):
        optim_single.zero_grad(set_to_none=True)
        out_single_gpu = model_single_gpu(x_single_gpu)
        loss = out_single_gpu.sum()
        loss.backward()
        optim_single.step()
    torch.cuda.nvtx.range_pop()
    end_single.record()

    torch.cuda.synchronize()
    time_single = start_single.elapsed_time(end_single) / 10

    dist_manager.barrier()

    start_multi.record()
    torch.cuda.nvtx.range_push(
        f"MULTI GPU: {dtype}, {partition_size}, {do_concat_trick}"
    )
    for _ in range(20):
        optim_multi.zero_grad(set_to_none=True)
        out_multi_gpu = model_multi_gpu(x_multi_gpu)
        loss = out_multi_gpu.sum()
        loss.backward()
        optim_multi.step()
    torch.cuda.nvtx.range_pop()
    end_multi.record()

    torch.cuda.synchronize()
    time_multi = start_multi.elapsed_time(end_multi) / 10

    dist_manager.barrier()

    if dist_manager.rank == 0:
        print(
            f"PASSED (partition_size: {partition_size}, dtype: {dtype}, concat_trick: {do_concat_trick}) in {time_single} / {time_multi}"
        )

    dist_manager.cleanup_group("graph_partition")


if __name__ == "__main__":
    DistributedManager.initialize()
    manager = DistributedManager()
    assert manager.world_size > 1
    test_distributed_graphcast(manager.world_size, torch.float32, True)
    test_distributed_graphcast(manager.world_size, torch.bfloat16, True)
    test_distributed_graphcast(manager.world_size, torch.float16, True)
    test_distributed_graphcast(manager.world_size, torch.float32, False)
    test_distributed_graphcast(manager.world_size, torch.bfloat16, False)
    test_distributed_graphcast(manager.world_size, torch.float16, False)
    DistributedManager.cleanup()
