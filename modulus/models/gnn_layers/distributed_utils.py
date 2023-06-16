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


def _all_gatherv_first_dim_fwd(
    tensor: torch.Tensor,
    sizes: List[int],
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert tensor.dim() == 2
    dim = tensor.size(1)
    # output = [
    #     torch.empty((size, dim), dtype=tensor.dtype, device=tensor.device)
    #     for size in sizes
    # ]
    # dist.all_gather(output, tensor, group=process_group)
    # return torch.cat(output, dim=0)
    global_size = sum(sizes)
    output_tensor = torch.empty(
        (global_size, dim), dtype=tensor.dtype, device=tensor.device
    )
    dist.all_gather_into_tensor(output_tensor, tensor, group=process_group)

    return output_tensor


def _gatherv_first_dim_fwd(
    tensor: torch.Tensor,
    sizes: List[int],
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert tensor.dim() == 2
    dim = tensor.size(1)
    num_global_nodes = sum(sizes)
    group_rank = dist.get_rank(process_group)
    output = (
        [
            torch.empty((size, dim), dtype=tensor.dtype, device=tensor.device)
            for size in sizes
        ]
        if group_rank == 0
        else None
    )
    dist.gather(tensor, output, dst_rank=0, group=process_group)

    return torch.cat(output, dim=0)


def _all_gatherv_first_dim_bwd(
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


def _gatherv_first_dim_bwd(
    grad_output: torch.Tensor,
    sizes: List[int],
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert grad_output.dim() == 2
    dim = grad_output.size(1)
    local_size = sizes[dist.get_rank(group=process_group)]
    group_rank = dist.get_rank(process_group)
    grad_tensor = torch.empty(
        (local_size, dim), dtype=grad_output.dtype, device=grad_output.device
    )
    scatter_list = torch.split(grad_output, sizes, dim=0) if group_rank == 0 else None
    dist.scatter(grad_tensor, scatter_list, 0, group=process_group)
    return grad_tensor


def _all_to_all_idx_first_dim_fwd(
    tensor: torch.Tensor,
    local_partition_group: dist.ProcessGroup,
    send_indices: torch.Tensor,
    recv_sizes: List[int],
    send_sizes: List[int],
    num_local_rows: int,
) -> torch.Tensor:
    total_recv_elem = sum(recv_sizes)
    local_rank = dist.get_rank(group=local_partition_group)
    global_rank = dist.get_rank()
    assert (
        total_recv_elem == num_local_rows
    ), f"Expected {num_local_rows} to receive on rank {local_rank} (global rank {global_rank}), got {total_recv_elem}."

    tensor_to_send = tensor[send_indices, :]
    tensor_to_recv = torch.empty(
        (total_recv_elem, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    dist.all_to_all_single(
        tensor_to_recv,
        tensor_to_send,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=local_partition_group,
    )

    return tensor_to_recv


def _all_to_all_idx_first_dim_bwd(
    tensor: torch.Tensor,
    local_partition_group: dist.ProcessGroup,
    send_indices: torch.Tensor,
    recv_sizes: List[int],
    send_sizes: List[int],
    num_local_rows: int,
) -> torch.Tensor:
    out = torch.empty(
        (num_local_rows, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    total_recv_elem = sum(send_sizes)

    tensor_to_send = tensor
    tensor_to_recv = torch.empty(
        (total_recv_elem, tensor.size(1)), dtype=tensor.dtype, device=tensor.device
    )

    dist.all_to_all_single(
        tensor_to_recv,
        tensor_to_send,
        output_split_sizes=send_sizes,
        input_split_sizes=recv_sizes,
        group=local_partition_group,
    )

    out.scatter_add_(
        src=tensor_to_recv,
        index=send_indices.view(-1, 1).expand(-1, tensor.size(1)),
        dim=0,
    )

    return out


class AllGathervFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        process_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.sizes = sizes
        ctx.process_group = process_group
        return _all_gatherv_first_dim_fwd(tensor, sizes, process_group)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_tensor = _all_gatherv_first_dim_bwd(
            grad_output, ctx.sizes, ctx.process_group
        )
        return grad_tensor, None, None


class GathervFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        process_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.sizes = sizes
        ctx.process_group = process_group
        return _gatherv_first_dim_fwd(tensor, sizes, process_group)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_tensor = _gatherv_first_dim_bwd(grad_output, ctx.sizes, ctx.process_group)
        return grad_tensor, None, None


class ScattervFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        process_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.sizes = sizes
        ctx.process_group = process_group
        return _gatherv_first_dim_bwd(tensor, sizes, process_group)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_tensor = _gatherv_first_dim_fwd(grad_output, ctx.sizes, ctx.process_group)
        return grad_tensor, None, None


class AllToAllIdxFirstDimAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        send_indices: torch.Tensor,
        recv_sizes: List[int],
        send_sizes: List[int],
        process_group: dist.ProcessGroup,
        size_per_partition: List[int],
    ) -> torch.Tensor:
        ctx.send_indices = send_indices
        ctx.recv_sizes = recv_sizes
        ctx.send_sizes = send_sizes
        ctx.size_per_partition = size_per_partition
        ctx.process_group = process_group

        local_rank = dist.get_rank(group=process_group)
        ctx.local_rank = local_rank
        global_rank = dist.get_rank()
        expected_rows = size_per_partition[local_rank]
        assert (
            tensor.size(0) == expected_rows
        ), f"Expected tensor.size(0) = {expected_rows} on rank {local_rank} (global rank {global_rank}), but got {tensor.size(0)} instead."

        gathered_tensor = _all_to_all_idx_first_dim_fwd(
            tensor,
            process_group,
            send_indices,
            recv_sizes,
            send_sizes,
            sum(recv_sizes),
        )
        ctx.gathered_shape = gathered_tensor.shape
        ctx.tensor_shape = tensor.shape
        return gathered_tensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_src_tensor: torch.Tensor):
        need_grad_src_tensor = ctx.needs_input_grad[0]
        local_rank = dist.get_rank(group=ctx.process_group)
        global_rank = dist.get_rank()
        num_local_rows = sum(ctx.recv_sizes)
        assert (
            grad_src_tensor.size(0) == num_local_rows
        ), f"Expected tensor.size(0) = {num_local_rows} on rank {local_rank} (global rank {global_rank}), but got {tensor.size(0)} instead."

        grad_dst_tensor = None
        if need_grad_src_tensor:
            grad_dst_tensor = _all_to_all_idx_first_dim_bwd(
                grad_src_tensor,
                ctx.process_group,
                ctx.send_indices,
                ctx.recv_sizes,
                ctx.send_sizes,
                ctx.size_per_partition[ctx.local_rank],
            )
        return grad_dst_tensor, None, None, None, None, None


def all_gatherv_first_dim(
    tensor: torch.Tensor, sizes: List[int], process_group: dist.ProcessGroup = None
) -> torch.Tensor:
    return AllGathervFirstDimAutograd.apply(tensor, sizes, process_group)


def gatherv_first_dim(
    tensor: torch.Tensor, sizes: List[int], process_group: dist.ProcessGroup = None
) -> torch.Tensor:
    return GathervFirstDimAutograd.apply(tensor, sizes, process_group)


def scatterv_first_dim(
    tensor: torch.Tensor, sizes: List[int], process_group: dist.ProcessGroup = None
) -> torch.Tensor:
    return ScattervFirstDimAutograd.apply(tensor, sizes, process_group)


def all_to_all_idx_first_dim(
    tensor: torch.Tensor,
    send_indices: torch.Tensor,
    recv_sizes: List[int],
    send_sizes: List[int],
    process_group: dist.ProcessGroup,
    size_per_partition: List[int],
) -> torch.Tensor:
    return AllToAllIdxFirstDimAutograd.apply(
        tensor, send_indices, recv_sizes, send_sizes, process_group, size_per_partition
    )
