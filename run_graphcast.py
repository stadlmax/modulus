import os
import sys
import logging

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

from collections import deque
from modulus.distributed import (
    DistributedManager,
    mark_module_as_shared,
    ProcessGroupConfig,
    ProcessGroupNode,
)
from modulus.models.graphcast.graph_cast_net import GraphCastNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_epochs: int,
        angular_res: float, 
        data_channels: int,
        dtype: torch.dtype,
    ):   
        res_h = int(180.0 / angular_res)
        res_w = int(360.0 / angular_res)

        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.dtype = dtype
        self.inputs = torch.randn((num_samples, data_channels, res_h, res_w))
        self.targets = self.inputs.sum(dim=1, keepdim=True) + torch.randn((num_samples, data_channels, res_h, res_w)) * 0.1
        self.inputs = self.inputs.to(dtype=dtype).chunk(num_samples, dim=0)
        self.targets = self.targets.to(dtype=dtype).chunk(num_samples, dim=0)

    def __len__(self):
        return len(self.inputs) * self.num_epochs
    
    def __getitem__(self, idx):
        return (
            self.inputs[idx % self.num_samples].squeeze(0),
            self.targets[idx % self.num_samples].squeeze(0),
        )


def run(
    data_channels: int,
    angular_res: float, 
    mesh_level: int,
    processor_layers: int,
    hidden_dim: int,
    use_lat_lon_partitoning: bool,
    num_epochs: int,
    num_batches: int,
):    
    # setup distributed things
    dist_manager = DistributedManager()
    device = dist_manager.device

    try:
        dp_group_size = dist_manager.group_size("data_parallel")
    except:
        dp_group_size = 1
        
    try:
        graph_partition_size = dist_manager.group_size("model_parallel")
    except:
        graph_partition_size = 1    
    
    # definition of model and dataset
    res_h = int(180.0 / angular_res)
    res_w = int(360.0 / angular_res)
    
    model_kwds = {
        "mesh_level": mesh_level,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": data_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": data_channels,
        "processor_layers": processor_layers,
        "hidden_dim": hidden_dim,
        "do_concat_trick": True,
        "norm_type": "TELayerNorm",
        "use_cugraphops_encoder": True,
        "use_cugraphops_processor": True,
        "use_cugraphops_decoder": True,
    }
    
    torch.cuda.init()
    dataset = DummyDataset(
        num_epochs=num_epochs,
        num_samples=dp_group_size * num_batches, 
        angular_res=angular_res, 
        data_channels=data_channels, 
        dtype=torch.bfloat16
    )

    model = GraphCastNet(
        partition_size=graph_partition_size if graph_partition_size > 1 else -1, 
        partition_group_name="model_parallel" if graph_partition_size > 1 else None,
        expect_partitioned_input=False,
        produce_aggregated_output=False,
        use_lat_lon_partitioning=use_lat_lon_partitoning,
        device=dist_manager.device,
        **model_kwds
    ).to(
        device=device, dtype=torch.bfloat16
    )
    
    model.train()
    
    if dist_manager.group_rank("data_parallel") == 0:
        logger.info(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of trainable params: {num_params}")
    
    # make model and dataset compatible with DDP
    model = DistributedDataParallel(
        model,
        process_group=dist_manager.group("data_parallel"),
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
    )

    if graph_partition_size > 1:
        mark_module_as_shared(model, "model_parallel", use_fp32_reduction=False)
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=dp_group_size, 
        rank=dist_manager.group_rank("data_parallel"),
        shuffle=False
    )
    loader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False, 
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
    )

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), fused=True)

    if dist_manager.group_rank("data_parallel") == 0:
        logger.info(f"starting to run ...")

    losses = []
    for batch_id, (batch_inputs, batch_targets) in enumerate(loader):
        epoch = batch_id // (dp_group_size * num_batches)
        batch_id = batch_id % (dp_group_size * num_batches)
        torch.cuda.nvtx.range_push(f"batch-{epoch}-{batch_id}")
        batch_inputs = batch_inputs.to(device=device, non_blocking=True)
        batch_targets = batch_targets.to(device=device, non_blocking=True)
        if model.module.is_distributed:
            targets = model.module.prepare_input(
                batch_targets,
                expect_partitioned_input=False,
                global_features_on_rank_0=False,
            )
            
        else:
            targets = batch_targets

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_inputs)
        # hierarchical sum
        loss = criterion(pred, targets).sum().sum()
        loss.backward()
        losses.append(loss.detach())
        optimizer.step()
        torch.cuda.nvtx.range_pop()


    print([l.item() for l in losses])


def main(
    model_parallel_size: int,
    data_channels: int,
    angular_res: float, 
    mesh_level: int,
    processor_layers: int,
    hidden_dim: int,
    use_lat_lon_partitoning: bool,
    num_epochs: int,
    num_batches: int,
    emit_nvtx: bool,
):
    DistributedManager.initialize()
    assert DistributedManager().is_initialized() and DistributedManager().distributed

    world_size = torch.distributed.get_world_size()
    assert world_size % model_parallel_size == 0
    
    world = ProcessGroupNode("world")
    pg_config = ProcessGroupConfig(world)
    pg_config.add_node(ProcessGroupNode("data_parallel"), parent=world)
    pg_config.add_node(ProcessGroupNode("model_parallel"), parent=world)
    pg_sizes = {
        "model_parallel": model_parallel_size,
        "data_parallel": world_size // model_parallel_size,
    }
    pg_config.set_leaf_group_sizes(pg_sizes)
    DistributedManager.create_groups_from_config(
        pg_config,
        verbose=False,
    )

    with torch.autograd.profiler.emit_nvtx(record_shapes=True, enabled=emit_nvtx):
        run(
            data_channels=data_channels,
            angular_res=angular_res,
            mesh_level=mesh_level,
            processor_layers=processor_layers,
            hidden_dim=hidden_dim,
            use_lat_lon_partitoning=use_lat_lon_partitoning,
            num_epochs=num_epochs,
            num_batches=num_batches,
        )

    DistributedManager.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mp_size', type=int, default=1)
    parser.add_argument('--data_channels', type=int, default=227)
    parser.add_argument('--angular_res', type=float, default=0.25)
    parser.add_argument('--mesh_level', type=int, default=6)
    parser.add_argument('--processor_layers', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--use_lat_lon_partitoning', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_batches', type=int, default=16)
    parser.add_argument('--emit_nvtx', action="store_true")
    args = parser.parse_args()    

    main(
        model_parallel_size=args.mp_size,
        data_channels=args.data_channels,
        angular_res=args.angular_res,
        mesh_level=args.mesh_level,
        processor_layers=args.processor_layers,
        hidden_dim=args.hidden_dim,
        use_lat_lon_partitoning=args.use_lat_lon_partitoning,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        emit_nvtx=args.emit_nvtx,
    )