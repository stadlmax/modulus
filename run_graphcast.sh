#!/bin/bash

# ERA5-dummy
nsys profile -o "graphcast_1x8_mp1_angular0.25_mesh6" -f true \
    torchrun --nnodes 1 --nproc-per-node 8 run_graphcast.py --mp_size 1 --angular_res 0.25 --mesh_level 6
nsys profile -o "graphcast_1x8_mp2_angular0.25_mesh6" -f true \
    torchrun --nnodes 1 --nproc-per-node 8 run_graphcast.py --mp_size 2 --angular_res 0.25 --mesh_level 6

nsys profile -o "graphcast_1x8_mp1_angular0.25_mesh6_detailed" -f true \
    --cuda-memory-usage true --gpu-metrics-device all \
    torchrun --nnodes 1 --nproc-per-node 8 run_graphcast.py --mp_size 1 --angular_res 0.25 --mesh_level 6 --emit_nvtx --num_batches 16

# Next-Gen dummy
nsys profile -o "graphcast_1x8_mp4_angular0.125_mesh7" -f true \
    torchrun --nnodes 1 --nproc-per-node 8 run_graphcast.py --mp_size 4 --angular_res 0.125 --mesh_level 7