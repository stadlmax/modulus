#!/bin/bash

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --standalone \
    test/models/graphcast/test_graphcast_snmg.py
