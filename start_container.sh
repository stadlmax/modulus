IMG=gitlab-master.nvidia.com:5005/mstadler/docker/modulus:25.01
# IMG=nvcr.io/nvidia/modulus/modulus:24.09

docker run \
    -it \
    --gpus all \
    --net=host \
    --uts=host \
    --ipc=host \
    --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 \
    --user $(id -u):$(id -g) \
    --ulimit=memlock=-1 \
    -e HOME="/tmp" \
    -e TRITON_HOME="/tmp" \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp" \
    -v /home/scratch.mstadler_gpu:/home/scratch.mstadler_gpu \
    $IMG /bin/bash
