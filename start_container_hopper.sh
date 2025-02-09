IMG=nvcr.io/nvidia/modulus/modulus:24.09

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
    -v /home/scratch.mstadler_gpu:/home/scratch.mstadler_gpu \
    $IMG /bin/bash
