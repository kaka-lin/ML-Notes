# Pytorch Learning Notes

## Run with docker

We have already made a docker image for this repository.
If you want to build yourself docker image or get information about this image, you can refer [here](https://github.com/kaka-lin/docker-image).

```bash
$ docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v $PWD/:/root/pytorch \
    -w /root/pytorch \
    kakalin/kimage:cuda11.3-torch1.12.0-devel
```

## Categories

- [Tensors](https://github.com/kaka-lin/ML-Notes/tree/master/Pytorch/tensors)
- [Datasets & DataLoaders](https://github.com/kaka-lin/ML-Notes/tree/master/Pytorch/datasets_dataloaders)
