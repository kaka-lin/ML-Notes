# Pytorch Learning Notes

## Run with docker

We have already made a docker image for this repository.
If you want to build yourself docker image or get information about this image, you can refer [here](https://github.com/kaka-lin/docker-image).

```bash
$ docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v $PWD/:/root/ML-Notes \
    kakalin/kimage:cuda11.1-torch1.8-devel
```
