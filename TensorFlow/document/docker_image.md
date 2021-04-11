# Docker image

We have already made a docker image for this repository.
If you want to build yourself docker image or get information about this image, you can refer [here](https://github.com/kaka-lin/docker-image).

## Download docker image

```bash
$ docker pull kakalin/kimage:cpu-mconda-py36-tf200
```

## Running a container

You need to mount directory into container and publish a container's port to the host.

- mount directory: 

    ```bash
    -v <localhost_directory>:<container_path>
    ```

- publish port:

    ```bash
    -p xxxx:8888 # xxxx can be any port(s), ex: 8888 or 7777 
    ```

### Example

```bash
$ docker run --rm -it kakalin/kimage:cpu-mconda-py36-tf200
```

```bash
$ docker run --rm -it -p 8888:8888 -v ~/tensorflow2-tutorials/:/root/tensorflow2-tutorials kakalin/kimage:cpu-mconda-py36-tf200
```

### In container

```bash
$ jupyter notebook --allow-root --no-browser --ip="*"
```
