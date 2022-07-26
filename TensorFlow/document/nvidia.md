# Install/Uninstall NVIDIA driver, CUDA Toolkit and cuDNN

## Install

- Ubuntu 18.04 (CUDA 11)

### 1. Add NVIDIA package repositories

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo apt-get update

$ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo apt-get update

$ wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
$ sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
$ sudo apt-get update
```

#### Notes

***2022/05/09 update***

Reference: [Updating the CUDA Linux GPG Repository Key](https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/)

Remove the outdated signing key

```bash
$ sudo apt-key del 7fa2af80
```

Install the `new key`, using `Ubuntu 18.04` as example

```bash
# wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
```
- Issue: an error not listed here

    ```bash
    E: Conflicting values set for option Signed-By regarding source https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /: /usr/share/keyrings/cuda-archive-keyring.gpg !=
    E: The list of sources could not be read.
    ```

- Solution: If you previously used add-apt-repository to enable the CUDA repository, then remove the duplicate entry.

    ```bash
    $ sudo sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
    ```

    Also check for and remove cuda*.list files under the /etc/apt/sources.d/ directory.


### 2. Install NVIDIA driver

```bash
# Search the nvidia-driver version
$ sudo apt search nvidia-driver

# Install nvidia-driver
$ sudo apt install --no-install-recommends nvidia-driver-470

# Reboot. Check that GPUs are visible using the command: nvidia-smi
$ sudo reboot now
```

### 3. Install development and runtime libraries (~4GB)

```bash
# $ sudo apt-get install --no-install-recommends \
#     cuda-10-0 \
#     libcudnn7=7.6.2.24-1+cuda10.0  \
#     libcudnn7-dev=7.6.2.24-1+cuda10.0

# $ sudo apt-get install --no-install-recommends \
#     cuda-11-0 \
#     libcudnn8=8.0.4.30-1+cuda11.0  \
#     libcudnn8-dev=8.0.4.30-1+cuda11.0

$ sudo apt-get install --no-install-recommends \
    cuda-11-4 \
    libcudnn8=8.2.4.15-1+cuda11.4  \
    libcudnn8-dev=8.2.4.15-1+cuda11.4
```

### 4. check CUDA and cuDNN version

1. CUDA

    ```bash
    $ cat /usr/local/cuda/version.txt

    # or

    $ nvcc --version
    ```

2. cuDNN

    ```bash
    $ cat /usr/include/cudnn_version.h  | grep CUDNN_MAJOR -A 2
    ```

## Uninstall

### 1. Uninstall NVIDIA driver

```bash
# bash
$ sudo apt remove --purge nvidia*

# zsh
$ sudo apt remove --purge "nvidia*"

# and then
$ sudo apt autoremove
```

### 2. Uninstall CUDA

```bash
# bash
$ sudo apt --purge remove cuda*

# zsh
$ sudo apt --purge remove "cuda*"

# and then
$ sudo apt autoremove
```
