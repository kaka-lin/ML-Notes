import os
import gzip
import six.moves.urllib as urllib

from tqdm import tqdm


def download_from_url(url, file_path):
    print(f"Downloading {url} to {file_path}")

    file_size = int(urllib.request.urlopen(
        url).info().get('Content-Length', -1))
    pbar = tqdm(total=file_size)

    def _progress(block_num, block_size, total_size):
        """callback func
        @block_num: 已經下載的資料塊
        @block_size: 資料塊的大小
        @total_size: 遠端檔案的大小
        """
        pbar.update(block_size)

    filepath, _ = urllib.request.urlretrieve(url, file_path, _progress)
    pbar.close()


def extract_gz_file(filename):
    out = filename[:-3]
    print(f"Extracting {filename} to {out}")

    with gzip.GzipFile(filename, 'rb') as input_file:
        s = input_file.read()

    with open(out, 'wb') as output_file:
        output_file.write(s)


def load_fashion_mnist():
    download_base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    resources = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # data dir
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    for resource in resources:
        url = download_base + resource
        file_path = os.path.join(data_dir, resource)
        print(f"Downloading {url}")
        download_from_url(url, file_path)
        extract_gz_file(file_path)
