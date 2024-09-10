# The Introduction of PyTorch Distributed Training

PyTorch 的分散式訓練（Distributed Training）可以讓你在多個設備（例如多塊 GPU）甚至多個節點（多台機器）上訓練深度學習模型。這種方式可以加速訓練過程，尤其在處理大型數據集或模型時，能顯著減少訓練時間。

## Pytorch 分散式訓練方法

Pytorch 主要提供了以下幾種分散式訓練的方法。

### Data Parallelism (資料平行)

1. [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel): `Parameter Server (PS) 架構`

    > 因為 DP 模式用的是 `Parameter Server (PS)` 架構，存在負載不均衡的問題，主卡往往會成為訓練的瓶頸，因為訓練速度會比DDP模式慢一些。

    ```
    僅支援單台機器多 GPU，任務中只會有一個 process
    ```
    - 最少的 code 更改，非常易於使用
    - 通常無法提供最佳性能，因為它在每次前向傳遞中都會復制模型，因為`single-process multi-thread parallelism`，所以會受到 [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) 競爭的影響。

    You can easily run your operations on multiple GPUs by making your model run parallelly using `DataParallel` as below:

    ```python
    # check if we have multiple GPUs.
    # If we have multiple GPUs,
    # we can wrap our model using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Then we can put our model on GPUs by model.to(device)
    model.to(device)
    ```

2. [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): `Ring-All-Reduce 架構`

    ```
    可支援多台機器多 GPU，能夠使用 multi-process
    ```

    - 執行速度更快，官方更推薦使用
    - 因為每個 process 在自己的 GPU 上進行計算，不需要大量的跨設備通訊。
    - 與 `DataParallel` 相比，`DistributedDataParallel` 需要多一步設置，即調用 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)。
    - DDP 使用 `multi-process parallelism`，因此在 model replicas 之間沒有 `GIL 競爭`。

    > DDP is shipped with `several performance optimization technologies`. For a more in-depth explanation, refer to this [paper](http://www.vldb.org/pvldb/vol13/p3005-li.pdf) (VLDB’20).。

### Mixed Parallelism (混合平行)

- [FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)

## Pytorch 分散式訓練的基礎概念和技術

在進行實作之前，讓我先來介紹一下分散式訓練的相關基礎概念和技術。

### 基礎概念

- `行程 (Process)`： 分散式訓練的最基本單位。在分散式訓練中，每個 process 可以在不同的 GPU 或節點上運行，並且每個 process 都有自己的計算任務。通常每個 GPU 對應一個 process。

- `節點 (Node)`： 節點是指單個計算機。當進行多機分散式訓練時，每台機器都是一個節點。每個節點上可以有多個 GPU，這些 GPU 通常分配給不同的 process。

- `Rank`： 指當前 process 的序號。當多個 process 協同工作時，系統會給每個 process 分配一個 rank，通常 rank 會用來區分各 process 的角色。 例如:

    ```
    rank=0 表示為 master，通常負責匯總結果或保存模型
    ```

- `World Size`： 指的是分散式訓練中所有參與計算的 process 數量。這是全局的概念，無論是在單機多 GPU 還是多機多 GPU 的場景下，都需要設置這個參數來告知有多少個 process 在同時運行。

    - 若使用單台機器多 GPU ，world_size 表示使用的 GPU 數量

    - 若使用多台機器多 GPU ，world_size 表示使用的機器數量

    > 指 Process Group（行程組）中的 process 數量

- `Backend`： 指 process 使用的通訊後端。Backend 是 PyTorch 分散式訓練中的核心部分，決定了 process 之間如何進行通訊。常見的 backend 有:

    - `NCCL`：專為 GPU 設計的高效後端，適合多 GPU 分散式訓練。NVIDIA 推出的   `NCCL（NVIDIA Collective Communications Library）`可以在同一台或多台 GPU 之間快速傳輸數據。

    - `Gloo`：適合 CPU 和 GPU 訓練，且可以用於多機環境下的通訊。相比 NCCL，`Gloo 更通用`，但在 GPU 上的性能不如 NCCL。

    - `MPI`：`Message Passing Interface`，適合超算集群或需要高效跨節點通訊的情境。

    > 若是使用 Nvidia GPU 推薦使用 NCCL。
    > 詳細資料可參考 [Distributed communication package](https://pytorch.org/docs/stable/distributed.html)

### 核心概念：Group

1. Process Group（行程組）： 行程組是 PyTorch 中用來組織多個 process 的結構，其允許不同的 process 之間協同工作。PyTorch 中的 `torch.distributed` module 通過 process group 來管理分散式訓練中的通訊。

    行程組的初始化是通過 `init_process_group` 完成的，每個 process 需要加入某一個行程組，這樣它們才能夠進行互相之間的通信。例如，如果你有四個 GPU，這四個 GPU 會形成一個行程組，進行同步操作。

    ```python
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', world_size=4, rank=0)
    ```

2. 通訊操作： 在分散式環境中，process 之間會進行不同的通訊操作，這些操作用來將數據同步到各個節點。常見的操作有：

    - broadcast：將一個 process 的數據廣播到其他所有 process。
    - all_reduce：所有 process 各自進行計算，然後將結果進行加總或其他聚合操作，再分發回所有進程。
    - gather：將各個 process 的數據收集到一個 process 中，常用於最終的結果收集。
    - scatter：從一個 process 將數據分發到多個 process。

## Reference

- [Pytorch 分散式訓練 DistributedDataParallel — 實作篇](https://medium.com/ching-i/pytorch-%E5%88%86%E6%95%A3%E5%BC%8F%E8%A8%93%E7%B7%B4-distributeddataparallel-%E5%AF%A6%E4%BD%9C%E7%AF%87-35c762cb7e08)
