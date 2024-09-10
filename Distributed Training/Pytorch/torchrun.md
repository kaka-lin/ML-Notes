# Pytorch - torchrun 介紹

`torchrun` 是 PyTorch 1.9 版本引入的一個命令行工具，用來啟動分散式訓練過程。與之前的 `torch.distributed.launch` 相比，`torchrun` 更簡潔且處理了許多啟動時的配置細節，例如: process group 初始化和端口管理。`torchrun` 可以很方便地在單機多卡或多機多卡的環境中啟動分散式訓練過程。

## 範例

在開始看詳細參數介紹前，讓我們先了解怎麼使用。如果你會了可以跳至 [torchrun 的常用參數介紹](#torchrun-的常用參數介紹)。

#### 1. 單機多卡訓練

如果你有一台機器並且有 4 個 GPU，可以運行以下命令來進行單機多卡分散式訓練：

```bash
$ torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=4 \
    your_training_script.py
```

另一個寫法：

```bash
$ torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    your_training_script.py
```

上面差異：

- 使用 --standalone：它會自動設置會合點，無需額外參數，非常適合簡單的單機多卡訓練。
- 不使用 --standalone：PyTorch 默認會使用 `localhost:29500` 來進行會合（如果不提供 --rdzv_endpoint），這種方式靈活度更高，尤其適合多節點訓練的場景。

#### 2. 多機多卡訓練

假設有兩台機器，每台機器有 4 個 GPU，可以這樣進行多機分散式訓練：

在第一台機器上運行：

```bash
$ torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv_endpoint="master_ip:29500" \
    your_training_script.py
```

在第二台機器上運行：

```bash
$ torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv_endpoint="master_ip:29500" \
    your_training_script.py
```

## torchrun 的常用參數介紹

#### 1. --standalone

- 說明：適用於單台機器上進行的多卡分散式訓練。這個選項會自動設置 `rdzv_backend` 和 `rdzv_endpoint`，簡化了配置，特別是在進行開發和調試的時候很方便。
- 用法： `--standalone`
- 範例：假設你有一台機器並且有 4 個 GPU，可以這樣運行分散式訓練

    ```bash
    $ torchrun --nnodes=1 --nproc_per_node=4 --standalone your_training_script.py
    ```

#### 2. --nproc_per_node

- 說明：指定每台節點上要運行的 process 數。通常這個數字對應於機器上的 GPU 數量，也就是使用的 GPU 數量。
- 用法：`--nproc_per_node=4`
- 預設值：1

#### 3. --nnodes

- 說明：指定分散式訓練中要使用的節點數量，也就是使用的機器數，通常用於多機訓練。
- 用法：`--nnodes=2` 表示使用兩台機器。
- 預設值：1（單機）

#### 4. --node_rank

- 說明：在多節點訓練時，指定每個節點的唯一標識符。不同的機器需要有不同的 node_rank。
- 用法：`--node_rank=0` 表示這是第一台機器。
- 預設值：0

#### 5. --rdzv_backend

- 說明：指定 PyTorch 使用的 rendezvous (會合) 後端。常見選擇有 `c10d`，用來初始化分散式行程組（process group）。大部分情況下使用默認值即可。
- 用法：`--rdzv_backend=c10d`
- 預設值：`c10d`

#### 6. --rdzv_endpoint

- 說明：用來指定主節點（主機）的 IP 地址和端口號，這是多機訓練中所有節點會合的端點。
- 用法：`--rdzv_endpoint="localhost:29500"` 或者指定主機的 IP 地址。
- 預設值：無

#### 7. --rdzv_id

- 說明：會合的唯一識別符號，用來區分不同的訓練任務。它的作用類似於 process group 的名字。
- 用法：`--rdzv_id=unique_training_job_name`
- 預設值：none

#### 8. --rdzv_conf

- 說明：提供 rendezvous 配置選項，用於更精細地控制會合行為。
- 用法：`--rdzv_conf=<key1>=<value1>,<key2>=<value2>`

#### 9. --max_restarts

- 說明：指定當 process group 失敗時的最大重啟次數。這在出現行程崩潰的情況下可以讓訓練過程自動嘗試重新啟動。
- 用法：`--max_restarts=3`
- 預設值：0（不重啟）

#### 10. --monitor_interval

- 說明：指定檢查行程組健康狀態的時間間隔（秒）。這個參數用來監控進程是否正常運行。
- 用法：`--monitor_interval=5`
- 預設值：5 秒

#### 11. --start_method

- 說明：指定行程的啟動方式，可以是 `spawn` 或 `fork`。通常使用 `spawn` 方法來啟動行程。
- 用法：`--start_method=spawn`
- 預設值：`spawn`

#### 12. --master_addr

- 說明：指定主節點的 IP 地址，用於多機分散式訓練時，讓其他節點能夠找到主節點並進行通信。
- 用法：`--master_addr="192.168.1.1"`
- 預設值：localhost

#### 13. --master_port

- 說明：指定主節點使用的端口號，其他節點會使用這個端口來與主節點通信。
- 用法：`--master_port=29500`
- 預設值：29500

#### 14. --tee

- 說明：將訓練的輸出同時重定向到 stdout 和日誌文件。
- 用法：`--tee=3`
- 預設值：無

#### 15. --role

- 說明：指定這個行程的角色，一般情況下使用默認值即可。
- 用法：`--role=worker`
- 預設值：`default`

## Reference

- [torchrun (Elastic Launch)](https://pytorch.org/docs/stable/elastic/run.html)
