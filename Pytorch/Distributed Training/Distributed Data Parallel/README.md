# Distributed Data Parallel Training

Pytorch 提供了下面兩種資料平行訓練方法:

1. [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html): 僅支援`單機多 GPU`
2. [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): 可支援`多機多 GPU`

DP 的 code 寫法較為簡單，但由於 DDP 的執行速度更快，官方更推薦使用 DDP。

## DataParallel (DP)

> 僅支援`單機多 GPU`

因為 DP 模式用的是 `Parameter Server (PS)` 架構，存在負載不均衡的問題，主卡往往會成為訓練的瓶頸，因為訓練速度會比DDP模式慢一些。

優缺點:
   - 最少的 code 更改，非常易於使用。

        ```
        只要對單 GPU 的程式碼修改其中一行就可以運行了
        ```

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

## DistributedDataParallel (DDP)

> 可支援`多機多 GPU`

優缺點:
   - 與 `DataParallel` 相比，`DistributedDataParallel` 需要多一步設置，即調用 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)。

   - DDP 使用 `multi-process parallelism`，因此在 model replicas 之間沒有 GIL 爭用。
   - Model 在 DDP construction 時 broadcast，而不是在每次前向傳播時，這也有助於加快訓練速度。

> DDP is shipped with `several performance optimization technologies`. For a more in-depth explanation, refer to this [paper](http://www.vldb.org/pvldb/vol13/p3005-li.pdf) (VLDB’20).。

## Reference

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [深度學習中的分散式訓練_OPPO數智技術](https://www.gushiciku.cn/pl/gXXt/zh-tw)
