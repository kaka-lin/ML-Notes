# Transposed Convolution

Tranposed convolution 主要作用就是起到上採樣的作用。`但 transposed convolution 不是 convolution 的逆運算(一般卷積操作是不可逆的)`，它只能恢復到原來的 shape 但數值與原來的不同。

Transposed convolution 的運算步驟可以歸為以下步驟:

1. 在輸入特徵圖間填充 `s-1` 個 0
2. 在輸入特徵圖四周填充 `k-p-1` 個 0
3. 將 kernel 逆時針旋轉180度 (上下、左右翻轉)
4. 做正常卷積運算: `padding=0, stride=1`
    - k: 表示轉置卷積的 kernel_size 大小
    - p: 為轉置卷積的 padding，注意這裡的 padding 和卷積操作中有些不同

如下圖所示:

| s=1, p=0, k=3 | s=2, p=0, k=3 | s=2, p=1, k=3 |
| --- | --- | --- |
| <img src="images/transposed_conv_1.gif">  | <img src="images/transposed_conv_2.gif"> | <img src="images/transposed_conv_3.gif"> |

## Reference

1. [抽絲剝繭，帶你理解轉置卷積（反捲積）](https://blog.csdn.net/tsyccnh/article/details/87357447)
2. [轉置卷積（Transposed Convolution）](https://blog.csdn.net/qq_37541097/article/details/120709865)
1. [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
