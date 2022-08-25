# Resize image, Keeping Aspect Ratio or not

在 Train model 時，前處理常常會需要將 image resize 成 model input 的 size，如 `YOLO` 的 `416x416`, `608x608` 等，這邊列舉幾種目前常見的 resize 方法，如下:

- Original image:

    ![](data/street_bbox.jpg)

- Resized without keeping aspect ratio - `cv::resize()`

    ![](data/street_resize.jpg)

- Resized with keeping aspect ratio - `letterbox_image()`

    ![](data/street_letterbox.jpg)

### 1. Resized without keeping aspect ratio - `cv::resize()`

直接對 image 進行 resize，使法改變了 image 的長寬比，image 會被拉伸。

在 [darknet - AB](https://github.com/AlexeyAB/darknet) 版本中就是使用此種前處理方式。

> image 被拉申後，對於訓練和測試效果上沒有影響，但 resized 可以使得目標尺寸變大，使得對於小目標檢測更加友好。

```python
new_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
```

![](data/street_resize.jpg)

### 2. Resized with keeping aspect ratio - `letterbox_image()`

> Resizing image, `keeping the aspect ratio consistent`, and padding the left out areas with the color (128,128,128)

像信封一樣，將 image 在保持長寬比之下縮小，並且填充到一個固定大小的盒子內。

在原始 [darknet](https://github.com/pjreddie/darknet) 中就是使用此種前處理方式。

```python
h, w, _ = image.shape
desired_w, desired_h = size
scale = min(desired_w/w, desired_h/h)
new_w, new_h = int(w * scale), int(h * scale)

image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
new_image = np.ones((desired_h, desired_w, 3), np.uint8) * 128

# Put the image that after resized into the center of new image
# 將縮放後的圖片放入新圖片的正中央
h_start = (desired_h - new_h) // 2
w_start = (desired_w - new_w) // 2
new_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = image
```

![](data/street_letterbox.jpg)

## Usage

```bash
$ python3 resize.py
```

## Reference

- [darknet-AB, Resizing : keeping aspect ratio, or not](https://github.com/AlexeyAB/darknet/issues/232)
- [YOLO網路圖像前處理的討論(resize or letterbox)](https://zhuanlan.zhihu.com/p/469436103)
