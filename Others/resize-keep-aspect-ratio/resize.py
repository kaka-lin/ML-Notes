import cv2
import numpy as np
import matplotlib.pyplot as plt


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''

    h, w, _ = image.shape
    print(f"Origin shape: ({w}, {h})")
    desired_w, desired_h = size
    scale = min(desired_w/w, desired_h/h)
    print(f"Scale rarion: {scale}")

    new_w, new_h = int(w * scale), int(h * scale)
    print(f"The shape after scaled: ({new_w}, {new_h})")

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((desired_h, desired_w, 3), np.uint8) * 128

    # Put the image that after resized into the center of new image
    # 將縮放後的圖片放入新圖片的正中央
    h_start = (desired_h - new_h) // 2
    w_start = (desired_w - new_w) // 2
    new_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = image

    return new_image

if __name__ == "__main__":
    image_path = "./images/street.jpg"
    img = cv2.imread(image_path)

    new_img = letterbox_image(img, (416, 416))
    #cv2.imwrite('./images/street_resized.jpg', new_img)

    # Show the result
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    f.tight_layout()

    ax1.set_title('Original')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ax2.set_title('Resize and keep aspect ratio')
    ax2.imshow(cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    plt.show()
