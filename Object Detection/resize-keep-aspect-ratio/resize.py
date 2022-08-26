import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import parse_annotation


def resize_image(image, size, bbox, save=True):
    print("[Reszie] Resize without keep aspect ratio")
    new_w, new_h = size
    new_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    ####################################################################
    # for resized bbox
    org_w, org_h = image.shape[1], image.shape[0]
    x1, y1, x2, y2 = bbox
    scale_w, scale_h = new_w/org_w, new_h/org_h
    resized_bbox = [int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)]
    resized_bbox_w, resized_bbox_h = resized_bbox[2] - resized_bbox[0], resized_bbox[3] - resized_bbox[1]
    # plot
    left, top, right, bottom = resized_bbox
    cv2.putText(new_image, f'{resized_bbox_w} x {resized_bbox_h}', (left, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if save:
        cv2.imwrite('data/street_resize.jpg', new_image)

    print(f"\tOrigin shape: ({org_w}, {org_h})")
    print(f"\tScale rarion: ({scale_w}, {scale_h})")
    print(f"\tThe shape after scaled: ({new_image.shape[1]}, {new_image.shape[0]})")
    print(f"\tThe shape of bbox after scaled: ({resized_bbox_w}, {resized_bbox_h})")

    return new_image


def resize_image_blob(image, size, bbox, save=True):
    print("[Reszie] Resize with keep aspect ratio - OpenCV blobFromImage()")
    new_w, new_h = size
    new_image = cv2.dnn.blobFromImage(image, scalefactor=1/255., size=size, crop=True)
    new_image = np.moveaxis(new_image[0], 0, 2)

    if save:
        new_image2 = new_image * 255
        cv2.imwrite('data/street_resize_blob.jpg', new_image2)

    print(f"\tThe shape after scaled: ({new_image.shape[1]}, {new_image.shape[0]})")

    return new_image


def letterbox_image(image, size, bbox, save=True):
    '''resize image with unchanged aspect ratio using padding'''
    print("[letterbox] Resize with keep aspect ratio")

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

    ####################################################################
    # for resized bbox
    resized_bbox = [int(box * scale) for box in bbox]
    resized_bbox_w, resized_bbox_h = resized_bbox[2] - resized_bbox[0], resized_bbox[3] - resized_bbox[1]
    # plot
    left, top, right, bottom = resized_bbox
    cv2.putText(new_image, f'{resized_bbox_w} x {resized_bbox_h}', (left, top + h_start - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if save:
        cv2.imwrite('data/street_letterbox.jpg', new_image)

    print(f"\tOrigin shape: ({w}, {h})")
    print(f"\tScale rarion: {scale}")
    print(f"\tThe shape after scaled: ({new_w}, {new_h})")
    print(f"\tThe shape of bbox after scaled: ({resized_bbox_w}, {resized_bbox_h})")

    return new_image


def letterbox_rect_image(image, size, bbox, auto=True, stride=32, save=True):
    """Resize image with keep aspect ratio using padding

    The amount of padding that was added
    Example:
      min((416, 416) / (1242, 375))
      resized_shape: (416.0, 125.60386473429952), unpad_size
      pad_size:      (0, 290.3961352657005)

    """
    print("[letterbox rect] Resize with keep aspect ratio")

    h, w, _ = image.shape
    desired_w, desired_h = size

    # 1. Scale ration: (new / old)
    scale = min(desired_w/w, desired_h/h)

    # 2. Compute padding
    new_w, new_h = int(w * scale), int(h * scale) # unpad size
    pad_w, pad_h = desired_w - new_w, desired_h - new_h
    if auto: # minimum rectangle
        # 取餘數操作，保證 padding 後的圖片是 32 (416x416) or 64(512x512) 的整數倍
        pad_w, pad_h = np.mod(pad_w, stride), np.mod(pad_h, stride)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Put the image that after resized into the center of new image
    # 將縮放後的圖片放入新圖片的正中央
    h_start = pad_h // 2
    w_start = pad_w // 2
    #new_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = image
    new_image = cv2.copyMakeBorder(image, h_start, h_start, w_start, w_start,
                                   cv2.BORDER_CONSTANT, value=(128, 128, 128))

    ####################################################################
    # for resized bbox
    resized_bbox = [int(box * scale) for box in bbox]
    resized_bbox_w, resized_bbox_h = resized_bbox[2] - resized_bbox[0], resized_bbox[3] - resized_bbox[1]
    # plot
    left, top, right, bottom = resized_bbox
    cv2.putText(new_image, f'{resized_bbox_w} x {resized_bbox_h}', (left, top + h_start - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if save:
        cv2.imwrite('data/street_letterbox_rect.jpg', new_image)

    print(f"\tOrigin shape: ({w}, {h})")
    print(f"\tScale rarion: {scale}")
    print(f"\tThe shape after scaled: ({new_image.shape[1]}, {new_image.shape[0]})")
    print(f"\tThe shape of bbox after scaled: ({resized_bbox_w}, {resized_bbox_h})")

    return new_image


if __name__ == "__main__":
    image_path = "data/street_bbox.jpg"
    annotations_path = "data/street.xml"
    img = cv2.imread(image_path)
    bbox = parse_annotation(annotations_path)

    # All methods of resizing image
    redized_img = resize_image(img, (416, 416), bbox)
    redized_img_blob = resize_image_blob(img, (416, 416), bbox)
    redized_img_letterbox = letterbox_image(img, (416, 416), bbox)
    redized_img_letterbox_rect = letterbox_rect_image(img, (416, 416), bbox)

    # Show the result
    f, axs = plt.subplots(2, 2, figsize=(10, 5))
    f.tight_layout()

    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    axs[0, 1].set_title('Resize without keep aspect ratio')
    axs[0, 1].imshow(cv2.cvtColor(redized_img, cv2.COLOR_RGB2BGR))
    axs[1, 0].set_title('Resize with keep aspect ratio')
    axs[1, 0].imshow(cv2.cvtColor(redized_img_letterbox, cv2.COLOR_RGB2BGR))
    axs[1, 1].set_title('Resize with keep aspect ratio - Rect')
    axs[1, 1].imshow(cv2.cvtColor(redized_img_letterbox_rect, cv2.COLOR_RGB2BGR))

    plt.show()
