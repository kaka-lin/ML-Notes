import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    impath = 'image/example.jpg'
    img = Image.open(impath)
    print(f'The image size is {img.size}')


    ### Affine transformation
    # 1. Translation
    # For example, we are going to translate the image
    # by 200 pixels right and 100 pixels down respectively.
    trans_matrix = np.array([[1, 0, 200], [0, 1, 100], [0, 0, 1]])
    trans_matrix = np.linalg.inv(trans_matrix)
    img_trans = img.transform(img.size, method=Image.AFFINE, data=trans_matrix.flatten())

    # 2. Scaling / Resizinq

    # In Pillow, we can use the resize method to resize the image.
    img_resized_example = img.resize((960, 640))
    print(f'The image size is {img_resized_example.size}')
    # Or we ca use the transform method with the affine matrix
    # to resize the image.
    # For example, set sx = 0.5 and sy = 0.5
    # to divide each dimension of the image by 2.
    resize_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    resize_inv = np.linalg.inv(resize_matrix)
    im_resized = img.transform((960, 640), method=Image.AFFINE, data=resize_inv.flatten())
    print(f'The image size is {im_resized.size}')

    # 3. Rotation
    # In Pillow, we can use the rotate method to rotate the image.
    img_rotated_example = img.rotate(45)
    # Or we can use the transform method with the affine matrix
    # to rotate the image.
    # For example, rotate the image by 45 degrees.
    angle = 45
    theta = np.radians(angle)
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
    rotate_inv = np.linalg.inv(rotate_matrix)
    img_rotated = img.transform(img.size, method=Image.AFFINE, data=rotate_inv.flatten())

    # 仿射變換矩陣在 PIL.Image.transform() 中是基於像素空間座標的轉換。
    # 這樣的變換通常不會考慮影像的中心點，所以默認的仿射變換是以影像的左上角為原點進行的旋轉。
    # 如果要以影像的中心點為原點進行旋轉，可以先將影像平移到原點，然後旋轉，最後再平移回去。

    # 取得影像的中心點
    cx, cy = img.size[0] / 2, img.size[1] / 2

    # 平移影像至中心，進行旋轉，然後再移回去
    translate_to_center = np.array([[1, 0, -cx],
                                    [0, 1, -cy],
                                    [0, 0, 1]])
    translate_back = np.array([[1, 0, cx],
                               [0, 1, cy],
                               [0, 0, 1]])

    # 組合平移與旋轉的仿射矩陣
    affine_matrix = translate_back @ rotate_matrix @ translate_to_center
    affine_inv = np.linalg.inv(affine_matrix)
    img_rotated_new = img.transform(img.size, method=Image.AFFINE, data=affine_inv.flatten())

    # 4. Shear
    # In Pillow, we can use the transform method with the shear matrix
    # to shear the image.
    # For example, shear the image by 0.5 in the x-direction.
    shear_matrix = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    shear_inv = np.linalg.inv(shear_matrix)
    img_sheared = img.transform(img.size, method=Image.AFFINE, data=shear_inv.flatten())

    # 5. Combination
    # We can combine multiple affine transformations by multiplying
    # the corresponding affine matrices.
    # For example, we are going to:
    # 1. translate the image by 200 pixels right and 100 pixels down.
    # 2. shear the image by 0.5 in the x-direction.
    translation = np.array([[1, 0, 200], [0, 1, 100], [0, 0, 1]])
    shearing = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    trans_matrix = translation @ shearing # matrix multiply, equal: np.matmul()
    trans_inv = np.linalg.inv(trans_matrix)
    img_combined = img.transform(img.size, method=Image.AFFINE, data=trans_inv.flatten())

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    axes[0, 0].set_title("Original")
    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")
    axes[0, 1].set_title("Translation")
    axes[0, 1].imshow(img_trans)
    axes[0, 1].axis("off")
    axes[0, 2].set_title("Resized")
    axes[0, 2].imshow(im_resized)
    axes[0, 2].axis("off")
    axes[1, 0].set_title("Rotation")
    axes[1, 0].imshow(img_rotated_new)
    axes[1, 0].axis("off")
    axes[1, 1].set_title("Shear")
    axes[1, 1].imshow(img_sheared)
    axes[1, 1].axis("off")
    axes[1, 2].set_title("Combination")
    axes[1, 2].imshow(img_combined)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig('image/example_affined.png')
    plt.show()
