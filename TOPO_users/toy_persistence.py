import os
import cv2
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import torch
from topologylayer.nn.alpha_dionysus import AlphaLayer

proj_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
dims = (0, 1)
persistence_filter = AlphaLayer(maxdim=dims[-1])


def det_circle(image):
    # 使用霍夫变换检测圆
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    points = circles[0, :, :2]

    # 确保至少检测到一个圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # 打印每个圆形的中心坐标和半径
            print(f"Center: {center}, Radius: {radius}")

            # 在原图上标记圆心和边界
            cv2.circle(image, center, 1, (0, 100, 100), 3)  # 圆心
            cv2.circle(image, center, radius, (255, 0, 255), 3)  # 圆边界

    plt.figure()
    plt.imshow(image)
    plt.show()
    plt.close()

    return points


def transform_diag(data):
    new_diags = []
    diags = data[0]

    for idx, diag in enumerate(list(reversed(diags))):
        dim = len(diags) - idx - 1
        new_diag = []
        for birth_to_death in diag.numpy():
            new_diag.append((dim, tuple(birth_to_death)))
        new_diags += new_diag

    return new_diags


def get_persistence(points):
    diags = persistence_filter(points)
    diags = transform_diag(diags)

    return diags


if __name__ == "__main__":
    # 读取图片
    image = cv2.imread('./circle.tif', cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    points = det_circle(image)
    new_points = list()
    for point in points:
        x, y = point
        new_points.append([x / w, y / h])
    new_points = torch.from_numpy(np.asarray(new_points))

    diags = get_persistence(new_points)

    # save diagram image
    plt.figure()
    gd.plot_persistence_diagram(diags)
    # plt.title('Persistence Diagram')
    # plt.legend()
    plt.savefig(os.path.join('./persistence_diagram.png'))
    plt.savefig(os.path.join('./persistence_diagram.svg'))
    plt.close()

    # time.sleep(1)

    # save barcode image
    plt.figure()
    gd.plot_persistence_barcode(diags)
    # plt.title('Persistence Barcode')
    # plt.legend()
    plt.savefig(os.path.join('./persistence_barcode.png'))
    plt.savefig(os.path.join('./persistence_barcode.svg'))
    plt.close()

    print('Finish!')
