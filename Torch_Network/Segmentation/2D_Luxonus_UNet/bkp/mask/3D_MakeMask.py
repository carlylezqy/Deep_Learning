import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.io as io
import os
import cv2
import time

def read_3d_images(fileDIR, filename):
    file_url = os.path.join(fileDIR, filename + ".mhd")
    data = io.imread(file_url, plugin='simpleitk')
    data = data.transpose(2, 1, 0)

    return data

def threshold_otsu(gray, min_value=0, max_value=255):

    # ヒストグラムの算出
    hist = [np.sum(gray == i) for i in range(256)]

    s_max = (0,-10)

    for th in range(256):
        
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0. : mu1 = 0.
        else : mu1 = sum([i * hist[i] for i in range(0, th)]) / n1   
        if n2 == 0. : mu2 = 0.
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]

    # 算出した閾値で二値化処理
    gray[gray < t] = min_value
    gray[gray >= t] = max_value

    return gray

a = read_3d_images("C:\CNN\Dataset\Luxonus_Data\P2-L\L0512_v100", "L0512_v100")
def getMask3d(a):
    a = np.max(a, axis=2)
    a = a.astype(np.uint16)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    a = clahe.apply(a)

    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    #a[a < 0.05] = 0

    print(np.min(a), np.max(a))

    a = threshold_otsu(a * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(a, connectivity=8)

    output = np.zeros((a.shape[0], a.shape[1], 3), np.uint8)


    for i in range(1, num_labels):
        if(stats[i, 4] > 10):
            mask = labels == i
            output[:, :, 0][mask] = 255
            output[:, :, 1][mask] = 255
            output[:, :, 2][mask] = 255

    return output


"""
cv2.imwrite("MASK.bmp", output)

plt.figure()
plt.imshow(output, cmap="gray") #vminとvmaxはデータごとに調整必要
plt.colorbar()
plt.show()


a = cv2.Canny(a, 450, 200)

a = 255 - a

cv2.imshow("opening_3(3,3).jpg", a)
cv2.imwrite(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".jpg", a)
cv2.waitKey(0)
"""