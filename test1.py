import pandas  as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from PIL import Image
from collections import Counter

num_colors = 8
# image = Image.open('2BIAOZHUN.png')
# pixels = list(image.getdata())
# counts = Counter(pixels)
# most_common = counts.most_common(num_colors)
# print([color for color, count in most_common])
# pixels = np.array(pixels)
# pixels = pixels.reshape((-1, 3))
# kmeans = KMeans(n_clusters=num_colors)
# kmeans.fit(pixels)
# cluster_centers = kmeans.cluster_centers_
# print(list(cluster_centers))

img = cv2.imread('2BIAOZHUN.png')
if (img is None):
    print(' Not read img. ')
# 获取初始图像的宽高比例，以便resize不改变图像比例
ratio = img.shape[0] / img.shape[1]
img_resize = cv2.resize(img, (640, int(640 * ratio)), interpolation=cv2.INTER_CUBIC)
# 将Opencv中图像默认BGR转换为通用的RGB格式
img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
(height, width, channels) = img_rgb.shape
# 将图像数据转换为需要进行Kmeans聚类的Data
img_data = img_rgb.reshape(height * width, channels)

print('[INFO] Kmeans 颜色聚类......')
# 调用sklearn中Kmeans函数
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(img_data)
# 建立颜色与标签的对应字典
color_label = {}
for i in range(len(kmeans.cluster_centers_)):
    color_label[i] = kmeans.cluster_centers_[i]
print('    颜色以及其对应的标签为： {}'.format(color_label))
# 计算聚类结果， 各颜色及其所含像素数目
color_num = {}
for m in range(len(np.unique(kmeans.labels_))):
    print(np.sum(kmeans.labels_ == m))
    print(color_label[m])  # 标签m对应的色彩
    color_num[np.sum(kmeans.labels_ == m)] = color_label[m]
print('    色彩排序前字典映射为： {}'.format(color_num))
color_num_sorted = sorted(color_num.items(), key=lambda x: x[0], reverse=True)
print('    色彩排序后字典映射为： {}'.format(color_num_sorted))
color_num_ratio = []
for i in range(len(color_num_sorted)):
    color_num_ratio.append(color_num_sorted[i][0])
color_num_ratio = color_num_ratio / np.sum(color_num_ratio)
print('    色彩数目求取比例之后： {}'.format(color_num_ratio))

print('[INFO] 显示色卡图像......')
# 创建带有色卡的图像
color_card = np.zeros(shape=(height, width + 100, 3), dtype=np.int32)
# 图像左侧区域复制源图像
for i in range(height):
    for j in range(width):
        color_card[i][j] = img_rgb[i][j]
# 图像右侧显示色卡
start = 0
for i in range(len(kmeans.cluster_centers_)):
    color = color_num_sorted[i][1]
    row_start = int(color_num_ratio[i] * height)
    # 由于前面的比例为小数，转为Int导致最后部分区域没有色彩，采用最后一种颜色进行填充
    if i == len(kmeans.cluster_centers_) - 1:
        color_card[start:, width:width + 100] = color
    color_card[start: start + row_start, width:width + 100] = color
    start += row_start

plt.imshow(color_card)
plt.show()
