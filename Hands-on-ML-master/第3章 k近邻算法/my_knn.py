import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io  # 图像输入输出
from skimage.color import rgb2lab, lab2rgb  # 图像通道转换
from sklearn.neighbors import KNeighborsRegressor  # KNN 回归器

# 设置工作目录
os.chdir('D:\sklearn\Hands-on-ML-master\第3章 k近邻算法\style_transfer')

# block_size表示向外扩展的层数，扩展1层即3*3
block_size = 1

def read_style_image(file_name, size=block_size):
    # 读入风格图像, 得到映射 X->Y
    # 其中X储存3*3像素格的灰度值，Y储存中心像素格的色彩值
    # 读取图像文件，设图像宽为W，高为H，得到W*H*3的RGB矩阵
    img = io.imread(file_name)

    # 将RGB矩阵转换成LAB表示法的矩阵，大小仍然是W*H*3，三维分别是L、A、B
    img = rgb2lab(img)
    # 取出图像的宽度和高度
    w, h = img.shape[:2]

    X = []
    Y = []
    # 枚举全部可能的中心点
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 保存中心点(x, y)对应窗口的灰度值L
            X.append(img[x - size : x + size + 1, y - size : y + size + 1, 0].flatten())
            # 保存中心点(x, y)对应窗口的色彩值a和b
            Y.append(img[x, y, 1:])
    return X, Y

X, Y = read_style_image('style.jpg') # 建立映射

# weights='distance'表示邻居的权重与其到样本的距离成反比
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn.fit(X, Y)

def rebuild(img, size=block_size):    
    # 将内容图像转为LAB表示
    img = rgb2lab(img) 
    w, h = img.shape[:2]
    
    # 初始化输出图像对应的矩阵
    photo = np.zeros([w, h, 3])
    # 枚举内容图像的中心点
    print('Constructing window...')
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 得到中心点(x, y)对应窗口的灰度值L
            window = img[x - size: x + size + 1, \
                y - size: y + size + 1, 0].flatten()
            X.append(window)
    X = np.array(X)

    # 用KNN回归器预测颜色
    print('Predicting...')
    pred_ab = knn.predict(X).reshape(w - 2 * size, h - 2 * size, -1)
    # 设置输出图像
    photo[:, :, 0] = img[:, :, 0]
    photo[size: w - size, size: h - size, 1:] = pred_ab
    
    # 由于最外面size层无法构造窗口，简单起见，我们直接把这些像素裁剪掉
    photo = photo[size: w - size, size: h - size, :]
    return photo

content = io.imread('input.jpg')
new_photo = rebuild(content)
# 为了展示图像，我们将其再转换为RGB表示
new_photo = lab2rgb(new_photo)

fig = plt.figure()
plt.imshow(new_photo)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
