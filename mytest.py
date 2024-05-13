import numpy as np
from PIL import Image

def test1():
    img = Image.open("C:\\Users\\Kecle\\Desktop\\图片3.png")
    img = img.convert('RGBA') # 修改颜色通道为RGBA
    x, y = img.size # 获得长和宽
    max_d = np.sqrt(np.power(410, 2) + np.power(205, 2)) + 50
    # 设置每个像素点颜色的透明度
    for i in range(x):
        for k in range(y):
            factor = np.sqrt(np.power(i - 205, 2) + np.power(k - 300, 2))

            factor = int((factor / max_d) * 255)
            color = img.getpixel((i, k))
            color = color[:-1] + (factor, )
            img.putpixel((i, k), color)

    img.save("C:\\Users\\Kecle\\Desktop\\mask.png") # 要保存为.PNG格式的图片才可以


def test2():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from IPython import embed

    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 随机生成（10000,）服从正态分布的数据
    data = np.random.randn(10000)
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """

    plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("频数/频率")
    # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()


def test3():
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_iris, load_digits
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import os
    from IPython import embed

    digits = load_digits()
    embed()
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    X_pca = PCA(n_components=2).fit_transform(digits.data)
    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, label="PCA")
    plt.legend()
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    # test2()
    test3()
