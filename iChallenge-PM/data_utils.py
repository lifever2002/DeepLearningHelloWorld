import cv2
import random
import numpy as np
import os


def data_preload(img):
    # 将图片尺寸缩放为 224x224
    img = cv2.resize(img, (224, 224))
    # [H, W, C] -> [C, H, W]
    img = np.transpose(img, (2, 0, 1))
    # 将数据类型转换为 float32
    img = img.astype("float32")

    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.0
    img = img * 2.0 - 1.0
    return img


def data_loader(datadir, batch_size=10, mode="train"):
    filenames = os.listdir(datadir)

    # 内部函数，生成器 yield，用于逐批次生成数据
    def reader():
        # 训练时随机打乱数据顺序
        if mode == "train":
            random.shuffle(filenames)

        # 数据读取和预处理
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = data_preload(img)

            # 标签分配
            if name[0] == "H" or name[0] == "N":
                # 高度近视(H)和正常视力(N)的样本，负样本，标签为0
                label = 0
            elif name[0] == "P":
                # 病理性近视(P)，正样本，标签为1
                label = 1
            else:
                raise ("Not excepted file name")

            # 数据存储和批次生成
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype("float32")
                labels_array = np.array(batch_labels).reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        # 处理剩余数据
        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype("float32")
            labels_array = np.array(batch_labels).reshape(-1, 1)
            yield imgs_array, labels_array

    return reader
