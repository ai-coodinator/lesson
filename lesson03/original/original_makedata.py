from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np

# 画像読み込み
caltech_dir = "./image"
categories = ["camera","chair","cup","dolphin","pizza","watch"]
nb_classes = len(categories)

# サイズ変換
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

# 画像データ作成
X = []
Y = []
for idx, cat in enumerate(categories):
    # カテゴリ指定
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# 学習データ作成
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/" + str(nb_classes) + "obj.npy", xy)

print("ok,", len(Y))
