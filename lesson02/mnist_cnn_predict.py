# coding:utf-8
import os
import re
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img

def list_pictures(directory, ext='jpg|gif|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

if __name__ == "__main__":

    model = load_model('mnist_model.h5')
    for picture in list_pictures('./predict/'):
        X = []
        img = img_to_array(
            load_img(picture, target_size=(28, 28), grayscale=True))
        X.append(img)

        X = np.asarray(X)
        X = X.astype('float32')
        X = X / 255.0

        features = model.predict(X)

        print('----------')
        print(picture,'→',features.argmax())
