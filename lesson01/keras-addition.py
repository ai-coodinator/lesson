from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam,SGD
import numpy as np

X = np.array([[1]],dtype=np.float32)
y = np.array([[2]],dtype=np.float32)

#学習データを増やす
# X = np.array([[1],[2],[4],[5]],dtype=np.float32)
# y = np.array([[2],[4],[8],[10]],dtype=np.float32)

#学習データの精度が低い
# X = np.array([[1],[2],[4],[5]],dtype=np.float32)
# y = np.array([[2],[4],[7],[10]],dtype=np.float32)

#モデル
model = Sequential()
model.add(Dense(1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='SGD')

#学習
for i in range(100):
    model.fit(X, y, batch_size=1, nb_epoch=1, verbose=0)
    print()
    print(i+1,"回目")
    print(np.round(model.predict(X), decimals=2))
    print()

#予測
# print("----------------------------------------------")
# print("未知データ＝　",np.round(model.predict(np.array([[1501]],dtype=np.float32))))
