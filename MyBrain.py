# coding: utf-8

from sklearn.datasets import load_digits
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
#import matplotlib.pyplot as plt

# 手書き文字認識用データセット
digits = load_digits(n_class=10)

# 平均値
avg = np.mean(digits.data)
print("average={:.3f}".format(avg))

# 平均値を閾値として2値化
for data in digits.data:
    blacks = np.where(data >= avg)
    white = np.where(data < avg)
    data[blacks] = 1
    data[white] = 0

# ラベルを配列化
labels = []
for target in digits.target:
    zero_list = np.zeros(10)
    zero_list[target] = 1
    labels.append(zero_list)


# データ
print("data={}".format(digits.data[0]))

# ターゲット
print("target={}".format(digits.target[0]))

# ラベル
print("label={}".format(labels[0]))

# 画像の表示
#plt.imshow(digits.data[0].reshape((8, 8)), cmap="binary")
#plt.show()

# ニューラルネットワークの生成
net = buildNetwork(64, 128, 10)

print(net["in"])
print(net["hidden0"])
print(net["out"])
