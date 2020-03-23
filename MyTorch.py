# coding: utf-8

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# 手書き文字認識用データセット
digits = load_digits(n_class=10)

# 正規化
digits.data = digits.data / np.max(digits.data)

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
#plt.imshow(digits.data[0].reshape((8, 8)), cmap="gray")
#plt.show()

#学習用と評価用に分割
splitted_dataset = train_test_split(digits.data, labels)
#print("Train Data:{}".format(splitted_dataset[0][0]))
#print("Test Data:{}".format(splitted_dataset[1][0]))
#print("Train Label:{}".format(splitted_dataset[2][0]))
#print("Test Label:{}".format(splitted_dataset[3][0]))

# 学習用データセット
train = torch.utils.data.TensorDataset(torch.from_numpy(np.array(splitted_dataset[0])).float(), torch.from_numpy(np.array(splitted_dataset[2])).float())
train_loader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)

# 評価用データセット
test = torch.utils.data.TensorDataset(torch.from_numpy(np.array(splitted_dataset[1])).float(), torch.from_numpy(np.array(splitted_dataset[3])).float())
test_loader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)

# フィードフォワード・ネットワーク
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

# ネットワークの初期化
network = Net()

# 損失関数（交差エントロピー誤差）
criterion = nn.CrossEntropyLoss()

# オプティマイザ（確率的勾配降下法）
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    for i, data in enumerate(train_loader):

        # 学習データとラベル
        inputs, labels = data
        
        # Variable型に変換
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        # オプティマイザの初期化
        optimizer.zero_grad()

        # 入力に対する出力を取得
        outputs = network(inputs)

        print("inputs: {}".format(inputs))
        print("outputs: {}".format(outputs))
        print("labels: {}".format(labels))
        
        # 損失の取得
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("Step. {} Loss={}".format(i, loss))
        
    
