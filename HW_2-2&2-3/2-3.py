import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)


    def fit(self, X, y):
        self._initialized_weights( X.shape[1])
        self.cost_= []
        self.cost_2=[]
        self.cost_10=[]

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []
            for xi ,target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            cost_2 = sum(cost) / 2
            cost_10 = sum(cost) / 10
            self.cost_.append(avg_cost)
            self.cost_2.append(cost_2)
            self.cost_10.append(cost_10)
        return self

    def partial_fit(self,X, y):
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi ,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self

    def _shuffle(self,X,y):
        r=np.random.permutation(len(y))
        return X[r], y[r]

    def _initialized_weights(self, m):
        self.w_=np.zeros(1+m)
        self.w_initialized=True

    def _update_weights(self,xi,target):
        output=self.net_input(xi)
        error=(target-output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost=0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
a = df.tail()
print(a)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [1, 3]].values

ada=AdalineSGD(n_iter=15,eta=0.01,random_state=1)
ada.fit(X, y)
b1=plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o',label="batch 1")
b2=plt.plot(range(1, len(ada.cost_2) + 1), ada.cost_2, marker='o',label="batch 2")
b10=plt.plot(range(1, len(ada.cost_10) + 1), ada.cost_10, marker='o',label="batch 10")

plt.xlabel("Epochs")
plt.ylabel("Averge Cost")
plt.legend()
plt.show()




