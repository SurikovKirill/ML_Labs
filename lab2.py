from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression, RANSACRegressor
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random


def rmse(y_pred, y_actual):
    return (sum([(y_pred_i - y_actual_i) ** 2 for y_pred_i, y_actual_i in zip(y_pred, y_actual)]) / len(y_actual)) ** (
            1 / 2)


def nrmse(y_pred, y_actual):
    return rmse(y_pred, y_actual) / (max(y_actual) - min(y_actual))


def batching(ent, batch_size):
    x_batching = []
    y_batching = []
    for i in range(int(len(ent) / batch_size)):
        x_batching.append(entity[32 * i][:-1])
        y_batching.append(entity[32 * i][-1])
    return x_batching, y_batching


x_train = []
y_train = []
x_test = []
y_test = []
with open("5.txt") as file_handler:
    m = int(file_handler.readline())
    n_train = int(file_handler.readline())
    for i in range(n_train):
        entity = [int(item) for item in file_handler.readline().split(' ')]
        x_train.append(entity[:-1])
        y_train.append(entity[-1])
    n_test = int(file_handler.readline())
    for i in range(n_test):
        entity = [int(item) for item in file_handler.readline().split(' ')]
        x_test.append(entity[:-1])
        y_test.append(entity[-1])

x_train_scaled = preprocessing.normalize(x_train)
x_test_scaled = preprocessing.normalize(x_test)

'''SVD'''
model = Ridge(alpha=8, solver='svd')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
error = nrmse(np.array(y_test), np.array(y_pred))
print("NRMSE for LSq " + str(error))

'''mini-batch SGD'''
iters = []
errors = []
t = np.array(y_train)
t = t.reshape((len(t), 1))
entity = np.hstack((x_train_scaled, t))
model_gradient = SGDRegressor(shuffle=True, penalty="elasticnet", alpha=0.01, learning_rate="invscaling",
                              eta0=0.0000001, l1_ratio=0.14, power_t=0.9)
for j in range(10, 1000, 10):
    iters.append(j)
    for i in range(j):
        x_btch, y_btch = batching(entity, 50)
        np.random.shuffle(entity)
        model_gradient.partial_fit(x_btch, y_btch)
    y_pred = model_gradient.predict(x_test)
    error = nrmse(y_pred, y_test)
    errors.append(error)
    print(error)

plt.plot(iters, errors, label="nrmse error for iter number")
plt.xlabel("iterations")
plt.ylabel("NRMSE")
plt.legend()
plt.show()

graph_model3 = []
graphY_model3 = []
for i in range(10, 200, 10):
    model3 = RANSACRegressor(loss='absolute_loss', max_trials=i, min_samples=200)
    model3.fit(x_train_scaled, y_train)
    y_pred = model3.predict(x_test_scaled)
    graph_model3.append(i)
    graphY_model3.append(nrmse(y_pred, y_test))
    print(nrmse(y_pred, y_test))

plt.plot(graph_model3, graphY_model3, label="nrmse error for iter number")
plt.xlabel("iterations")
plt.ylabel("NRMSE")
plt.legend()
plt.show()
