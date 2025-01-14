import numpy as np

from ai import Perceptron
from ai import svm
from ai import gradient
from ai import neuron_math
from matplotlib import pyplot as plt

#first task
#perceptron = Perceptron(3, [2, 1])
# samples = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
# targets = [[0], [1], [0], [1], [0], [1], [0], [1]]
#
# perceptron.train(samples, targets, 30000, 1)
#
# for i in range(len(samples)):
#     print(perceptron.process(samples[i]), targets[i])
#
# print()
# layers = perceptron.layers
# for l in layers:
#     for i in l:
#         print(i.get_weights())
#     print()

#second task
# samples = []
# reader = open('samples_second_task.txt')
#
# for line in reader.readlines():
#     samples.append([float(a) for a in line.split(' ')])
# w0 = np.ones(2)
#
# def fun(w):
#     res = 0
#     for s in samples:
#         res += (s[0]*w[0] + w[1] - s[1])**2
#     return res
#
# def der(w):
#     d0 = 0
#     d1 = 0
#     for s in samples:
#         d0 += 2*(s[0]*w[0] + w[1] - s[1])*s[0]
#         d1 += 2*(s[0]*w[0] + w[1] - s[1])
#     return np.array([d0, d1])
#
# fcoef = gradient(fun, der, w0, 10**(-3))
# print(fcoef)
#
# #now lets show the results
# loss = 0
# xs = []
# ys = []
# for s in samples:
#     xs.append(s[0])
#     ys.append(s[1])
#     loss += (fcoef[0]*s[0] + fcoef[1] - s[1])**2
# print(loss/1000)
#
# x = np.linspace(-55, 55, 100)
# y = [fcoef[0]*t + fcoef[1] for t in x]
# plt.scatter(xs, ys, color='blue')
# plt.plot(x, y, color='black')
# plt.show()

#third task
# samples = []
# targets = []
# reader = open('samples_third_task.txt')
#
# for line in reader.readlines():
#     splitted = [float(a) for a in line.split(' ')]
#     if splitted[0] == 1:
#         targets.append([splitted[0]])
#     else:
#         targets.append([0.0])
#     samples.append(splitted[1:])
#
# perceptron = Perceptron(2, [1])
# perceptron.train(samples, targets, 300, 1)
#
# mistakes = 0
# for i in range(len(samples)):
#     v = perceptron.process(samples[i])[0]
#     if v < 0.5:
#         mistakes += (0.0 - targets[i][0])**2
#     else:
#         mistakes += (1.0 - targets[i][0])**2
# mistakes /= 10000
# print('Accuracy:', 1 - mistakes)
#
# print()
# layers = perceptron.layers
# for l in layers:
#     for i in l:
#         print(i.get_weights())
#     print()
#
# #lets plot the results
# px = []
# py = []
# nx = []
# ny = []
# for i in range(len(samples)):
#     if targets[i][0] == 1.0:
#         px.append(samples[i][0])
#         py.append(samples[i][1])
#     else:
#         nx.append(samples[i][0])
#         ny.append(samples[i][1])
#
# x = np.linspace(-10, 10, 1000)
# y = np.array([(-68*t + 0.38)/34 for t in x])
#
# plt.scatter(px, py, color='green')
# plt.scatter(nx, ny, color='red')
# plt.plot(x, y, color='black')
# plt.show()

#first task guess the coef style
# samples = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
# targets = [0, 0, 0, 1, 0, 1, 0, 1]
# min = [-1.0, -1.0, -1.0, -1.0]
# max = [1.0, 1.0, 1.0, 1.0]
# step = [0.5, 0.5, 0.5, 0.5]
# neurons = neuron_math(samples, targets, min, max, step)
# print([n.get_weights() for n in neurons])
#
# perceptron = Perceptron(3, [2, 1])
# perceptron.layers[0][0] = neurons[0]
# perceptron.layers[0][1] = neurons[1]
# perceptron.layers[1][0] = neurons[2]
# for i in range(len(targets)):
#     print(perceptron.process(samples[i]), targets[i])