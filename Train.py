

import os
from scipy import misc
import numpy as np
np.set_printoptions(threshold=np.nan)


def read_files(path, size, bias):
    os.chdir(path)
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    all_data = []

    for file in files:
        img = misc.imread(file)
        type(img)
        img.shape
        img = img.reshape(size, )
        img = np.append(img, bias)
        all_data.append(img)
    os.chdir(path+"//..")

    return np.array(all_data)


def get_targets(labels, desired_class):
    new_lable = []
    for n, i in enumerate(labels):
        if i == desired_class:
            new_lable.append(1)
        else:
            new_lable.append(-1)
    return np.array(new_lable)


def error_summation(x, y, w):
    error = 0
    for point, xi in enumerate(x):
        if (np.dot(x[point], w)*y[point]) <= 0:
            error += -(np.dot(x[point], w)*y[point])
    return error


def error_points(x, y, w):
    error = []
    for point, xi in enumerate(x):
        if (np.dot(x[point], w) * y[point]) <= 0:
            error.append([x[point], y[point]])
    return error


def error_number(x, y, w):
    error = []
    for point, xi in enumerate(x):
        if (np.dot(x[point], w) * y[point]) <= 0:
            error.append([x[point], y[point]])
    return len(error)


def testing(x, w):
    out = 0
    for i, xi in enumerate(x):
        out = np.dot(x[i][:-1], w[:-1])+w[-1]
    return out


def perceptron(x, y, w, eta):

    errors = error_number(x, y, w)

    while errors != 0:
        for point, xi in enumerate(x):

            if (np.dot(x[point], w)*y[point]) <= 0:
                w = w + eta*x[point]*y[point]
                print('>Summation=%.3f error point' % (error_summation(x, y, w)), error_number(x, y, w))
        errors = error_number(x, y, w)
    return w


Train_data = read_files("Train\\", 784, 1)
Test_data = read_files("Test\\", 784, 1)
Test_Labels = np.loadtxt('Test Labels.txt')
Training_Labels = np.loadtxt('Training Labels.txt')
initial_weights = np.zeros(785)
initial_weights[0] = 1
etas_dict = {}
all_dict = []
etas=[1, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5, 10 ** -6, 10 ** -7, 10 ** -8, 10 ** -9]
for etai in etas:
    dic = {}
    print("------------------eta="+str(etai))
    for i in range(10):
        print("Class==>", i)
        target_values=get_targets(Training_Labels, i)
        print("****")
        dic[i] = perceptron(Train_data, target_values, initial_weights, etai)
    etas_dict[etai] = dic

f = open("etas.txt", "w")
f.write(str(etas_dict))
f.close()
