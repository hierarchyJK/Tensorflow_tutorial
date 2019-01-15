# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2019-01-10 10:57:21
@month:一月
"""
import h5py
import numpy as np
import math
def load_dataset():
    train_dataset = h5py.File('F:\\吴恩达DL作业\\课后作业\\代码作业\\第二课第三周编程作业\\assignment3\\datasets\\train_signs.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])####your train set feature
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])####your train set labels

    test_dataset = h5py.File('F:\\吴恩达DL作业\\课后作业\\代码作业\\第二课第三周编程作业\\assignment3\\datasets\\test_signs.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])######your test set feature
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])######your test set labels

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

def convert_to_one_hot (Y,C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
        mini_batche = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batche)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        mini_batche = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batche)
    return mini_batches





