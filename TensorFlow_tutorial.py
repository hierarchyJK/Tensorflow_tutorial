# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:JACK
@file:.py
@ide:untitled3
@time:2019-01-10 10:55:33
@month:一月
"""
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from tf_utils import load_dataset,convert_to_one_hot,random_mini_batches
import matplotlib.pyplot as plt
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes =load_dataset()
index = 0
plt.imshow(X_train_orig[index])
print('y = '+ str(np.squeeze(Y_train_orig[:,index])))
plt.show()

#######处理成一列就是一个样本
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T
####normalize image vector
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

Y_train = convert_to_one_hot(Y_train_orig,6)
Y_test = convert_to_one_hot(Y_test_orig,6)

print("number of training example = " + str(X_train.shape[1]))
print("number of test example = " + str(X_test.shape[1]))
print("X_train shape = " + str(X_train.shape))
print("Y_train shape  = " + str(Y_train.shape))
print("X_test shape = " + str(X_test.shape))
print("Y_test shape = " + str(Y_test.shape))
"""linear--->relu--->linear--->relu--->linear--->softmax(softmax fit with more then two classes)"""

def create_palcehold(n_x,n_y):
    X = tf.placeholder(tf.float32,[n_x,None],name='X')
    Y = tf.placeholder(tf.float32,[n_y,None],name='Y')
    return X,Y
X,Y  = create_palcehold(12288,6)
print('X = '+str(X))
print('Y = '+str(Y))

def initialize_parameters():
    W1 = tf.get_variable('W1',shape = [25,12288],initializer=tf.random_normal_initializer(seed=1))
    b1 = tf.get_variable('b1',shape = [25,1],initializer= tf.zeros_initializer())
    W2 = tf.get_variable("W2",shape = [12,25],initializer=tf.random_normal_initializer(seed=1))
    b2 = tf.get_variable('b2',shape = [12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3',shape = [6,12],initializer=tf.random_normal_initializer(seed=1))
    b3 = tf.get_variable('b3',shape = [6,1],initializer=tf.zeros_initializer())
    parameter = {'W1':W1,
                 'b1':b1,
                 'W2':W2,
                 'b2':b2,
                 'W3':W3,
                 'b3':b3}
    return parameter
tf.reset_default_graph();
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters['W1']))
    print("b1 = " + str(parameters['b1']))
    print("W2 = " + str(parameters['W2']))
    print("b2 = " + str(parameters['b2']))
    print("W3 = " + str(parameters['W3']))
    print("b3 = " + str(parameters['b3']))
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3
tf.reset_default_graph()
with tf.Session() as sess:
    X,Y = create_palcehold(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    print('Z3 = '+str(Z3))
def computer_cost(Z3,Y):
    logist = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logist,labels=labels))
    return cost
tf.reset_default_graph()
with tf.Session() as sess:
    X,Y = create_palcehold(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = computer_cost(Z3,Y)
    print("cost = " + str(cost))

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 1500,minibatch_size = 32,print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    #####流程
    X,Y = create_palcehold(n_x,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = computer_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init  = tf.global_variables_initializer()
    print(Y_train.shape[0],X_train.shape[1])
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost /num_minibatches

            if print_cost==True and epoch % 100 == 0:
                print("Cost after epoch %i:%f"%(epoch,epoch_cost))
            if print_cost == True and epoch % 5 ==0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('interations(per tens)')
        plt.title('Learning rate = '+str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print('parameter have been trained!')
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        print("Train Acc:",accuracy.eval({X:X_train,Y:Y_train}))
        print("Test Acc",accuracy.eval({X:X_test,Y:Y_test}))

        return parameters
model(X_train,Y_train,X_test,Y_test)
