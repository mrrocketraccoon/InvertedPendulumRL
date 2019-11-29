import gym
import torch
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1  as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.framework import ops


def mdn_cost(mu, sigma, y):
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y))

def f(x):
    return x**2-6*x+9


def data_generator(x,sigma_0,samples):
    return np.random.normal(f(x),sigma_0*x,samples)

sigma_0 = 0.1
x_vals = np.arange(1,5.2,0.2)
x_arr = np.array([])
y_arr = np.array([])
samples = 50
for x in x_vals:
    #Take the previous array and stack a (50,) array with values x
    x_arr = np.append(x_arr, np.full(samples,x))
    # Take the previous array and stack a (50,) array with values y_generated
    y_arr = np.append(y_arr, data_generator(x,sigma_0,samples))

x_arr, y_arr = shuffle(x_arr, y_arr)
x_test = np.arange(1.1,5.1,0.2)

'''
fig, ax = plt.subplots(figsize=(10,10))
plt.grid(True)
plt.xlabel('x')
plt.ylabel('g(x)')
ax.scatter(x_arr,y_arr,label='sampled data')
ax.plot(x_vals,f(x_vals),c='m',label='f(x)')
ax.legend(loc='upper center',fontsize='large',shadow=True)
plt.show()
'''

epochs = 500
batch_size = 50
learning_rate = 0.0003
display_step = 50
batch_num = int(len(x_arr) / batch_size)

### NN Architecture: 2 hidden layers, each with 12 nodes and tanh(x), no activation on
### last layer.
ops.reset_default_graph()
x = tf.placeholder(name='x', shape=(None, 1), dtype=tf.float32)
y = tf.placeholder(name='y', shape=(None, 1), dtype=tf.float32)

layer = x
for _ in range(3):
    layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
mu = tf.layers.dense(inputs=layer, units=1)
sigma = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)

cost = mdn_cost(mu, sigma, y)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
x_batches = np.array_split(x_arr, batch_num)
y_batches = np.array_split(y_arr, batch_num)

'''fig, ax = plt.subplots(figsize=(10,10))
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
ax.errorbar(x_test,mu_pred,yerr=np.absolute(sigma_pred),c='r',ls='None',marker='.',ms=10,label='predicted distributions')
ax.scatter(x_arr,y_arr,c='b',alpha=0.05,label='sampled data')
ax.errorbar(x_vals,f(x_vals),yerr=sigma_0*x_vals,c='b',lw=2,ls='None',marker='.',ms=10,label='true distributions')
ax.plot(x_vals,f(x_vals),c='m',label='f(x)')
ax.legend(loc='upper center',fontsize='large',shadow=True)
plt.show()
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0.0
        x_batches, y_batches = shuffle(x_batches, y_batches)
        for i in range(batch_num):
            x_batch = np.expand_dims(x_batches[i], axis=1)
            y_batch = np.expand_dims(y_batches[i], axis=1)
            _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
            avg_cost += c / batch_num
        if epoch % display_step == 0:
            print('Epoch {0} | cost = {1:.4f}'.format(epoch, avg_cost))
    mu_pred, sigma_pred = sess.run([mu, sigma], feed_dict={x: np.expand_dims(x_test, axis=1)})
    print('Final cost: {0:.4f}'.format(avg_cost))

fig, ax = plt.subplots(figsize=(10,10))
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
ax.errorbar(x_test,mu_pred,yerr=np.absolute(sigma_pred),c='r',ls='None',marker='.',ms=10,label='predicted distributions')
ax.scatter(x_arr,y_arr,c='b',alpha=0.05,label='sampled data')
ax.errorbar(x_vals,f(x_vals),yerr=sigma_0*x_vals,c='b',lw=2,ls='None',marker='.',ms=10,label='true distributions')
ax.plot(x_vals,f(x_vals),c='m',label='f(x)')
ax.legend(loc='upper center',fontsize='large',shadow=True)
plt.show()