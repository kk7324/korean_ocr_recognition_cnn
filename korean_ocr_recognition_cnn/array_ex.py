import argparse
import io
import os
import numpy as np
import tensorflow as tf
import functools, operator

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

#def op2Numpy(op):
#    with tf.Session() as sess:

#    sess = tf.InteractiveSession()
#    init = tf.initialize_all_variables()
#    sess.run(init)
#    ret = sess.run(op)
#    sess.close()

  #  return ret

#def showOperation(op):
 #   print(op2Numpy(op))




def weight_variable(shape):
    """Generates a weight variable of a given shape."""
    # mean = 0.0, standard deviation = 0.1인 정규분포, 2 * stddev를 넘는 범위는 양측 모두 제거
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    #showOperation(initial)
    return tf.Variable(initial, name='weight')

def bias_variable(shape):
    """Generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

'''
# First convolutional layer. 32 feature maps.
W_conv1 = weight_variable([5, 5, 1, 32])
#showOperation(W_conv1)

b_conv1 = bias_variable([32])
showOperation(b_conv1)



x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                       padding='SAME')
h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

# Max-pooling.
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')
'''


x1 = np.array([0.4784,0.5333,0.7059,0.7882,0.7843,0.3765,0.3961,0.6824,0.8902,0.8706,0.2314,0.1216,0.6314,1.0000,0.9020,0.3333,0.3804,0.6902,0.9098,0.8706,0.4706,0.5647,0.7529,0.8510,0.8235])
x2 = np.array([0.8275,0.7882,0.6549,0.3765,0.3137,0.9020,0.8824,0.6667,0.2784,0.2314,0.9137,1.0000,0.6863,0.2118,0.2667,0.8824,0.9294,0.7843,0.5098,0.4392,0.7843,0.8353,0.7647,0.6118,0.5216])
x3 = np.array([0.4784,0.5333,0.7059,0.7882,0.7843,0.3765,0.3961,0.6824,0.8902,0.8706,0.2314,0.1216,0.6314,1.0000,0.9020,0.3333,0.3804,0.6902,0.9098,0.8706,0.4706,0.5647,0.7529,0.8510,0.8235])
x4 = np.array([0.3569,0.1804,0.1216,0.0902,0.1647,0.4235,0.2235,0.0902,0.2157,0.3529,0.6706,0.6118,0.6314,0.6353,0.6549,0.8275,0.9059,1.0000,0.9176,0.8588,0.9216,0.9529,0.9412,0.8863,0.8275])
x5 = np.array([0.8196,0.8627,0.8588,0.8667,0.8863,0.6824,0.7608,0.8980,0.9961,0.8980,0.5804,0.5294,0.6980,0.8314,0.8314,0.3647,0.2275,0.4196,0.6000,0.7333,0.2275,0.2549,0.3098,0.4000,0.5725])
x6 = np.array([0.6196,0.4588,0.3373,0.2275,0.1804,0.7412,0.5647,0.3059,0.0745,0.2157,0.8824,0.8314,0.6392,0.4118,0.4353,0.8941,0.9725,0.8471,0.6863,0.6000,0.8706,0.8784,0.8627,0.8392,0.7608])
x7 = np.array([0.9294,0.8941,0.8196,0.7294,0.6431,0.9020,1.0000,0.8392,0.6353,0.5529,0.8196,0.8392,0.6667,0.4627,0.4980,0.7529,0.6863,0.5020,0.2784,0.4157,0.6745,0.6157,0.5412,0.4510,0.4000])
x8 = np.array([0.2549,0.3529,0.4510,0.4471,0.7412,0.2667,0.1765,0.4235,0.6824,0.8471,0.3569,0.3882,0.6471,0.8588,0.8824,0.4980,0.6353,0.8510,1.0000,0.8863,0.6588,0.7804,0.8431,0.8667,0.8549])
r1 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r2 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r3 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r4 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r5 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r6 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r7 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r8 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r9 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r10 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r11 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r12 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r13 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r14 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r15 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r16 = tf.random.truncated_normal([5,5,1], stddev=0.1)
r17= tf.random.truncated_normal([5,5,1], stddev=0.1)
r18= tf.random.truncated_normal([5,5,1], stddev=0.1)
r19= tf.random.truncated_normal([5,5,1], stddev=0.1)
r20= tf.random.truncated_normal([5,5,1], stddev=0.1)
r21= tf.random.truncated_normal([5,5,1], stddev=0.1)
r22= tf.random.truncated_normal([5,5,1], stddev=0.1)
r23= tf.random.truncated_normal([5,5,1], stddev=0.1)
r24= tf.random.truncated_normal([5,5,1], stddev=0.1,dtype=float)
#r25= tf.random.truncated_normal([5,5,1], stddev=0.1)
#r26= tf.random.truncated_normal([5,5,1], stddev=0.1)
#r27= tf.random.truncated_normal([5,5,1], stddev=0.1)
#r28= tf.random.truncated_normal([5,5,1], stddev=0.1)
k1= np.reshape(x1, (5, 5, 1))
k2= np.reshape(x2, (5, 5, 1))
k3= np.reshape(x3, (5, 5, 1))
k4= np.reshape(x4, (5, 5, 1))
k5= np.reshape(x5, (5, 5, 1))
k6= np.reshape(x6, (5, 5, 1))
k7= np.reshape(x7, (5, 5, 1))
k8= np.reshape(x8, (5, 5, 1))
z1=tf.Variable(k1,dtype=float)
z2=tf.Variable(k2,dtype=float)
z3=tf.Variable(k3,dtype=float)
z4=tf.Variable(k4,dtype=float)
z5=tf.Variable(k5,dtype=float)
z6=tf.Variable(k6,dtype=float)
z7=tf.Variable(k7,dtype=float)
z8=tf.Variable(k8,dtype=float)


#z = tf.stack([k1,k2,k3,k4,k5,k6,k7,k8,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24], axis=3)
z = tf.stack([z1,z2,z3,z4,z5,z6,z7,z8,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24], axis=3)



with tf.Session() as sess:
 #   print(result)
   # print(x)
#    result= sess.run(add)\
#    print(add)
    init = tf.initialize_all_variables()
    sess.run(init)
    result = sess.run(z)
#   st= np.array2string(result)
    print(result)
    print(z)
#    f = open("C:/Users/stork/Desktop/hangul/새파일.txt", 'w')

#    f.write(st)

    f.close

#a=weight_variable([5, 5, 1, 5])

