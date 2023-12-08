import numpy as np
import pandas as pd
import math

#Will tidy all this up and put in a class


# define a vectorised sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

# define a vectorised tanh function
def mytanh(x):
  return math.tanh(x)

tanh_v =  np.vectorize(mytanh)

#Load matrices

matrices = np.load('GRUMatrices.npz', mmap_mode=None, allow_pickle=False, fix_imports=True)

#Get W, U snd bias matrices
wMatrix = matrices['wMatrix']
uMatrix = matrices['uMatrix']
biases = matrices['biases']

#Separate matrices into their z,r and h components  
wZ = wMatrix[:,0:6]
wR = wMatrix[:,6:12]
wH = wMatrix[:,12:18]

uZ = uMatrix[:,0:6]
uR = uMatrix[:,6:12]
uH = uMatrix[:,12:18]

rbZ = biases[0:6]
rbR = biases[6:12]
rbH = biases[12:18]

'''
ibZ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ibR = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ibH = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
'''
ibZ = biases[0,0:6]
ibR = biases[0,6:12]
ibH = biases[0,12:18]

rbZ = biases[1,0:6]
rbR = biases[1,6:12]
rbH = biases[1,12:18]

print('\nW matrices')
print(wMatrix)
print('\nwZ')
print(wZ)
print('\nwR')
print(wR)
print('\nwH')
print(wH)

print('\nU matrices')
print(uMatrix)
print('\nuZ')
print(uZ)
print('\nuR')
print(uR)
print('\nuH')
print(uH)

print('\nbias matrices')
print(biases)
print('\nibZ')
print(ibZ)
print('\nibR')
print(ibR)
print('\nibH')
print(ibH)
print('\nrbZ')
print(rbZ)
print('\nrbR')
print(rbR)
print('\nrbH')
print(rbH)

#open text file and read xValue inputs
data = np.genfromtxt('xInput.txt', delimiter=' ')

print(data.shape)
h_tminus1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(data.shape[0]):
  x_t=data[i]
  #Calculate components of GRU
  #x_t = np.array([-0.10292942821979523, -0.7129440307617188, 0.3375035226345062, 0.10583724081516266, -0.1298450380563736, 0.28884604573249817])

  z_t =  np.matmul(np.transpose(wZ),x_t) + np.matmul(np.transpose(uZ), h_tminus1) + ibZ + rbZ
  z_t = sigmoid_v(z_t)

  r_t = np.matmul(np.transpose(wR),x_t) + np.matmul(np.transpose(uR), h_tminus1) + ibR + rbR
  r_t = sigmoid_v(r_t)

  uH_update = uH
  for i in range(uH.shape[0]):
    uH_update[:,i] = uH[:,i] * r_t[i]

  h_that = np.matmul(np.transpose(wH), x_t) + np.matmul(np.transpose(uH_update), h_tminus1) + ibH + (rbH * r_t)
  h_that = tanh_v(h_that)

  #h = np.multiply(z_t, h_that) + np.multiply((1 - z_t),h_tminus1) 
  h = np.multiply((1-z_t), h_that) + np.multiply(z_t, h_tminus1) 
  h_tminus1 = h
  print(h)

#https://kaixih.github.io/keras-cudnn-rnn/
#https://gist.github.com/bzamecnik/bd3786a074f8cb891bc2a397343070f1