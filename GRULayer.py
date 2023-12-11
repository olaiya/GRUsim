import numpy as np
import pandas as pd
import math

#Will tidy all this up and put in a class
class GRULayer():

  def __init__(self, inputFileName='xInput.txt',  matricesFileName='GRUMatrices.npz'):
    self.setInputFile(inputFileName)
    self.matricesFilename = matricesFileName
    self.h_tminus1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    self.verboseFlag = False 
    self.loadMatrices()
    self.setMatrices()
    self.sigmoid_v = np.vectorize(self.sigmoid)
    self.tanh_v =  np.vectorize(self.mytanh)

  def loadMatrices(self, matricesFileName='GRUMatrices.npz'):
    #Load matrices
    self.matrices = np.load(matricesFileName, mmap_mode=None, allow_pickle=False, fix_imports=True)

  def setInputFile(self, inputFileName):
    self.inputFileName = inputFileName
    #open text file and read xValue inputs
    self.data = np.genfromtxt(self.inputFileName, delimiter=' ')

  # define a vectorised sigmoid function
  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  # define a vectorised tanh function
  def mytanh(self, x):
    return math.tanh(x)

  

  def setVerboseFlag(self, vbFlag):
    self.verboseFlag = vbFlag

  def setMatrices(self):
    #Get W, U snd bias matrices
    wMatrix = self.matrices['wMatrix']
    uMatrix = self.matrices['uMatrix']
    biases = self.matrices['biases']

    #Separate matrices into their z,r and h components  
    self.wZ = wMatrix[:,0:6]
    self.wR = wMatrix[:,6:12]
    self.wH = wMatrix[:,12:18]

    self.uZ = uMatrix[:,0:6]
    self.uR = uMatrix[:,6:12]
    self.uH = uMatrix[:,12:18]

    self.rbZ = biases[0:6]
    self.rbR = biases[6:12]
    self.rbH = biases[12:18]

    self.ibZ = biases[0,0:6]
    self.ibR = biases[0,6:12]
    self.ibH = biases[0,12:18]

    self.rbZ = biases[1,0:6]
    self.rbR = biases[1,6:12]
    self.rbH = biases[1,12:18]

    if (self.verboseFlag):

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

  def run(self):
    for i in range(self.data.shape[0]):
      x_t=self.data[i]
      #Calculate components of GRU
      z_t =  np.matmul(np.transpose(self.wZ),x_t) + np.matmul(np.transpose(self.uZ), self.h_tminus1) + self.ibZ + self.rbZ
      z_t = self.sigmoid_v(z_t)

      r_t = np.matmul(np.transpose(self.wR),x_t) + np.matmul(np.transpose(self.uR), self.h_tminus1) + self.ibR + self.rbR
      r_t = self.sigmoid_v(r_t)

      uH_update = np.copy(self.uH)
      for j in range(self.uH.shape[0]):
        uH_update[:,j] = self.uH[:,j] * r_t[j]
        

      h_that = np.matmul(np.transpose(self.wH), x_t) + np.matmul(np.transpose(uH_update), self.h_tminus1) + self.ibH + (self.rbH * r_t)
      h_that = self.tanh_v(h_that)

      #h = np.multiply(z_t, h_that) + np.multiply((1 - z_t),h_tminus1) 
      h = np.multiply((1-z_t), h_that) + np.multiply(z_t, self.h_tminus1) 
      self.h_tminus1 = h
      print(h)

if __name__ == '__main__':
  print('In the Main')

  myGRULayer = GRULayer()
  myGRULayer.run()


#https://kaixih.github.io/keras-cudnn-rnn/
#https://gist.github.com/bzamecnik/bd3786a074f8cb891bc2a397343070f1