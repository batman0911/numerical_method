import numpy as np
import matplotlib.pyplot as plt
from array import array
import struct
import os


def read_images_labels(images_filepath, labels_filepath):        
  labels = []
  with open(labels_filepath, 'rb') as file:
    magic, size = struct.unpack(">II", file.read(8))
    if magic != 2049:
      raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
    labels = array("B", file.read())        
      
  with open(images_filepath, 'rb') as file:
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    if magic != 2051:
      raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
    image_data = array("B", file.read())
  images = []
  for i in range(size):
    images.append([0] * rows * cols)
  for i in range(size):
    img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
    img = img.reshape(28, 28)
    images[i][:] = img            
      
  return np.array(images), np.array(labels)


def load_data_np(train_images, train_labels, test_images, test_labels):
  X_train, y_train = read_images_labels(train_images, train_labels)
  X_test, y_test = read_images_labels(test_images, test_labels)
  dim_train = y_train.shape[0]
  dim_test = y_test.shape[0]
  return X_train.reshape((dim_train, 784)), X_test.reshape((dim_test, 784)), y_train, y_test


def visualize20(X, y):
  fig, axes = plt.subplots(2, 10, figsize=(16, 6))
  for i in range(20):
    axes[i//10, i %10].imshow(X[i], cmap='gray')
    axes[i//10, i %10].axis('off')
    axes[i//10, i %10].set_title(f"target: {y[i]}")


if __name__ == '__main__':
  train_images = 'data/in/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte'
  train_labels = 'data/in/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
  test_images = 'data/in/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
  test_labels = 'data/in/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
  
  X_train, X_test, y_train, y_test = load_data_np(train_images, train_labels, test_images, test_labels)
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

