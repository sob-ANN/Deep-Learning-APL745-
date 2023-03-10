import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import scipy
import scipy.io
from matplotlib import pyplot as plt

# %% Load data

data = scipy.io.loadmat("train_images.mat")
X_train = data['train_images']
lab = scipy.io.loadmat("train_labels.mat")
y_train = lab['train_labels']

data1 = scipy.io.loadmat("test_images.mat")
X_test = data1['test_images']
lab1 = scipy.io.loadmat("test_labels.mat")
y_test = lab1['test_labels']


y_train = y_train.reshape(y_train.shape[1])
y_test = y_test.reshape(y_test.shape[1])

# %% Model definition
conv = Conv3x3(8)                   # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)                # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # Transform the grayscale image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with.
  image_ = image/255 - 0.5
  out = conv.forward(image_)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Compute cross-entropy loss and accuracy.
  loss = np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0
  
  return out, loss, acc
  
def train(im, label, lr=.005):
  '''
  A training step on the given image and label.
  Shall return the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

# %% Training the model for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(X_train))
  X_train = X_train[permutation]
  y_train = y_train[permutation]

  # Train!
  loss = 0
  num_correct = 0

  for i, (im, label) in enumerate(zip(X_train, y_train)):
    if i % 100 == 0:
      print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i, loss / 100, num_correct))
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l 
    num_correct += acc 
    
# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(X_test, y_test):
  _, l, acc = forward(im, label)
  loss +=l
  num_correct += acc

num_tests = len(X_test)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

# %% Plotting
''' Plot some of the images with predicted and actual labels'''



