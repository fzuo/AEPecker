from os.path import abspath
import sys
import os
import tensorflow as tf
sys.path.append(abspath('.'))

import keras
import keras.backend as k
import numpy as np
import pickle


from keras.models import load_model
my_model = load_model('cifar10_ResNet32v1.h5')

with open('test_labels_cifar10_1k.pkl', 'rb') as f3:
    test_labels = pickle.load(f3)


with open('test_jsma_cifar10_after_inpainting_1k.pkl', 'rb') as f3:
    x_test_adv = pickle.load(f3)


print("\n Inpaint evaluation:\n")
print(x_test_adv.shape)

len_index = x_test_adv.shape[0]

sum_succss = 0

assert len_index == len(test_labels)
test_num = len_index

for i in range(test_num):
    y_pred = my_model.predict(x_test_adv[i][1].reshape(1,32,32,3))
    if (np.argmax(y_pred) == test_labels[i]):
        sum_succss += 1

print('No. of successful classification case: ', sum_succss)
print('Accuracy on Test AEs: ', sum_succss/test_num)


