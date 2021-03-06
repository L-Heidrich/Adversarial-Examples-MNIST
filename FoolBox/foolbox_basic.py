# -*- coding: utf-8 -*-
"""FoolBox_tf_images.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rjqTkImqjaCDfvMxWFF5xRnP8QwMqTa1
"""


import tensorflow as tf


import numpy as np
import eagerpy as ep
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

model = keras.models.load_model('../CNN_model.h5')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28,1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28,1))
test_images = test_images.astype('float32') / 255

test_labels = tf.convert_to_tensor(test_labels )
test_images = tf.convert_to_tensor(test_images )

test_labels = tf.dtypes.cast(test_labels, tf.int32)

import foolbox as fb

bounds = (0, 1)
fmodel = fb.TensorFlowModel(model, bounds=bounds)

attack_fgsm = fb.attacks.LinfFastGradientAttack()
attack_pgd= fb.attacks.LinfProjectedGradientDescentAttack(steps = 20,random_start=True)
attack_df = fb.attacks.LinfDeepFoolAttack()

raw_fgsm, clipped_fgsm, is_adv_fgsm = attack_fgsm(fmodel, test_images, test_labels, epsilons=0.3)
raw_pgd, clipped_pgd, is_adv_pgd = attack_pgd(fmodel, test_images, test_labels, epsilons=0.3)
raw_df, clipped_df, is_adv_df = attack_df(fmodel,test_images,test_labels, epsilons=0.3)

predictions_clean = model.predict(test_images)
predictions_df = model.predict(clipped_df)
predictions_fgsm = model.predict(clipped_fgsm)
predictions_pgd = model.predict(clipped_pgd)

"""Accuracies"""

print("clean acc: ",fb.utils.accuracy(fmodel, test_images, test_labels))
print("df acc: ",fb.utils.accuracy(fmodel, clipped_df, labels))
print("fgsm acc: ",fb.utils.accuracy(fmodel, clipped_fgsm, test_labels))
print("Pgd acc: ",fb.utils.accuracy(fmodel, clipped_pgd, labels))

"""Visuals"""

random_index = np.random.randint(len(test_images))

pred_clean = np.argmax(predictions_clean[random_index])
pred_fgsm = np.argmax(predictions_fgsm[random_index])
pred_pgd = np.argmax(predictions_pgd[random_index])
pred_df = np.argmax(predictions_df[random_index])
 
clean_images = tf.convert_to_tensor(test_images)
clipped_df = tf.convert_to_tensor(clipped_df)
clipped_fgsm = tf.convert_to_tensor(clipped_fgsm)
clipped_pgd = tf.convert_to_tensor(clipped_pgd)
  
plot_image_clean = np.reshape(clean_images[random_index], (28,28))
plot_image_df = np.reshape(clipped_df[random_index], (28,28))
plot_image_fgsm = np.reshape(clipped_fgsm[random_index], (28,28))
plot_image_pgd = np.reshape(clipped_pgd[random_index], (28,28))

"""Perturbations"""

r_df = tf.subtract(clipped_fgsm[random_index],test_images[random_index])
r_fgsm = tf.subtract(clipped_df[random_index],test_images[random_index])
r_pgd = tf.subtract(clipped_pgd[random_index],test_images[random_index])

"""Plots"""
f, axarr = plt.subplots(2,4)
axarr[0][0].imshow(plot_image_clean, cmap='gray')
axarr[0][1].imshow(plot_image_df, cmap='gray')
axarr[0][2].imshow(plot_image_fgsm, cmap='gray')
axarr[0][3].imshow(plot_image_pgd, cmap='gray')

axarr[1][1].imshow(r_df, cmap='gray')
axarr[1][2].imshow(r_fgsm, cmap='gray')
axarr[1][3].imshow(r_pgd, cmap='gray')


axarr[0][0].title.set_text("Clean: %i" % pred_clean)
axarr[0][1].title.set_text("DeepFool: %i" %  pred_df)
axarr[0][2].title.set_text("FGSM: %i" % pred_fgsm)
axarr[0][3].title.set_text("PGD: %i" % pred_pgd)

axarr[0][1].title.set_text("DF r")
axarr[0][2].title.set_text("FGSM r" )
axarr[0][3].title.set_text("PGD r" )

plt.show()