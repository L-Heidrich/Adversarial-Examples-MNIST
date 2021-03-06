# -*- coding: utf-8 -*-
"""Foolbox_tensorflow_plot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14-fl-cWTyzJva1fKHOQ7F5-SE-eq7tf-
"""


import tensorflow as tf
import foolbox as fb
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

bounds = (0, 1)
fmodel = fb.TensorFlowModel(model, bounds=bounds)

attack_fgsm = fb.attacks.LinfFastGradientAttack()
attack_pgd= fb.attacks.LinfProjectedGradientDescentAttack(steps = 20,random_start=True)
attack_df = fb.attacks.LinfDeepFoolAttack()

"""Multiple epsilons"""

epsilons = np.linspace(0.0, 0.3, num=5)

raw_fgsm, clipped_fgsm, is_adv_fgsm = attack_fgsm(fmodel, test_images, test_labels, epsilons=epsilons)
raw_pgd, clipped_pgd, is_adv_pgd = attack_pgd(fmodel, test_images, test_labels, epsilons=epsilons)
raw_df, clipped_df, is_adv_df = attack_df(fmodel,test_images,test_labels, epsilons=epsilons)

import eagerpy as ep

is_adv_fgsm = ep.astensor(is_adv_fgsm)
is_adv_pgd = ep.astensor(is_adv_pgd)
is_adv_df = ep.astensor(is_adv_df)

"""Mean of all adversarial examples created by a certain epsilon which cause misclassification"""

robust_accuracy_fgsm = 1 - is_adv_fgsm.float32().mean(axis=-1)
robust_accuracy_df= 1 - is_adv_pgd.float32().mean(axis=-1)
robust_accuracy_pgd = 1 - is_adv_df.float32().mean(axis=-1)

fig, axs = plt.subplots(1,3)
axs[0].plot(epsilons, robust_accuracy_fgsm.numpy())
axs[1].plot(epsilons, robust_accuracy_df.numpy())
axs[2].plot(epsilons, robust_accuracy_pgd.numpy())

axs[0].title.set_text("FGSM")
axs[1].title.set_text("DeepFool")
axs[2].title.set_text("PGD")

for ax in axs.flat:
    ax.set(xlabel='Epsilon', ylabel='Accuracy')
for ax in axs.flat:
    ax.label_outer()
plt.show()