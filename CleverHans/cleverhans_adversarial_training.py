import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation="relu")
        self.conv2 = Conv2D(64, (3, 3), activation="relu")
        self.conv3 = Conv2D(64, (3, 3), activation="relu")
        self.maxpooling1 = MaxPooling2D(2,2)
        self.maxpooling2 = MaxPooling2D(2,2)

        self.flatten = Flatten()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load(
        "mnist", with_info=True, as_supervised=True
    )

    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)

data = ld_mnist()
model = Net()
loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.001)

train_loss = tf.metrics.Mean(name="train_loss")
test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    # Record operations run during the forward pass
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    # Gradient tape to automatically calculate gradients in respect to the
    # loss and trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)

    #One step of gradient descent
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# Train model with adversarial training
for epoch in range(4):
    for (x, y) in data.train:
        x =  fast_gradient_method(model, x, 0.3, np.inf)
        train_step(x, y)

for x, y in data.test:
    y_pred = model(x)
    test_acc_clean(y, y_pred)

    x_fgm = fast_gradient_method(model, x, 0.3, np.inf)
    y_pred_fgm = model(x_fgm)
    test_acc_fgsm(y, y_pred_fgm)

    x_pgd = projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf)
    y_pred_pgd = model(x_pgd)
    test_acc_pgd(y, y_pred_pgd)


print( "Clean acc:", format(test_acc_clean.result() * 100))
print("FGSM acc: ", format( test_acc_fgsm.result() * 100))
print("PGD acc: ", format( test_acc_pgd.result() * 100))




