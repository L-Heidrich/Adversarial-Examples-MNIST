import keras.models
from keras import layers
from keras import models
from keras.models import load_model
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28,1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28,1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model = keras.models.load_model('../CNN_model.h5')

model.fit(train_images, train_labels, epochs=4, batch_size=32)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test loss:" ,test_loss)
print("Test acc:", test_acc)


from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

x_pgd = projected_gradient_descent(model, image, 0.3, 0.01, 40, np.inf)
x_fgm = fast_gradient_method(model, image, 0.3, np.inf)

for i in range(10):
#Choose a random image
    random_index = np.random.randint(10000)
    image = test_images[random_index]

    image = np.expand_dims(image, axis=0)
    pred_clean = model.predict(image)

    #Create adversarial example

    pred_fgsm = model(x_fgm)
    pred_pgd = model(x_pgd)

    #Show plot
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(np.reshape(image,(28,28)), cmap='gray')
    axarr[1].imshow(x_fgm.numpy()[0], cmap='gray')
    axarr[2].imshow(x_pgd.numpy()[0], cmap='gray')

    axarr[0].title.set_text("Clean: %i" % np.argmax(pred_clean))
    axarr[1].title.set_text("FGSM: %i" % np.argmax(pred_fgsm))
    axarr[2].title.set_text("PGD: %i" % np.argmax(pred_pgd))
    plt.show()






