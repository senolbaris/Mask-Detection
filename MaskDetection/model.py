import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_validation = np.load("x_validation.npy")
y_validation = np.load("y_validation.npy")

model = Sequential()

model.add(Conv2D(filters=32,
				kernel_size=(5, 5),
				padding="same",
				activation="relu",
				input_shape=(40, 40, 1)))
model.add(MaxPooling2D(pool_size=(3, 3),
					  padding="same"))

model.add(Conv2D(filters=64,
				kernel_size=(5, 5),
				padding="same",
				activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),
					  padding="same"))

model.add(Flatten())

model.add(Dense(units=256,
				activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=128,
				activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=2,
				activation="sigmoid"))

model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics=["accuracy"])

history = model.fit(x=x_train, 
					y=y_train,
					epochs=10,
					batch_size=500,
					validation_data=(x_validation, y_validation))

'''
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
'''

model.save("mask.h5")