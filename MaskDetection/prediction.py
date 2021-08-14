import numpy as np
import cv2
from tensorflow import keras
import datetime
from keras.preprocessing import image
model = keras.models.load_model("mask.h5")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

def test():
	score = 0
	prediction = model.predict(x_test)
	for i in range(992):
		if np.argmax(prediction[i]) == np.argmax(y_test[i]):
			score = score + 1

	print("Score: ", score/992) # Score is 0.97580645 with test data

def process_image(image):
	image = image / 255.
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (40, 40))
	image = np.array(image)
	image = np.reshape(image, (-1, 40, 40, 1))

	return image

def predict(image):
	prediction = model.predict(image)

	return np.argmax(prediction)


