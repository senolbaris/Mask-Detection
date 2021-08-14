import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Mask ----> [0, 1]
# No mask ----> [1, 0]

def read_data(path):
	images = []
	label = []
	
	in_path = os.listdir(path)
	for folder in in_path:
		if folder == "WithMask":
			folder_path = os.path.join(path, folder)
			in_folder = os.listdir(folder_path)
			for image in in_folder:
				image_path = os.path.join(folder_path, image)
				img = cv2.imread(image_path, 0)
				img = img / 255.
				img = cv2.resize(img, (40, 40))
				label.append(1)
				images.append(img)
		elif folder == "WithoutMask":
			folder_path = os.path.join(path, folder)
			in_folder = os.listdir(folder_path)
			for image in in_folder:
				image_path = os.path.join(folder_path, image)
				img = cv2.imread(image_path, 0)
				img = img / 255.
				img = cv2.resize(img, (40, 40))
				label.append(0)
				images.append(img)

	images = np.array(images)
	images = np.reshape(images, (-1, 40, 40, 1))

	label = np.array(label)
	label = to_categorical(label, num_classes=2)

	return images, label

images_train, labels_train = read_data(r"C:\Users\Pc\Desktop\MaskDetection\Face Mask Dataset\Train")
images_test, labels_test = read_data(r"C:\Users\Pc\Desktop\MaskDetection\Face Mask Dataset\Test")
images_validation, labels_validation = read_data(r"C:\Users\Pc\Desktop\MaskDetection\Face Mask Dataset\Validation")

images_train, __, labels_train, ___ = train_test_split(images_train, labels_train, test_size=0.01, random_state=2, shuffle=True)

np.save("x_train", images_train)
np.save("y_train", labels_train)
np.save("x_test", images_test)
np.save("y_test", labels_test)
np.save("x_validation", images_validation)
np.save("y_validation", labels_validation)






