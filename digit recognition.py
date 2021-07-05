import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('Model.h5')

for x in range(1,6):
	img = cv2.imread(f'{x}.png')[:,:,0]
	img = np.invert(np.array([img]))
	prediction = model.predict(img)
	print("Berikut hasil dari gambar tersebut : ", np.argmax(prediction))
	plt.imshow(img[0], cmap=plt.cm.binary)
	plt.show()
