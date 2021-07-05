# How it's works
1.First import the library that we need
````
import tensorflow as tf
````
2. Load the dataset that we need
````
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
````
3. Normalize the data so the neural network can read
````
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
````
4. Create the neural network(Input layaer, hidden layer, and output layer)
````
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units =10, activation=tf.nn.softmax))
    model.compile(optimizer= "adam", loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
````
5. Train the dataset
````
new_model = create_model()
new_model.fit(X_train, y_train, epochs=3)

loss, acc = new_model.evaluate(X_test,y_test)
print(loss)
print(acc)

new_model.save("Model.h5")
````
6. Test the model that has been created
````
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
````