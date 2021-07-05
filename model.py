import tensorflow as tf

# memuat dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# normalize
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# neural
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units =10, activation=tf.nn.softmax))
    model.compile(optimizer= "adam", loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])

# training
new_model = create_model()
new_model.fit(X_train, y_train, epochs=3)

loss, acc = new_model.evaluate(X_test,y_test)
print(loss)
print(acc)

new_model.save("Model.h5")
