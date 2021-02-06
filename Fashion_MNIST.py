import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SOLUTION for "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

"""
plt.imshow(x_train_full[15])
plt.show()
"""

class_names = ["T-shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Normlization
x_train_n = x_train_full/255
x_test_n = x_test/255


x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_n


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# print(model.summary())


import pydot 
#keras.utils.plot_model(model)


## Information about weights and biases of each layer
## nr is number of layer witch is checked
# nr = 1
# weigths, biases = model.layers[nr].get_weights()
# print(f"{weigths} \n {weigths.shape} \n {biases}")

model.compile(loss="sparse_categorical_crossentropy", 
                    optimizer="sgd", metrics=["accuracy"])
model_history = model.fit(x_train, y_train, epochs=30,
                            validation_data=(x_valid, y_valid))
#print(f"Params:\n{model_history.params}\nHistory:\n{model_history.history}")

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#print(model.evaluate(x_test,y_test))

X_new= x_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
y_pred = model.predict_classes(X_new)

print(np.array(class_names)[y_pred])
plt.imshow(x_test[0])
plt.imshow(x_test[1])
plt.imshow(x_test[2])