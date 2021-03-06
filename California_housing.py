import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", 
            optimizer = keras.optimizers.SGD(lr=1e-3),
            metrics = ['mae'])

model_history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mae_test = model.evaluate(X_test,y_test)
model_history.history

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)

plt.show()

X_new = X_test[:3]

y_pred = model.predict(X_new)
print(y_pred)
print(y_test[:3])

del model

keras.backend.clear_session()

input_ = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation = "relu")(input_)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_], outputs = [output])

model.summary()

model.compile(loss="mean_squared_error", 
            optimizer = keras.optimizers.SGD(lr=1e-3),
            metrics = ['mae'])

model_history = model.fit(X_train, y_train, epochs=40, validation_data=(X_valid, y_valid))

mae_test = model.evaluate(X_test, y_test)

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.save("California_housing_model.h5")

del model

keras.backend.clear_session()

model = keras.models.load_model("California_housing_model.h5")

model.summary()

y_pred = model.predict(X_new)
print(y_pred)

del model
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer = keras.optimizers.SGD(lr=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint("california_model_checkpoints/Model-{epoch:02d}.h5")

history = model.fit(X_train, y_train, epochs=10, 
    validation_data=(X_valid, y_valid), 
    callbacks=[checkpoint_cb])

del model
keras.backend.clear_session()

model = keras.models.load_model("california_model_checkpoints/Model-10.h5")

mse_test = model.evaluate(X_test, y_test)

del model
keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer = keras.optimizers.SGD(lr=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint("california_model_checkpoints/Best-Model.h5", save_best_pnly=True)

history = model.fit(X_train, y_train, epochs=10, 
    validation_data=(X_valid, y_valid), 
    callbacks=[checkpoint_cb])

model = keras.models.load_model("california_model_checkpoints/Best-Model.h5")
mse_test = model.evaluate(X_test, y_test)

# Early stop project

del model
keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer = keras.optimizers.SGD(lr=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint("california_model_checkpoints/early_stop_model.h5", save_best_pnly=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights = True)

history = model.fit(X_train, y_train, epochs=200, 
    validation_data=(X_valid, y_valid), 
    callbacks=[checkpoint_cb, early_stopping_cb])

model = keras.models.load_model("california_model_checkpoints/early_stop_model.h5")
mse_test = model.evaluate(X_test, y_test)

