import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCH = 3

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(X_train.shape, y_train.shape)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1))
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
instances, img_height, img_width, _ = X_train.shape
print(instances, img_height, img_width)
print("Numer of classes", num_classes := y_train.shape[1])
print(y_train[1])
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            input_shape=(img_height, img_width, 1),
        ),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train, epochs=EPOCH, validation_data=(X_test, y_test), batch_size=32
)
print(model.summary())
loss_value, accuracy_value = model.evaluate(X_test, y_test)
print(
f"""Loss value: {loss_value}\nAccuracy value: {accuracy_value}"""
)

print(history.__dict__.keys())
print(history.history.keys())
fig, ax = plt.subplots(1, 1, figsize=(15,15))
ax.plot(history.history['accuracy'], label="training accuracy")
ax.set_xticks(np.arange(stop=EPOCH, step=1.0))
ax.plot(history.history['val_accuracy'], label="validation accuracy")
# ax.text(EPOCH-1, 0.87, f"Loss value: {loss_value}\nAccuracy value: {accuracy_value}")
ax.set_title('model accuracy')
ax.legend(loc="lower right")
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
fig.savefig("accuracy.png")

fig, ax = plt.subplots(1, 1, figsize=(15,15))
ax.plot(history.history['loss'], label="training loss")
ax.set_xticks(np.arange(stop=EPOCH, step=1.0))
ax.plot(history.history['val_loss'], label="validation loss")
ax.set_title('model loss')
ax.legend(loc="lower right")
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
fig.savefig("loss.png")