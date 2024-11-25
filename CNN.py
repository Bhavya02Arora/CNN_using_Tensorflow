import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare the MNIST dataset
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
print(X_train.shape)

y_train = y_train.reshape(-1,)
print(y_train[:5])

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))

    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()

# image = X_train[0]
# print(image)

plot_sample(X_train, y_train, 1)



# # Normalize pixel values to be between 0 and 1 why do we need it?
# Why do we need to normalize the pixel values?
# Normalizing the pixel values helps the model to converge faster.
# It reduces the effect of varying scales of pixel values.
# It helps to reduce the effect of illumination differences.
X_train, X_test = X_train / 255.0, X_test / 255.0


# # Build the Convolutional Neural Network
# The architecture of the CNN is:
# 1. Convolutional Layer with 32 filters and 3x3 kernel size
# 2. ReLU Activation Layer
# 3. Convolutional Layer with 64 filters and 3x3 kernel size
# 4. ReLU Activation Layer
# 5. Max Pooling Layer with 2x2 pool size
# 6. Flatten Layer
# 7. Dense Layer with 64 units
# 8. Dense Layer with 10 units (output layer)
# ann = models.Sequential([
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(3000, activation = 'relu'),
#     layers.Dense(1000, activation = 'relu'),
#     layers.Dense(10, activation= 'sigmoid')
# ])
#
# # Compile and train the model
# ann.compile(optimizer='SGD',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# print(ann.fit(X_train, y_train, epochs=5))

ann = models.Sequential([
        
        # layers.Input(shape=(32, 32, 3)),
        layers.Flatten(),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(ann.fit(X_train, y_train, epochs=5))
