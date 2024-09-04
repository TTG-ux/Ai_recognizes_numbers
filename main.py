import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import utils

images, labels = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784)) 
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20)) 

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

ckl = 3 
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(ckl):
    print(f"Цикл №{ckl}")

    for image, label in zip (images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        # Sigmoid
        hidden = 1 / (1 + np.exp(-hidden_raw))

        # Forward propagation (to output layer)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        # Sigmoid
        output = 1 / (1 + np.exp(-output_raw))

        # Loss / Error calculation (Расчет потерь и точностей) формула из MSE(The Mean Squared Error)
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation (output layer) (Большая сложная функция)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output@np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Backpropagation (hidden layer)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    # Print same debugs info between ckl (Вывод дебагов из циклов)
    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# Custom 
main2_image = plt.imread("custom.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors
gray = lambda rgb:np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
main2_image = 1 - (gray(main2_image).astype("float32") / 255)

# Reshape
main2_image = np.reshape(main2_image, (main2_image.shape[0] * main2_image.shape[1]))

# Predict
image - np.reshape(main2_image, (-1, 1))

# Test
import random

main_image = random.choice(images)

# Predict
image = np.reshape(main_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
# Sigmoid
hidden = 1 / (1 + np.exp(-hidden_raw))

# Forward propagation (to hidden layer)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
# Sigmoid
output = 1 / (1 + np.exp(-output_raw))


plt.imshow(main_image.reshape(28, 28), cmap="Greys")
plt.title(f"Хмм, преполагаемая твоя цифра: {output.argmax()}")
plt.show()