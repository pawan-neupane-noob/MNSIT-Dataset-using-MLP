import gzip
import struct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#loading mnist data set 
DATA_DIR = Path(__file__).resolve().parent

def load_images(filename):
    with gzip.open(filename, "rb") as f:

        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)

    return images.astype( np.float32)  / 255.0


def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        magic, num_labels = struct.unpack ( ">II", f.read(8))
        labels = np.frombuffer(f.read() , dtype=np.uint8)
    return labels
 

train_images = load_images (DATA_DIR / "train-images-idx3-ubyte.gz")
train_labels = load_labels (DATA_DIR / "train-labels-idx1-ubyte.gz")

test_images = load_images (DATA_DIR / "t10k-images-idx3-ubyte.gz")
test_labels = load_labels (DATA_DIR / "t10k-labels-idx1-ubyte.gz")

print("Train images :", train_images.shape)
print("Train labels :", train_labels.shape)
print("Test images  :", test_images.shape)
print("Test labels  :", test_labels.shape)  

#to see sample image
"""index = 8  

plt.imshow(test_images[index].reshape(28, 28), cmap="gray")
plt.title(f"Label: {test_labels[index]}")
plt.axis("off")
plt.show()
"""

#functions to evalaute np

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x) :
    return (x > 0).astype(np.float32)

def softmax (z) :
    z = z - np.max(z, axis=1, keepdims=True)

    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def one_hot(labels) :

    encoded = np.zeros( (labels.size, 10), dtype=np.float32)

    encoded[np.arange(labels.size), labels] = 1
    return encoded

def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# MLP neural netrok with dynamic hidden layer node's number control
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):

        self.lr = lr

        # initializing the wt and biasis
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))


    def forward(self, X):

        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)

        return self.a2
    
    def backward(self, X, y):

        m = X.shape[0]

        y_hot = one_hot(y)

        # output layer wt ,bias error
        d_z2 = self.a2 - y_hot
        dW2 = (self.a1.T @ d_z2) / m
        db2 = np.sum(d_z2, axis=0, keepdims=True) / m

        # hidden layer wt ., bias error
        d_z1 = (d_z2 @ self.W2.T) * relu_derivative(self.z1)
        dW1 = (X.T @ d_z1) / m
        db1 = np.sum(d_z1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    def cross_entropy(self, y):

        m = len(y)
        return -np.mean(np.log(self.a2[np.arange(m), y] + 1e-12))

    # ---------------- Accuracy ----------------

    def accuracy(self, X, y):

        pred = np.argmax(self.forward(X), axis=1)
        return np.mean(pred == y)

    # ---------------- Training ----------------

    def train(self, X, y, epochs=10, batch_size=64):

        losses = []

        for epoch in range(epochs):

            X, y = shuffle_data(X, y)

            for start in range(0, len(X), batch_size):

                end = start + batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            self.forward(X)
            loss = self.cross_entropy(y)
            losses.append(loss)

            acc = self.accuracy(test_images, test_labels)

            print(
                f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Test Accuracy: {acc*100:.2f}%"
            )

        return losses

# trianing the model

np.random.seed(42)

nn = NeuralNetwork(
    input_size=784,
    hidden_size=256,
    output_size=10,
     lr=0.45
)

nn.train(train_images, train_labels, epochs=10, batch_size=64)

#testing the model
test_acc = nn.accuracy(test_images, test_labels)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")


#saving the params
np.savez(
    "mnist_model.npz",
    W1=nn.W1,
    b1=nn.b1,
    W2=nn.W2,
    b2=nn.b2
)

with open("model_weights.txt", "w") as f:

    f.write("W1\n")
    np.savetxt(f, nn.W1, fmt="%8.5f")
    f.write("\n\nb1\n")
    np.savetxt(f, nn.b1, fmt="%8.5f")
    f.write("\n\nW2\n")
    np.savetxt(f, nn.W2, fmt="%8.5f")
    f.write("\n\nb2\n")
    np.savetxt(f, nn.b2, fmt="%8.5f")

print("\nModel saved.")
#testing the results
print("\nFirst 10 predictions")

pred = np.argmax(nn.forward(test_images[:10]), axis=1)

for i in range(10):
    print(
        f"Image {i:2d} | Predicted: {pred[i]} | Actual: {test_labels[i]}"
    )


