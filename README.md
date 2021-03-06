# Deep learning with python

Python codes to the same deep learning model in numpy, pytorch, tensorflow, keras and sklearn.

## Open source deep learning libraries in python
<table>
<tr>
    <td><b> <a href="http://www.numpy.org/">numpy</a> </b></td>
    <td> The fundamental package for scientific computing with Python </td>
</tr>
<tr>
    <td><b> <a href="https://pytorch.org/">pytorch</a> </b></td>
    <td> Tensors and Dynamic neural networks in Python with strong GPU acceleration  </td>
</tr>
<tr>
    <td><b> <a href="https://www.tensorflow.org/">tensorflow</a> </b></td>
    <td> an open source software library for high performance numerical computation </td>
</tr>
<tr>
    <td><b> <a href="https://keras.io/">keras</a> </b></td>
    <td>  a high-level neural networks API and capable of running on top of TensorFlow or Theano </td>
</tr>
<tr>
    <td><b> <a href="http://scikit-learn.org/stable/#">sklearn</a> </b></td>
    <td> Machine Learning in Python, simple and efficient tools for data mining and data analysis </td>
</tr>
</table>

## General machinie learning procedure

1. Set hyper-paramerters
   - The number of epochs / batch size / learning rate
2. Load dataset
   - Training data / validation data / test data
   - Data / target
3. Setup a model
   - Sequences of layers
   - Activation functions
   - Criterion: loss function
   - Optimizer to minimize the loss
   - Accuracy
4. Train the model
   - Forward progapation
   - Backward propagation
   - Update medel parameters
5. Evaluate the trained model


## [0. Linear Regression (y = 2 x + 10)](https://github.com/nampluskr/deep-learning-with-python/tree/master/00_linear_regression)
- Model: **Linear(y = w x + b) - Mean squared error**
- Learning rate: 0.01
- Epochs: 1000

## [1. MNIST - Neural Networks - Basic](https://github.com/nampluskr/deep-learning-with-python/tree/master/01_mnist_nn_basic)
- Model: **Linear(784,200) - Sigmoid - Linear(200,10) - Softmax - Cross entropy error**
- Initialization: Xavier (mean=0, stddev=sqrt(1/n))
- Optimizer: Gradient descent method
- Learning rate: 0.01
- Epochs: 10 (batch_size: 64, shuffle)


## [2. MNIST - Neural Networks - Advanced](https://github.com/nampluskr/deep-learning-with-python/tree/master/02_mnist_nn_advanced)
- Model: **Linear(784,200) - ReLU - Linear(200,200) - ReLU - Linear(200,10) - Softmax - Cross entropy error**
- Initialization: He (mean=0, stddev=sqrt(2/n))
- Optimizer: Adam method
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64, shuffle)


## [3. MNIST - Convolutional Neural Networks](https://github.com/nampluskr/deep-learning-with-python/tree/master/03_mnist_cnn)
- Model: **Layer1 - Layer2 - Layer3 - Softmax - Cross entropy error**
  - Layer1: Convolution(N x 32 x 28 x 28) - Relu - Max Pooling(N x 32 x 14 x 14) - Dropout(0.5)
  - Layer2: Convolution(N x 64 x 14 x 14) - Relu - Max Pooling(N x 64 x 7 x 7) - Dropout(0.5)
  - Layer3: Fully connected(64 x 7 x 7, 256) - Relu - Dropout(0.5) - Fully connected(256, 10)
- Optimizer: Adam method
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64, shuffle)


## [4. Autoencoder of MNIST using neural networks](https://github.com/nampluskr/deep-learning-with-python/tree/master/04_mnist_autoencoder)

- Model: **Encoder - Decoder - Mean squared error**
  - Encoder: Linear(784, 50) - Relu
  - Decoder: Linear(50, 10) - Sigmoid
- Optimizer: Adam method
- Learning rate: 0.0001
- Epochs: 10 (batch_size: 64, shuffle)


# Tips

## How to load MNIST datasets
- Load MNIST dataset using `minit(onehot=True, flatten=False)`:
```python
import sys, os
sys.path.append(os.pardir)
import common.mnist as mnist

# Load data:
x_train, y_train, x_test, y_test = mnist.load()
```

- `load()` is defined in `data\mnist.py`.
```python
def mnist(onehot=True, flatten=True):
    mnist = pickle.load(open('..\data\mnist.pkl', 'rb'))
    x_train = mnist['train_img']/255.
    x_test  = mnist['test_img']/255.
    y_train = mnist['train_label']
    y_test  = mnist['test_label']

    if onehot:
        y_train = onehot_encode(y_train, 10)
        y_test = onehot_encode(y_test, 10)

    if not flatten:
        x_train = x_train.reshape(-1,1,28,28)
        x_test = x_test.reshape(-1,1,28,28)

    return x_train, y_train, x_test, y_test

def onehot_encode(y, size):
    y_onehot = np.zeros((y.shape[0], size), dtype=int)
    for i, row in enumerate(y_onehot):
        row[int(y[i])] = 1.0
    return y_onehot
```

- pytorch
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='../data/', train=True, download=true,
                    transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='../data/', train=False, download=true,
                    transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

- keras
```python
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)/255.
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.reshape(-1, 784)/255.
y_test = keras.utils.to_categorical(y_test, 10)
```

- tensorflow
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train.labels
x_val,   y_val   = mnist.validation.images, mnist.validation.lables
x_test,  y_test  = mnist.test.images, mnist.test.labels

for epoch in range(n_epoch):
    for i in range(n_batch):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        pass
```

- sklearn
```python
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
x_train, x_test, y_train, y_test = train_test_split(
        mnist.data/255., mnist.target, test_size=10000)
```

## How to prevent `CUDA_ERROR_OUT_OF_MEMORY`
- tesnsorflow:
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
```

- Keras:
```python
config = keras.backend.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = keras.backend.tf.Session(config=config)
```

# References
## numpy
- 신경망 첫걸음 (https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)
- 밑바닥부터 시작하는 딥러닝 (https://github.com/WegraLee/deep-learning-from-scratch)

## pytorch
- https://github.com/pytorch
- https://github.com/GunhoChoi/PyTorch-FastCampus
- https://github.com/hunkim/PyTorchZeroToAll

## tensorflow
- https://github.com/hunkim/DeepLearningZeroToAll
- 핸즈온 머신러닝 (https://github.com/rickiepark/handson-ml)
- 골빈해커의 3분 딥러닝 텐서플로맛 (https://github.com/golbin/TensorFlow-Tutorials)

## keras
- https://github.com/keras-team/keras
- 코딩셰프의 3분 딥러닝, 케라스맛 (https://github.com/jskDr/keraspp)

## sklearn
- 파이썬 라이브러리를 활용한 머신러닝 (https://github.com/rickiepark/introduction_to_ml_with_python)
