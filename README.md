# Deep learning with python
Deep learning in numpy, pytorch, tensorflow, keras and sklearn.

## 0. Linear Regression
- Model: Linear(784,1) - MSE
- Criterion: Mean squared error
- Learning rate: 0.10
- Epochs: 1000

## 1. MNIST Neural Net - Basic
- Model: Linear(784,200) - Sigmoid - Linear(200,10) - Softmax - Cross entropy error
  - 1 Hidden layer (nodes = 200)
- Criterion: Cross entropy error
- Activation: sigmoid
- Initialization: Random normal distribution (or Xavier)
- Optimizer: Gradient descent method
- Learning rate: 0.01
- Epochs: 10 (batch_size: 64)


## 2. MNIST Neural Net - Advanced
- Model: Linear(784,200) - ReLU - Linear(200,200) - ReLU - Linear(200,10) - Softmax - Cross entropy error
  - 2 Hidden layer (nodes = 200, 200)
- Criterion: Cross entropy error
- Activation: relu
- Initialization: (Xavier or He)
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64)


## 3. MNIST Convolutional Neural Net
- Model: Layer1 - Layer2 - Layer3
  - Layer1: Convolution(N x 32 x 28 x 28) - Relu - Max Pooling(N x 32 x 14 x 14) - Dropout(0.5)
  - Layer2: Convolution(N x 64 x 14 x 14) - Relu - Max Pooling(N x 64 x 7 x 7) - Dropout(0.5)
  - Layer3: FC(64 x 7 x 7, 256) - Relu - Dropout(0.5) - FC(256, 10)
- Criterion: Cross entropy error
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64)


# Tips

## TO prevent `CUDA_ERROR_OUT_OF_MEMORY`

- tesnsorflow:
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
```

- Keras:
```
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
- 강화학습 첫걸음 (https://github.com/awjuliani/DeepRL-Agents)

## keras
- https://github.com/keras-team/keras
- 코딩셰프의 3분 딥러닝, 케라스맛 (https://github.com/jskDr/keraspp)

## sklearn
- 파이썬 라이브러리를 활용한 머신러닝 (https://github.com/rickiepark/introduction_to_ml_with_python)
