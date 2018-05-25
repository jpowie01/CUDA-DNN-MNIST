CUDA DNN MNIST
--------------

This project is an example implementation for training simple feed forward neural network
 on a MNIST dataset in pure C++ CUDA code.

**NOTE:** This project is still under development and was created only for fun and to pass
 CUDA project on my University. Do not use it for anything except experiments - I cannot
 guarantee that it will work! :)

Feel free to fork, implement and experiment on your own!

## How to run?

To compile and run this project, all you need is Make and NVCC. Then, simply run:

```bash
$> make dataset
$> make build
$> make run
```

**Example output**:

```
Epoch 0:
  - Train Loss=3.21774
  - Train Accuracy=1.97817%
  - Test Loss=nan
  - Test Accuracy=2.84745%

Epoch 1:
  - Train Loss=3.17043
  - Train Accuracy=2.00154%
  - Test Loss=nan
  - Test Accuracy=3.05304%

...

Epoch 99:
  - Train Loss=1.50781
  - Train Accuracy=82.80081%
  - Test Loss=nan
  - Test Accuracy=94.37705%
```

## What have I done?

I've implemented basic object oriented framework with:
 - Dense Layers,
 - ReLU Layers,
 - Binary Cross Entropy Loss Function,
 - SGD Optimizer,
 - Sequential Model.

Framework API was inspired by PyTorch and TensorFlow with a little bit of Keras, mixed into
 an implementation that is ready to extend and experiment easily (I hope so...). What's more,
 all linear algebra and math calculations were implemented by myself - I didn't use cuDNN or
 cuBLAS just for training and due to University project restrictions.

Due to limited amount of time I haven't implemented any regularization method, dropout layer or
 any other optimizer. If I find enough time, I'll definitely try to do this :)

## Configuration

Training parameters can be configured via environment variables like this:

```
$> export NUMBER_OF_EPOCHS=500
$> export BATCH_SIZE=128
$> export LEARNING_RATE=1e-6
$> ...
$> make run
```

Default values that will be used if above variables were not defined can be found in
 `src/configuration.h` file.

## Experiments/Benchmarks

**Will be done in the near future...**
