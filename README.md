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
=====================================
            Configuration
=====================================
 NumberOfEpochs: 100
 BatchSize: 512
 LearningRate: 1.000000e-06
=====================================

=====================================
         CUDA Configuration
=====================================
 Device name: GeForce GTX 1060 6GB
 Memory Clock Rate (KHz): 4004000
 Memory Bus Width (bits): 192
-------------------------------------
 Tensor2DAddBlockSize: 8
 Tensor2DSubtractBlockSize: 8
 Tensor2DScaleBlockSize: 8
 Tensor2DMultiplyBlockSize: 8
 Tensor2DMultiplyBlockNumber: -1
 Tensor2DMeanBlockSize: 8
-------------------------------------
 ReLuBlockSize: 8
-------------------------------------
 CrossEntropyGetMetricBlockSize: 64
 CrossEntropyCalculateBlockSize: 64
=====================================

Epoch 0:
  - [Train] Loss=3.25744
  - [Train] Accuracy=2.03158%
  - [Train] Total Forward Time=137.54800ms
  - [Train] Total Backward Time=282.41300ms
  - [Train] Batch Forward Time=1.17562ms
  - [Train] Batch Backward Time=2.41379ms
  - [Test] Loss=nan
  - [Test] Accuracy=2.64186%

Epoch 1:
  - [Train] Loss=3.19937
  - [Train] Accuracy=1.93142%
  - [Train] Total Forward Time=117.23500ms
  - [Train] Total Backward Time=242.62400ms
  - [Train] Batch Forward Time=1.00201ms
  - [Train] Batch Backward Time=2.07371ms
  - [Test] Loss=nan
  - [Test] Accuracy=2.69326%

...

Epoch 99:
  - [Train] Loss=1.51196
  - [Train] Accuracy=78.76603%
  - [Train] Total Forward Time=118.42700ms
  - [Train] Total Backward Time=245.76100ms
  - [Train] Batch Forward Time=1.01220ms
  - [Train] Batch Backward Time=2.10052ms
  - [Test] Loss=nan
  - [Test] Accuracy=93.28741%
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

Default values can be found in `src/configuration.h` file.

## Experiments/Benchmarks

**Will be done in the near future...**
