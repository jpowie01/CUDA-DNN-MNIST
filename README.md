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

```bash
$> export NUMBER_OF_EPOCHS=500
$> export BATCH_SIZE=128
$> export LEARNING_RATE=1e-6
$> ...
$> make run
```

Default values can be found in `src/configuration.hpp` file.

## Experiments/Benchmarks

To run my experiments please use `run.py` script from `experiments` directory like this:

```bash
$> python3 experiments/run.py --logs-dir="logs_100/GTX 1060"
```

If you want to run above experiments on one of your multi-GPU setup, remember to use
`CUDA_VISIBLE_DEVICES` variable like this:

```bash
$> CUDA_VISIBLE_DEVICES=1 python3 experiments/run.py --logs-dir="logs_100/GTX 780 Ti"
```

**NOTE:** All experiment logs that I've collected are available in `experiments/logs.zip` file.

### Introduction

Below experiments mesure execution time on GPU for forward and backward passes over three similar
neural networks. All of them have a single hidden layer with different number of neurons:

```cpp
SequentialModel* model = new SequentialModel(optimizer, loss);
model->addLayer(new DenseLayer(28*28, 100));
model->addLayer(new ReLuLayer(100));
model->addLayer(new DenseLayer(100, 10));
```

Networks were labelled as:

 - Small Network - 100 neurons in hidden layer,
 - Medium Network - 200 neurons in hidden layer,
 - Large Network - 400 neurons in hidden layer.

All experiments were performed on three Nvidia GPUs:

 - GTX 780 Ti,
 - GTX 1060,
 - GTX Titan X.

Once all logs were collected to one directory, I've prepared a Jupyter Notebook that helped me
to generate plots you can find above. Notebook is available in `experiments` directory and requires
`matplotlib` with `pandas`.

### Experiment #1 - Different Batch Size

In this experiment I've tried to apply several different Batch Sizes to measure time that was
needed for forward and backward pass over a network. All of below experiments used fixed Number
of Threads per Block (equal to 16) and dynamic Number of Blocks (enough to calculate output matrix
with final result).

It's not a big surprise that larger batch size = smaller execution time due to less communication
between host and device.

#### Small Network - Forward Pass

![Small Network](assets/experiment1_small_forward_first.png)
![Small Network](assets/experiment1_small_forward_second.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment1_medium_forward_first.png)
![Medium Network](assets/experiment1_medium_forward_second.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment1_large_forward_first.png)
![Large Network](assets/experiment1_large_forward_second.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment1_small_backward_first.png)
![Small Network](assets/experiment1_small_backward_second.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment1_medium_backward_first.png)
![Medium Network](assets/experiment1_medium_backward_second.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment1_large_backward_first.png)
![Large Network](assets/experiment1_large_backward_second.png)

### Experiment #2 - Different number of Threads per Block

In this experiment I've applied a few different number of Threads per Block. During this
experiment Batch Size was fixed to 128 images, while number of Blocks was calculated
dynamically (similarly to previous experiment).

#### Small Network - Forward Pass

![Small Network](assets/experiment2_small_forward.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment2_medium_forward.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment2_large_forward.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment2_small_backward.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment2_medium_backward.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment2_large_backward.png)

### Experiment #3 - Different Number of Blocks

In this experiment I've tried several different Number of Blocks, while Number of Threads
was fixed and equals 16. Also, Batch Size was fixed to 128 images.

#### Small Network - Forward Pass

![Small Network](assets/experiment3_small_forward.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment3_medium_forward.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment3_large_forward.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment3_small_backward.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment3_medium_backward.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment3_large_backward.png)

### Experiment #4 - Matrix multiplication using Shared Memory

This experiment shows time needed to execute forward and backward pass on different GPUs
using Shared Memory as an optimization for matrix multiplication. I've tried to apply several
different values for Batch Size, while Number of Threads was fixed and equal 16.

If you compare below values with Experiment #1, you can see a significant boost in
execution time (at least 2x)!

#### Small Network - Forward Pass

![Small Network](assets/experiment4_small_forward.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment4_medium_forward.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment4_large_forward.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment4_small_backward.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment4_medium_backward.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment4_large_backward.png)

### Experiment #5 - Trying to find best values - Without Shared Memory

In this experiment I've tried to combine all best values from above plots and
find optimum combination of execution parameters. Below experiments tries to find
parameters for different Batch Sizes and uses implementation **without** Shared Memory.

#### Small Network - Forward Pass

![Small Network](assets/experiment5_small_forward.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment5_medium_forward.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment5_large_forward.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment5_small_backward.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment5_medium_backward.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment5_large_backward.png)

### Experiment #6 - Trying to find best values - With Shared Memory

This experiment is very similar to the one above. This one tries to find best values for
implementation of matrix multiplication using Shared Memory. Also, Number of Blocks is
calculated dynamically as it cannot be changed in my implementation.

On below charts you can find some strange behavious which shows that olders GPUs are faster
than newer. I didn't find the root cause for this one. I bet that this has to be somehow
connected with memory coalescing as it was only found during backward pass (which uses
matrix multiplication with transpositions).

#### Small Network - Forward Pass

![Small Network](assets/experiment6_small_forward.png)

#### Medium Network - Forward Pass

![Medium Network](assets/experiment6_medium_forward.png)

#### Large Network - Forward Pass

![Large Network](assets/experiment6_large_forward.png)

#### Small Network - Backward Pass

![Small Network](assets/experiment6_small_backward.png)

#### Medium Network - Backward Pass

![Medium Network](assets/experiment6_medium_backward.png)

#### Large Network - Backward Pass

![Large Network](assets/experiment6_large_backward.png)

### Profiling run without Shared Memory

```
$> nvprof ./build/CUDA-DNN-MNIST

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
 Tensor2DMultiplySharedMemory: 0
 Tensor2DMeanBlockSize: 8
-------------------------------------
 ReLuBlockSize: 8
-------------------------------------
 CrossEntropyGetMetricBlockSize: 64
 CrossEntropyCalculateBlockSize: 64
=====================================

[...]

==14648== Profiling application: ./build/CUDA-DNN-MNIST
==14648== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.42%  16.0404s     23400  685.49us  18.528us  3.7671ms  kMultiplyByTransposition(int, int, int, int, float*, int, int, float*, int, int, float*)
                   33.15%  13.8379s     27200  508.75us  72.095us  1.3625ms  kMultiply(int, int, int, int, float*, int, int, float*, int, int, float*)
                   20.91%  8.73106s     23400  373.12us  41.279us  746.52us  kTransposeAndMultiply(int, int, int, int, float*, int, int, float*, int, int, float*)
                    5.34%  2.22900s     54404  40.971us     479ns  240.51us  [CUDA memcpy HtoD]
                    0.51%  212.27ms     23400  9.0710us  8.6720us  14.368us  kMeanX(float*, int, int, float*)
                    0.32%  133.91ms     13600  9.8460us  9.2800us  12.352us  kSoftMaxCrossEntropyLoss(float*, int, int, float*, float*)
                    0.25%  105.33ms     25300  4.1630us  3.6470us  11.136us  kReLu(float*, int, int, float*)
                    0.24%  99.087ms     11700  8.4680us  7.9030us  13.088us  kSoftMaxCrossEntropy(float*, int, int, float*, float*)
                    0.23%  96.608ms     23400  4.1280us     895ns  2.3653ms  kScale(float*, float, int, int)
                    0.19%  80.514ms     23400  3.4400us     896ns  2.3759ms  kSubtract(float*, float*, int, int)
                    0.16%  64.817ms     27200  2.3820us  1.1510us  10.784us  kAdd1D(float*, float*, int, int)
                    0.09%  37.797ms     13600  2.7790us  2.2720us  12.032us  kSoftMaxCrossEntropyAccuracy(float*, int, int, float*, float*)
                    0.07%  27.193ms     23400  1.1620us     831ns  10.912us  kScale(float*, float, int)
                    0.06%  27.087ms     23400  1.1570us     831ns  10.112us  kSubtract(float*, float*, int)
                    0.06%  24.534ms     27200     902ns     480ns  11.520us  [CUDA memcpy DtoH]
      API calls:   75.15%  41.9718s     23400  1.7937ms  770.97us  6.8354ms  cudaEventSynchronize
                   21.04%  11.7509s     81604  144.00us  3.5690us  32.886ms  cudaMemcpy
                    2.14%  1.19280s    282400  4.2230us  3.0980us  6.3342ms  cudaLaunch
                    0.43%  238.46ms     54414  4.3820us  2.7560us  301.62us  cudaMalloc
                    0.39%  215.61ms     54400  3.9630us  2.2890us  43.260us  cudaFree
                    0.31%  174.21ms   1639700     106ns      77ns  355.90us  cudaSetupArgument
                    0.28%  155.65ms         2  77.825ms  1.7400us  155.65ms  cudaEventCreate
                    0.13%  74.986ms     46800  1.6020us  1.3010us  25.062us  cudaEventRecord
                    0.08%  45.967ms    282400     162ns      94ns  326.66us  cudaConfigureCall
                    0.05%  28.050ms     23400  1.1980us     935ns  286.02us  cudaEventElapsedTime
                    0.00%  292.09us         1  292.09us  292.09us  292.09us  cudaGetDeviceProperties
                    0.00%  229.12us        94  2.4370us      98ns  99.937us  cuDeviceGetAttribute
                    0.00%  43.008us         1  43.008us  43.008us  43.008us  cuDeviceGetName
                    0.00%  31.665us         1  31.665us  31.665us  31.665us  cuDeviceTotalMem
                    0.00%  1.6150us         3     538ns     116ns     948ns  cuDeviceGetCount
                    0.00%     699ns         2     349ns     213ns     486ns  cuDeviceGet
```

### Profiling run with Shared Memory

```
$> nvprof ./build/CUDA-DNN-MNIST

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
 Tensor2DMultiplySharedMemory: 1
 Tensor2DMeanBlockSize: 8
-------------------------------------
 ReLuBlockSize: 8
-------------------------------------
 CrossEntropyGetMetricBlockSize: 64
 CrossEntropyCalculateBlockSize: 64
=====================================

[...]

==14615== Profiling application: ./build/CUDA-DNN-MNIST
==14615== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   28.03%  3.95460s     27200  145.39us  7.7110us  541.05us  kMultiplyWithSharedMemory(float*, int, int, float*, int, int, float*)
                   26.20%  3.69663s     23400  157.98us  28.480us  305.85us  kTransposeAndMultiplyWithSharedMemory(float*, int, int, float*, int, int, float*)
                   23.95%  3.37863s     23400  144.39us  8.6080us  307.80us  kMultiplyByTranspositionWithSharedMemory(float*, int, int, float*, int, int, float*)
                   15.63%  2.20584s     54404  40.545us     479ns  242.81us  [CUDA memcpy HtoD]
                    1.53%  215.32ms     23400  9.2010us  8.5760us  14.528us  kMeanX(float*, int, int, float*)
                    0.94%  132.54ms     13600  9.7450us  9.2480us  12.192us  kSoftMaxCrossEntropyLoss(float*, int, int, float*, float*)
                    0.76%  107.87ms     25300  4.2630us  3.4240us  10.176us  kReLu(float*, int, int, float*)
                    0.70%  98.539ms     11700  8.4220us  7.8710us  11.584us  kSoftMaxCrossEntropy(float*, int, int, float*, float*)
                    0.54%  76.761ms     23400  3.2800us     896ns  12.479us  kScale(float*, float, int, int)
                    0.46%  65.574ms     23400  2.8020us     927ns  10.495us  kSubtract(float*, float*, int, int)
                    0.45%  64.116ms     27200  2.3570us  1.1510us  11.359us  kAdd1D(float*, float*, int, int)
                    0.27%  38.277ms     13600  2.8140us  2.4000us  11.808us  kSoftMaxCrossEntropyAccuracy(float*, int, int, float*, float*)
                    0.18%  25.519ms     23400  1.0900us     831ns  10.464us  kScale(float*, float, int)
                    0.18%  25.415ms     23400  1.0860us     831ns  9.1830us  kSubtract(float*, float*, int)
                    0.17%  23.705ms     27200     871ns     768ns  11.904us  [CUDA memcpy DtoH]
      API calls:   55.62%  15.7338s     23400  672.38us  40.850us  2.9389ms  cudaEventSynchronize
                   36.66%  10.3698s     81604  127.07us  3.5850us  32.556ms  cudaMemcpy
                    4.09%  1.15701s    282400  4.0970us  3.0420us  4.1125ms  cudaLaunch
                    0.97%  275.76ms         2  137.88ms  1.1490us  275.76ms  cudaEventCreate
                    0.85%  241.33ms     54414  4.4350us  2.8870us  235.44us  cudaMalloc
                    0.79%  224.37ms     54400  4.1240us  2.5040us  1.0894ms  cudaFree
                    0.52%  145.86ms   1343700     108ns      76ns  323.20us  cudaSetupArgument
                    0.25%  71.962ms     46800  1.5370us  1.2490us  319.73us  cudaEventRecord
                    0.13%  37.676ms    282400     133ns      88ns  295.90us  cudaConfigureCall
                    0.10%  28.298ms     23400  1.2090us     917ns  266.18us  cudaEventElapsedTime
                    0.00%  270.00us         1  270.00us  270.00us  270.00us  cudaGetDeviceProperties
                    0.00%  228.16us        94  2.4270us      97ns  99.357us  cuDeviceGetAttribute
                    0.00%  42.754us         1  42.754us  42.754us  42.754us  cuDeviceGetName
                    0.00%  32.095us         1  32.095us  32.095us  32.095us  cuDeviceTotalMem
                    0.00%  1.7430us         3     581ns     106ns  1.0910us  cuDeviceGetCount
                    0.00%     714ns         2     357ns     227ns     487ns  cuDeviceGet
```
