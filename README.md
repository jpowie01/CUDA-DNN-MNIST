CUDA DNN MNIST
--------------

This project is an example implementation for training simple feed forward neural network
 on a MNIST dataset in pure C++ CUDA code.

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
  - Train Loss=1.58667
  - Train Accuracy=78.38375%
  - Test Loss=nan
  - Test Accuracy=93.47245%
```
