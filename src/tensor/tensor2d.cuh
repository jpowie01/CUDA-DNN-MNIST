#pragma once
#ifndef TENSOR2D_HPP
#define TENSOR2D_HPP

#include <cstdio>

#include "../configuration.cuh"
#include "tensor1d.cuh"

enum Tensor2DAxis {
    X,
    Y
};

class Tensor2D {
private:
    int sizeX;
    int sizeY;
    float* devData;

public:
    Tensor2D(int sizeX, int sizeY);
    Tensor2D(int sizeX, int sizeY, float** hostData);
    Tensor2D(int sizeX, int sizeY, float* devData);
    ~Tensor2D();

    int getSize(Tensor2DAxis size);
    float* getDeviceData();
    float** fetchDataFromDevice();
    
    void add(Tensor2D* tensor);
    void add(Tensor1D* tensor);
    void subtract(Tensor2D* tensor);
    void scale(float factor);
    Tensor2D* multiply(Tensor2D* tensor, Tensor2D* output);
    Tensor2D* multiplyByTransposition(Tensor2D* tensor, Tensor2D* output);
    Tensor2D* transposeAndMultiply(Tensor2D* tensor, Tensor2D* output);
    Tensor1D* meanX(Tensor1D* output);

    void debugPrint();
};

#endif  /* !TENSOR2D_HPP */
