#pragma once
#ifndef MNIST_HPP
#define MNIST_HPP

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "../tensor/tensor2d.cuh"
#include "../utils.hpp"


enum DataSetType {
    TRAIN,
    TEST
};


class MNISTDataSet {
private:
    float** images;
    float** labels;
    int size;

public:
    MNISTDataSet(DataSetType type = TRAIN);
    
    int getSize();
    void shuffle();
    Tensor2D* getBatchOfImages(int index, int size);
    Tensor2D* getBatchOfLabels(int index, int size);
};

#endif  /* !MNIST_HPP */
