#pragma once
#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <string>
#include <cstdlib>
#include <cstdio>

#define DEFAULT_NUMBER_OF_EPOCHS    100
#define DEFAULT_BATCH_SIZE          512
#define DEFAULT_LEARNING_RATE       1e-06

#define DEFAULT_TENSOR2D_ADD_BLOCK_SIZE         8
#define DEFAULT_TENSOR2D_SUBTRACT_BLOCK_SIZE    8
#define DEFAULT_TENSOR2D_SCALE_BLOCK_SIZE       8
#define DEFAULT_TENSOR2D_MULTIPLY_BLOCK_SIZE    8
#define DEFAULT_TENSOR2D_MULTIPLY_BLOCK_NUMBER  -1  // Dynamic
#define DEFAULT_TENSOR2D_MULTIPLY_SHARED_MEMORY 0   // Turned off
#define DEFAULT_TENSOR2D_MEAN_BLOCK_SIZE        8

#define DEFAULT_TENSOR2D_RELU_BLOCK_SIZE        8

#define DEFAULT_CROSSENTROPY_GET_METRIC_BLOCK_SIZE      64
#define DEFAULT_CROSSENTROPY_CALCULATE_BLOCK_SIZE       64

#define DEFAULT_LOG_FILE_NAME       "logs/experiment.csv"

class Configuration {
private:
    static int getIntValue(std::string variableName, int defaultValue);
    static float getFloatValue(std::string variableName, float defaultValue);
    static std::string getStringValue(std::string variableName, std::string defaultValue);

public:
    static float learningRate;
    static int numberOfEpochs;
    static int batchSize;

    static int tensor2DAddBlockSize;
    static int tensor2DSubtractBlockSize;
    static int tensor2DScaleBlockSize;
    static int tensor2DMultiplyBlockSize;
    static int tensor2DMultiplyBlockNumber;
    static int tensor2DMultiplySharedMemory;
    static int tensor2DMeanBlockSize;

    static int reLuBlockSize;

    static int crossEntropyGetMetricBlockSize;
    static int crossEntropyCalculateBlockSize;

    static std::string logFileName;

    static void printCurrentConfiguration();
    static void printCUDAConfiguration();
};

#endif /* !CONFIGURATION_HPP */
