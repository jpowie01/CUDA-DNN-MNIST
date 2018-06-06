#pragma once
#ifndef CSV_LOGGER_HPP
#define CSV_LOGGER_HPP

#include <string>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include "../configuration.cuh"

class CSVLogger {
private:
    int epoch;
    std::ofstream csvFile;

public:
    CSVLogger(std::string fileName);
    ~CSVLogger();

    void logEpoch(double trainingLoss, double trainingAccuracy, double testLoss, double testAccuracy,
                  double totalForwardTime, double totalBackwardTime, double batchForwardTime, double batchBackwardTime);
};

#endif /* !CSV_LOGGER_HPP */
