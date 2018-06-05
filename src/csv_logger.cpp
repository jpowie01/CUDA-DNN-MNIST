#include "csv_logger.hpp"

CSVLogger::CSVLogger(std::string fileName) {
    this->csvFile.open(fileName.c_str());
    if (!this->csvFile.is_open()) {
        printf("ERROR! Cannot open file for CSVLogger.");
        exit(1);
    }

    this->epoch = 0;
    this->csvFile << "epoch,"
                  << "trainingLoss,"
                  << "trainingAccuracy,"
                  << "testLoss,"
                  << "testAccuracy,"
                  << "totalForwardTime,"
                  << "totalBackwardTime,"
                  << "batchForwardTime,"
                  << "batchBackwardTime\n";
}

CSVLogger::~CSVLogger() {
    this->csvFile.close();
}

void CSVLogger::logEpoch(double trainingLoss, double trainingAccuracy,
                         double testLoss, double testAccuracy,
                         double totalForwardTime, double totalBackwardTime,
                         double batchForwardTime, double batchBackwardTime) {
    this->csvFile << this->epoch++ << ","
                  << trainingLoss << ","
                  << trainingAccuracy << ","
                  << testLoss << ","
                  << testAccuracy << ","
                  << totalForwardTime << ","
                  << totalBackwardTime << ","
                  << batchForwardTime << ","
                  << batchBackwardTime << "\n";
}

