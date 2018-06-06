#include "mnist.hpp"

#define BEGIN_OF_PIXELS 16
#define BEGIN_OF_LABELS 8

#define NUMBER_OF_PROBE_IMAGES 1000

MNISTDataSet::MNISTDataSet(DataSetType type) {
    // Prepare some placeholders for our dataset
    FILE *file;
    long length;

    // Open file with images and check how long it is
    if (type == TRAIN) {
        file = fopen("data/train-images", "rb");
    } else if (type == TEST) {
        file = fopen("data/test-images", "rb");
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    // Read whole file with images to the buffer
    unsigned char* bufferImages = (unsigned char *)malloc((length+1)*sizeof(unsigned char));
    fread(bufferImages, length, 1, file);
    fclose(file);

    // Open file with labels and check how long it is
    if (type == TRAIN) {
        file = fopen("data/train-labels", "rb");
    } else if (type == TEST) {
        file = fopen("data/test-labels", "rb");
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    // Read whole file with labels to the buffer
    unsigned char* bufferLabels = (unsigned char *)malloc((length+1)*sizeof(unsigned char));
    fread(bufferLabels, length, 1, file);
    fclose(file);

    // Keep size of the dataset in the property as we'll be using this value a lot!
    this->size = (int)((bufferImages[4] << 24) + (bufferImages[5] << 16) + (bufferImages[6] << 8) + bufferImages[7]);

    // Prepare value for mean image - we don't have to calculate exact mean value. Mean value of a 1000 images will be enough!
    float* meanImage = new float[28*28];
    for (int i = 0; i < 28*28; i++) {
        meanImage[i] = 0.0;
    }
    for (int image = 0; image < NUMBER_OF_PROBE_IMAGES; image++) {
        for (int i = 0; i < 28*28; i++) {
            meanImage[i] += bufferImages[BEGIN_OF_PIXELS + image*28*28 + i];
        }
    }
    for (int i = 0; i < 28*28; i++) {
        meanImage[i] /= NUMBER_OF_PROBE_IMAGES;
    }

    // Prepare value for standard deviation. It's the same story as above - 1000 images will be enough!
    float* stdDevImage = new float[28*28];
    for (int i = 0; i < 28*28; i++) {
        stdDevImage[i] = 0.0;
    }
    for (int image = 0; image < NUMBER_OF_PROBE_IMAGES; image++) {
        for (int i = 0; i < 28*28; i++) {
            stdDevImage[i] += pow(bufferImages[BEGIN_OF_PIXELS + image*28*28 + i] - meanImage[i], 2.0);
        }
    }
    for (int i = 0; i < 28*28; i++) {
        stdDevImage[i] = sqrt(stdDevImage[i] / (NUMBER_OF_PROBE_IMAGES - 1));
    }

    // Now let's read all images and convert them to floating point values
    // Together with this operation, let's perform simple image preprocessing
    //   Final image = (Input Image - Mean Image) / Std Dev Image
    this->images = new float*[this->size];
    *this->images = new float[this->size*28*28];
    for (int image = 1; image < this->size; image++) this->images[image] = this->images[image-1] + 28*28;
    for (int image = 0; image < this->size; image++) {
        for (int i = 0; i < 28*28; i++) {
            if (stdDevImage[i] > 1e-10) {
                // TODO: Test set shouldn't apply its mean and std dev values!
                // TODO: It should use the same values as were used in the training dataset!
                this->images[image][i] = (float)(bufferImages[BEGIN_OF_PIXELS + image*28*28 + i] - meanImage[i]) / stdDevImage[i];
            } else {
                this->images[image][i] = 0.0;
            }
        }
    }

    // And now let's read all labels from the MNIST dataset
    // Once we've read this values, let's convert them to the one-hot encoding
    //   4 -> 0000100000
    this->labels = new float*[this->size];
    *this->labels = new float[this->size*10];
    for (int image = 1; image < this->size; image++) this->labels[image] = this->labels[image-1] + 10;
    for (int image = 0; image < this->size; image++) {
        for(int i = 0; i < 10; i++) {
            this->labels[image][i] = 0;
        }
        int label = (int)bufferLabels[BEGIN_OF_LABELS + image];
        this->labels[image][label] = 1;
    }
}

int MNISTDataSet::getSize() {
    return this->size;
}

void MNISTDataSet::shuffle() {
    float tmpImage[28*28];
    float tmpLabel[10];

    // Each image should find new place for itself (randomly)
    for (int i = 0; i < sqrt(this->size); i++) {
        int newPosition = randomInt(sqrt(this->size), this->size-1);

        // We cannot simply swap two pointers to these images as we assume later that all images
        // are linearly aligned in memory, one next to the other - this helps with mapping
        // host CPU data to the GPU global memory
        int memorySize = 28 * 28 * sizeof(float);
        void* sourceImage = *(this->images + i);
        void* destinationImage = *(this->images + newPosition);
        memcpy(tmpImage, sourceImage, memorySize);
        memcpy(sourceImage, destinationImage, memorySize);
        memcpy(destinationImage, tmpImage, memorySize);

        // The same story as above - we've got to copy memory between two labels
        memorySize = 10 * sizeof(float);
        void* sourceLabel = *(this->labels + i);
        void* destinationLabel = *(this->labels + newPosition);
        memcpy(tmpLabel, sourceLabel, memorySize);
        memcpy(sourceLabel, destinationLabel, memorySize);
        memcpy(destinationLabel, tmpLabel, memorySize);
    }
}

Tensor2D* MNISTDataSet::getBatchOfImages(int index, int size) {
    return new Tensor2D(28*28, size, (float**)(this->images + index*size));
}

Tensor2D* MNISTDataSet::getBatchOfLabels(int index, int size) {
    return new Tensor2D(10, size, (float**)(this->labels + index*size));
}
