#include "mnist.h"

#define BEGIN_OF_PIXELS 16
#define BEGIN_OF_LABELS 8

MNISTDataSet::MNISTDataSet() {
    FILE *fileptr;
    long filelen;

    fileptr = fopen("data/train-images", "rb");
    fseek(fileptr, 0, SEEK_END);
    filelen = ftell(fileptr);
    rewind(fileptr);

    this->bufferImages = (unsigned char *)malloc((filelen+1)*sizeof(unsigned char));
    fread(this->bufferImages, filelen, 1, fileptr);
    fclose(fileptr);

    fileptr = fopen("data/train-labels", "rb");
    fseek(fileptr, 0, SEEK_END);
    filelen = ftell(fileptr);
    rewind(fileptr);

    this->bufferLabels = (unsigned char *)malloc((filelen+1)*sizeof(unsigned char));
    fread(this->bufferLabels, filelen, 1, fileptr);
    fclose(fileptr);

    this->size = (int)((bufferImages[4] << 24) + (bufferImages[5] << 16) + (bufferImages[6] << 8) + bufferImages[7]);
}

int MNISTDataSet::getSize() {
    return this->size;
}

Tensor2D* MNISTDataSet::getBatchOfImages(int index, int size) {
    float** rawBatch = new float*[size];
    *rawBatch = new float[size*28*28];
    for (int i = 1; i < size; i++) rawBatch[i] = rawBatch[i-1] + 28*28;
    for (int batch = 0; batch < size; batch++) {
        for(int i = 0; i < 28*28; i++) {
            rawBatch[batch][i] = (float)(this->bufferImages[index*size*28*28 + batch*28*28 + i + BEGIN_OF_PIXELS]);
        }
    }
    return new Tensor2D(28*28, size, rawBatch);
}

Tensor2D* MNISTDataSet::getBatchOfLabels(int index, int size) {
    float** rawBatch = new float*[size];
    *rawBatch = new float[size*10];
    for (int i = 1; i < size; i++) rawBatch[i] = rawBatch[i-1] + 10;
    for (int batch = 0; batch < size; batch++) {
        int label = (int)this->bufferLabels[index*size + batch + BEGIN_OF_LABELS];
        for(int i = 0; i < 10; i++) {
            rawBatch[batch][i] = 0;
        }
        rawBatch[batch][label] = 1;
    }
    return new Tensor2D(10, size, rawBatch);
}
