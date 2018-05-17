#include <stdio.h>
#include "tensor2d.h"


int main() {
    // Prepare example data
    const int N = 100; 

    float** a = new float*[N];
    *a = new float[N * N];
    for (int i = 1; i < N; i++) a[i] = a[i-1] + N;

    float** b = new float*[N];
    *b = new float[N * N];
    for (int i = 1; i < N; i++) b[i] = b[i-1] + N;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            a[y][x] = b[y][x] = y*N + x;
        }
    }

    // Execute on GPU
    Tensor2D* tensorA = new Tensor2D(N, N, a);
    Tensor2D* tensorB = new Tensor2D(N, N, b);
    Tensor2D* tensorC = tensorA->multiply(tensorB);

    // Fetch data and print results
    float** data = tensorC->fetchDataFromDevice();
    printf("X: %d Y: %d\n", tensorC->sizeX, tensorC->sizeY);
    for (int y = 0; y < tensorC->sizeY; y++) {
        for (int x = 0; x < tensorC->sizeX; x++) {
            printf("X: %f Y: %f => %d\n", x, y, data[y][x]);
        }
    }

    // Clean memory
    delete tensorA;
    delete tensorB;
    delete data;
    delete b;
    delete a;

    return 0;
}
