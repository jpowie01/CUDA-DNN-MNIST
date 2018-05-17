#include <stdio.h>
#include "tensor2d.h"


int main() {
    // Prepare example data
    const int N = 100;
    
    int** a = new int*[N];
    *a = new int[N * N];
    for (int i = 1; i < N; i++) a[i] = a[i-1] + N;

    int** b = new int*[N];
    *b = new int[N * N];
    for (int i = 1; i < N; i++) b[i] = b[i-1] + N;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            a[y][x] = b[y][x] = y*N + x;
        }
    }

    // Execute on GPU
    Tensor2D* tensorA = new Tensor2D(N, N, a);
    Tensor2D* tensorB = new Tensor2D(N, N, b);
    tensorA->add(tensorB);

    // Fetch data and print results
    int** data = tensorA->fetchDataFromDevice();
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            printf("%d\n", data[y][x]);
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
