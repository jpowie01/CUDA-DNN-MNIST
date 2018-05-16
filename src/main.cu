#include <stdio.h>
#include "tensor1d.h"


int main() {
    // Prepare example data
    const int N = 1000;
    int a[N], b[N];
    for (int i = 0; i<N; ++i) {
        a[i] = b[i] = i;
    }

    // Execute on GPU
    Tensor1D* tensorA = new Tensor1D(N, a);
    Tensor1D* tensorB = new Tensor1D(N, b);
    tensorA->add(tensorB);

    // Fetch data and print results
    int* data = tensorA->fetchDataFromDevice();
    for (int i = 0; i<N; ++i) {
        printf("%d\n", data[i]);
    }

    // Clean memory
    delete tensorA;
    delete tensorB;
    return 0;
}
