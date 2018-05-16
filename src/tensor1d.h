class Tensor1D {
private:
    int  size;
    int* devData;

public:
    Tensor1D(int size, int* data);
    ~Tensor1D();

    int* getDeviceData();
    int* fetchDataFromDevice();
    
    void add(Tensor1D* tensor);
};
