class Tensor1D {
private:
    int  size;
    float* devData;

public:
    Tensor1D(int size, float* data);
    ~Tensor1D();

    float* getDeviceData();
    float* fetchDataFromDevice();
    
    void add(Tensor1D* tensor);
};
