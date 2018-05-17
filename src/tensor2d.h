class Tensor2D {
public:
    // TODO: Make me private!
    int sizeX;
    int sizeY;
    float* devData;
    
    Tensor2D(int sizeX, int sizeY, float** hostData);
    Tensor2D(int sizeX, int sizeY, float* devData);
    ~Tensor2D();

    float* getDeviceData();
    float** fetchDataFromDevice();
    
    void add(Tensor2D* tensor);
    Tensor2D* multiply(Tensor2D* tensor);
};
