class Tensor2D {
private:
    int sizeX;
    int sizeY;
    int* devData;

public:
    Tensor2D(int sizeX, int sizeY, int** hostData);
    ~Tensor2D();

    int* getDeviceData();
    int** fetchDataFromDevice();
    
    void add(Tensor2D* tensor);
};
