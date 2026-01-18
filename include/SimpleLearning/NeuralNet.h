#pragma once
#include <SimpleLearning/DenseLayer.h>

class NeuralNet {
    private:
    std::vector<DenseLayer> layers;

    public:
    NeuralNet(const std::vector<int>& layerSizes): layers() {
        for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
            layers.emplace_back(layerSizes[i], layerSizes[i + 1]);
        }
    }
    void train(const Eigen::MatrixXf& inputData, const Eigen::MatrixXf& targetData, int epochs, float learningRate);
    Eigen::MatrixXf predict(const Eigen::MatrixXf& inputData);
    

};