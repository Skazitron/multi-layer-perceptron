#include <SimpleLearning/NeuralNet.h>
#include <iostream>

void NeuralNet::train(const Eigen::MatrixXf& inputData, const Eigen::MatrixXf& targetData, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {

        Eigen::MatrixXf activation = inputData;
        for (auto& layer : layers) {
            activation = layer.forward(activation);
        }

        Eigen::MatrixXf dOutput = (activation - targetData);
        
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            dOutput = it->backward(dOutput, learningRate);
        }
    }
}

Eigen::MatrixXf NeuralNet::predict(const Eigen::MatrixXf& inputData) {
    Eigen::MatrixXf activation = inputData;
    for (auto& layer : layers) {
        activation = layer.forward(activation);
    }
    return activation;
}