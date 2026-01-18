#include <SimpleLearning/DenseLayer.h>
#include <cmath>
#include <iostream>

DenseLayer::DenseLayer(int inputSize, int outputSize) {
    weights = Eigen::MatrixXf::Random(outputSize, inputSize);
    biases = Eigen::VectorXf::Random(outputSize);
}

float DenseLayer::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float DenseLayer::sigmoidDerivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& inputData) {
    input = inputData;
    zCache = (weights * input).colwise() + biases; 
    output = zCache.unaryExpr(&DenseLayer::sigmoid);
    return output;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& dOutput, float learningRate) {
    Eigen::MatrixXf dZ = dOutput.cwiseProduct(zCache.unaryExpr(&DenseLayer::sigmoidDerivative));
    
    Eigen::MatrixXf dW = dZ * input.transpose();
    Eigen::VectorXf dB = dZ.rowwise().sum();
    Eigen::MatrixXf dInput = weights.transpose() * dZ;
    
    weights -= learningRate * dW;
    biases -= learningRate * dB;
    
    return dInput;
}