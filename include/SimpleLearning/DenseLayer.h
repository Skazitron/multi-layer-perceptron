#pragma once
#include <Eigen/Eigen>

class DenseLayer {
    private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::MatrixXf input;
    Eigen::MatrixXf output;
    Eigen::MatrixXf zCache;

    

public:

    DenseLayer(int inputSize, int outputSize);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& inputData);
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput, float learningRate);

    private:
    static float sigmoid(float x);
    static float sigmoidDerivative(float x);
};