#pragma once
#include <SimpleLearning/DenseLayer.h>
#define cimg_display 0
#include <CImg.h>
#include <string>
#include <vector>
#include <utility>

class DataLoader {
    public:
    // Return pair of images
    static std::pair<Eigen::MatrixXf, Eigen::MatrixXf> loadData(const std::string& path, int imgSize = 28);
};