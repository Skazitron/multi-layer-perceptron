#include <SimpleLearning/DataLoader.h>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace cimg_library;

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> DataLoader::loadData(const std::string& path, int imgSize) {
    std::vector<Eigen::VectorXf> imageVectors;
    std::vector<int> labelIndices;
    int numClasses = 0;

    if (!fs::exists(path)) {
        std::cerr << "Path does not exist: " << path << std::endl;
        return {};
    }

    bool hasSubdirs = false;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            hasSubdirs = true;
            break;
        }
    }

    if (hasSubdirs) {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                int currentLabel = numClasses++;
                for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                    if (imgEntry.is_regular_file()) {
                         try {
                            CImg<float> img(imgEntry.path().string().c_str());
                            img.resize(imgSize, imgSize);
                            img.channel(0);                             
                            
                            img /= 255.0f;

                            Eigen::VectorXf vec(imgSize * imgSize);
                            int idx = 0;
                            cimg_forXY(img, x, y) {
                                vec(idx++) = img(x, y);
                            }
                            imageVectors.push_back(vec);
                            labelIndices.push_back(currentLabel);
                        } catch (const cimg_library::CImgException& e) {
                            std::cerr << "Error loading image " << imgEntry.path() << ": " << e.what() << std::endl;
                        }
                    }
                }
            }
        }
    } else {


        numClasses = 1;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {
               try {
                    CImg<float> img(entry.path().string().c_str());
                    img.resize(imgSize, imgSize);


                    img.channel(0);
                    img /= 255.0f;




                    Eigen::VectorXf vec(imgSize * imgSize);
                    int idx = 0;
                    cimg_forXY(img, x, y) {
                        vec(idx++) = img(x, y);
                    }
                    imageVectors.push_back(vec);
                    labelIndices.push_back(0);
                } catch (const cimg_library::CImgException& e) {
                     
                }
            }
        }
    }

    if (imageVectors.empty()) {
        std::cout << "No images found." << std::endl;
        return {};
    }

    int numSamples = imageVectors.size();
    int inputDim = imgSize * imgSize;
    


    Eigen::MatrixXf X(inputDim, numSamples);

    Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(numClasses, numSamples);

    for (int i = 0; i < numSamples; ++i) {

        X.col(i) = imageVectors[i];

        Y(labelIndices[i], i) = 1.0f; 
    }

    std::cout << "Loaded " << numSamples << " images from " << numClasses << " classes." << std::endl;

    return {X, Y};
}