#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <SimpleLearning/DataLoader.h>
#include <SimpleLearning/NeuralNet.h>

void printUsage() {
    std::cout << "Commands:\n"
              << "  load <path_to_data>      Load images from directory\n"
              << "  train <epochs> <lr>      Train the model\n"
              << "  predict <image_path>     Predict a single image\n"
              << "  exit                     Exit the application\n"
              << "  help                     Show this message\n";
}

int main() {
    std::unique_ptr<NeuralNet> net;
    Eigen::MatrixXf X, Y;
    bool dataLoaded = false;
    int imgSize = 28;

    std::cout << "SimpleLearning CLI" << std::endl;
    printUsage();

    std::string command;
    while (true) {
        std::cout << "> ";
        std::cin >> command;

        if (command == "exit") {
            break;
        } else if (command == "help") {
            printUsage();
        } else if (command == "load") {
            std::string path;
            std::cin >> path;
            auto data = DataLoader::loadData(path, imgSize);
            if (data.first.size() > 0) {
                X = data.first;
                Y = data.second;
                dataLoaded = true;
                
                // Initialize network: Input -> 128 -> Classes
                int inputSize = X.rows();
                int numClasses = Y.rows();
                int hiddenSize = 128;
                
                std::cout << "Data loaded. Input size: " << inputSize << ", Classes: " << numClasses << std::endl;
                std::cout << "Initializing network: " << inputSize << " -> " << hiddenSize << " -> " << numClasses << std::endl;
                
                net = std::make_unique<NeuralNet>(std::vector<int>{inputSize, hiddenSize, numClasses});
            } else {
                std::cout << "Failed to load data or empty." << std::endl;
            }
        } else if (command == "train") {
            if (!dataLoaded || !net) {
                std::cout << "Load data first!" << std::endl;
                // consume rest of line
                std::string dummy; std::getline(std::cin, dummy);
                continue;
            }
            int epochs;
            float lr;
            std::cin >> epochs >> lr;
            std::cout << "Training for " << epochs << " epochs with LR " << lr << "..." << std::endl;
            net->train(X, Y, epochs, lr);
            std::cout << "Training complete." << std::endl;
        } else if (command == "predict") {
             if (!net) {
                std::cout << "Network not initialized. Load data and train first (or implement load model logic)." << std::endl;
                 std::string dummy; std::getline(std::cin, dummy);
                continue;
            }
            std::string path;
            std::cin >> path;
            
            // Load single image using CImg logic here or expose helper?
            // Re-use logic:
            try {
                cimg_library::CImg<float> img(path.c_str());
                img.resize(imgSize, imgSize);
                img.channel(0);
                img /= 255.0f;
                
                Eigen::MatrixXf inputVec(imgSize * imgSize, 1);
                int idx = 0;
                cimg_forXY(img, x, y) {
                    inputVec(idx++, 0) = img(x, y);
                }
                
                Eigen::MatrixXf output = net->predict(inputVec);
                
                int predictedClass = 0;
                output.col(0).maxCoeff(&predictedClass);
                
                std::cout << "Predicted Class: " << predictedClass << std::endl;
                std::cout << "Raw Output: " << output.transpose() << std::endl;
                
            } catch (const cimg_library::CImgException& e) {
                std::cerr << "Error loading image: " << e.what() << std::endl;
            }

        } else {
            std::cout << "Unknown command." << std::endl;
            // consume line
            std::string dummy; std::getline(std::cin, dummy);
        }
    }

    return 0;
}