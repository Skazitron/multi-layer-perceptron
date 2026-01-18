cmake_minimum_required(VERSION 3.10)
project(SimpleLearning VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(SYSTEM /Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/c++/v1)
find_package(Eigen3 5.0.1 REQUIRED NO_MODULE)
include_directories(include)

add_executable(SimpleLearning
    apps/main.cpp
    include/SimpleLearning/DenseLayer.h
    include/SimpleLearning/NeuralNet.h
    include/SimpleLearning/DataLoader.h
    src/DenseLayer.cpp
    src/NeuralNet.cpp
    src/DataLoader.cpp
)

 This automatically handles the include paths for you
target_link_libraries(SimpleLearning PRIVATE Eigen3::Eigen)