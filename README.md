# SimpleLearning
A simple multi-layer perceptron that I built to learn how neural nets work at a fundamental level. I used the CImg library for image parsing and the Eigen library for Matri xmultiplications.

In order to build and run it, it's important for you to edit the CMakeLists.txt file to fit your own hardware. In my case it's a MacOS. If you're on Linux or Windows, you have to change the include directory for your default headers. You will also need to have the Eigen3 library on your hardware. After that you'll have to build the program with CMake.

Take a look at the main file, which was mostly AI generated, to get a better understanding of how to use this program. It supports a number of commands. Namely:
```
                load <path_to_data>      Load images from directory"`
                train <epochs> <lr>      Train the model
                predict <image_path>     Predict a single image"
                exit                     Exit the application"
                help                     Show this message";

```

Enjoy!
