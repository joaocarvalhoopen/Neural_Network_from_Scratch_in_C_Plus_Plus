# Neural network from scratch in C++
This is an re-implementation of a neural network in C++ from scratch, it doesn't use any framework. <br>

[main.cpp](./src/main.cpp)

## Description
This re-implementation of a neural network in C++ from scratch based on the implementation in Python of the following article. Read it to really understand how a neural network works, then study the code in Python and in C++: <br>

[A Neural Network From Scratch by victorzhou](https://victorzhou.com/blog/intro-to-neural-networks/) <br>
[Github by victorzhou](https://github.com/vzhou842/neural-network-from-scratch) <br>
<br>
Note: The code is all in the main.cpp and not distributed by different files (*.h and *.cpp) to maintain consistency with the original article, were the Python code was all in the same file. <br>  
<br>
This project uses CMake as it's building system, integrated with a plugin inside Visual Studio Code. <br>
[CMakeLists.txt](./CMakeLists.txt) <br>

## License
MIT open source license

## Example of output

```
    Start training network....

    Epoch 0 loss: 0.483   
    Epoch 10 loss: 0.463  
    Epoch 20 loss: 0.414 
    Epoch 30 loss: 0.273 
    Epoch 40 loss: 0.152 
    Epoch 50 loss: 0.113 
    Epoch 60 loss: 0.091 
    Epoch 70 loss: 0.075 
    Epoch 80 loss: 0.063 
    Epoch 90 loss: 0.054 
    Epoch 100 loss: 0.047
    Epoch 110 loss: 0.041
    Epoch 120 loss: 0.036
    Epoch 130 loss: 0.032
    Epoch 140 loss: 0.029
    Epoch 150 loss: 0.026
    Epoch 160 loss: 0.024
    Epoch 170 loss: 0.022
    Epoch 180 loss: 0.020
    Epoch 190 loss: 0.019
    Epoch 200 loss: 0.018
    Epoch 210 loss: 0.017
    Epoch 220 loss: 0.016
    Epoch 230 loss: 0.015
    Epoch 240 loss: 0.014
    Epoch 250 loss: 0.013
    Epoch 260 loss: 0.012
    Epoch 280 loss: 0.011
    Epoch 300 loss: 0.010
    Epoch 310 loss: 0.010
    Epoch 320 loss: 0.009
    Epoch 350 loss: 0.008
    Epoch 390 loss: 0.007
    ...
    Epoch 490 loss: 0.006
    ...
    Epoch 580 loss: 0.005
    ...
    Epoch 730 loss: 0.003
    ...
    Epoch 990 loss: 0.002

     ...training network ended.

    Make some predictions:
       Emily pred: 0.948  ..... y_true 1 - Female 
       Frank pred: 0.040  ..... y_true 0 - Male

```

