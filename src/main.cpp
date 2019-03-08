/******************************************************************************
 * Author: Joao Nuno Carvalho                                                 *
 * License: MIT open source                                                   *
 * Description: This is an re-implementation of a neural network in C++ from  *
 *              scratch based on the implementation in Python of the          *
 *              following article:                                            *
 *              A Neural Network From Scratch                                 *
 *              https://victorzhou.com/blog/intro-to-neural-networks/         *
 *              https://github.com/vzhou842/neural-network-from-scratch       *
 *                                                                            *
 * Note: The code is all in the main.cpp and not distributed by different     *
 *       files (*.h and *.cpp) to maintain consistency with the original      *
 *       article, were the Python code was all in the same file.              *
 *                                                                            *
 ******************************************************************************/ 


#include <stdio.h>      /* printf */
#include <iostream>
#include <math.h>       /* exp */
#include <vector>
#include <algorithm>    // std::all_of
#include <random>       // Normal distribution

using namespace std;

// Declaration
double sigmoid(double x);
double deriv_sigmoid(double x);
double mse_loss(vector<double> y_true, vector<double> y_pred);

// Declarations
class Neuron {
    public:
        Neuron();
        Neuron(vector<double> weights, double bias);
        ~Neuron();
        double feedForward(vector<double> inputs);
        static bool testNeuron(void);

    private:
        vector<double> weights;
        double bias;
        vector<double> output;
};

// Definition
double sigmoid(double x){
    // Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + exp(-x));
}

double deriv_sigmoid(double x){
    // Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

double mse_loss(vector<double> y_true, vector<double> y_pred){
  // y_true and y_pred are vector's of the same length.
  // return ((y_true - y_pred) ** 2).mean()
  double res = 0;
  for(size_t i = 0; i < y_true.size(); ++i)
  {
      res += pow(y_true[i] - y_pred[i], 2);
  }
  res /= y_true.size();
  return res;
}

// Definition
Neuron::Neuron(){
    //  dummy constructor.
}

Neuron::Neuron(vector<double> _weights, double _bias){
    weights = _weights;
    bias    = _bias;
}

Neuron::~Neuron(){
    // TODO: Implement
}

double Neuron::feedForward(vector<double> inputs){
    // Weight inputs, add bias, then use the activation function
    // Implementation of the dot product.
    double total = 0;
    for(size_t i = 0; i < inputs.size(); ++i){
        total += weights[i] * inputs[i];
    }
    total += bias;
    return sigmoid(total);
}

bool Neuron::testNeuron(void){
    bool res = false;
    vector<double> weights = {0, 1};  // w1 = 0, w2 = 1
    double bias = 4;                  // b = 4
    Neuron n(weights, bias);
    vector<double> x_inputs = {2, 3}; // x1 = 2, x2 = 3
    double y_output = n.feedForward(x_inputs);
    if ((y_output > 0.9990889488055994 - 0.0001) && (y_output < 0.9990889488055994 + 0.0001)){
        res = true;
        printf("testNeuron, the output should be 0.9990889488055994 and is %lf\n", y_output);
    } else {
        printf("Error in testNeuron!\n");
        printf("testNeuron, the output should be 0.9990889488055994 and is %lf\n", y_output);
    }
    return res;
}

////////////////

class OurTestNeuralNetwork {
    /*
        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0
    */

    public:
        OurTestNeuralNetwork();
        ~OurTestNeuralNetwork();
        double feedForward(vector<double> x);
        static bool testOurTestNeuralNetwork(void);

    private:
        vector<double> weights;
        double bias;
        Neuron h1;
        Neuron h2;
        Neuron o1;
};

// Declaration
OurTestNeuralNetwork::OurTestNeuralNetwork(){
    weights = {0, 1};
    bias    = 0;

    // The Neuron class here is from the previous section
    h1 = Neuron(weights, bias);
    h2 = Neuron(weights, bias);
    o1 = Neuron(weights, bias);
}

OurTestNeuralNetwork::~OurTestNeuralNetwork(){

}

double OurTestNeuralNetwork::feedForward(vector<double> x){
    double out_h1 = h1.feedForward(x);
    double out_h2 = h2.feedForward(x);

    // The inputs for o1 are the outputs from h1 and h2
    vector<double> out_h1_h2_vec = {out_h1, out_h2};
    double out_o1 = o1.feedForward(out_h1_h2_vec);

    return out_o1;
}

bool OurTestNeuralNetwork::testOurTestNeuralNetwork(void){
    bool res = false;

    OurTestNeuralNetwork network;
    vector<double> x = {2, 3};
    double y_pred =  network.feedForward(x);

    if ((y_pred > 0.7216325609518421 - 0.0001) && (y_pred < 0.7216325609518421 + 0.0001)){
        res = true;
        printf("testOurTestNeuralNetwork, the output should be 0.7216325609518421 and is %lf\n", y_pred);
    } else {
        printf("Error in testNeuron!\n");
        printf("test, the output should be 0.7216325609518421 and is %lf\n", y_pred);
    }
    return res;
}

////////////////

// Definition
class OurNeuralNetwork {
    /*
        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0


        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)

        *** DISCLAIMER ***:
        The code below is intended to be simple and educational, NOT optimal.
        Real neural net code looks nothing like this. DO NOT use this code.
        Instead, read/run it to understand how this specific network works.

    */

    public:
        OurNeuralNetwork();
        ~OurNeuralNetwork();
        double feedForward(vector<double> x);
        void train(vector<vector<double>> data, vector<double> all_y_true);
        static bool testOurNeuralNetwork(void);

    private:
        // Weights
        double w1;
        double w2;
        double w3;
        double w4;
        double w5;
        double w6;

        // Biases
        double b1;
        double b2;
        double b3;
};

// Declaration.
OurNeuralNetwork::OurNeuralNetwork(){
    default_random_engine generator;
    normal_distribution<double> distribution(1.0,0.0);

    // Weights
    w1 = distribution(generator);
    w2 = distribution(generator);
    w3 = distribution(generator);
    w4 = distribution(generator);
    w5 = distribution(generator);
    w6 = distribution(generator);

    // Biases
    b1 = distribution(generator);
    b2 = distribution(generator);
    b3 = distribution(generator);
}

OurNeuralNetwork::~OurNeuralNetwork(){

}

double OurNeuralNetwork::feedForward(vector<double> x){
    // x is a vector with 2 elements.
    double h1 = sigmoid(w1 * x[0] + w2 * x[1] + b1);
    double h2 = sigmoid(w3 * x[0] + w4 * x[1] + b2);
    double o1 = sigmoid(w5 * h1   + w6 * h2   + b3);
    return o1;
}

void OurNeuralNetwork::train(vector<vector<double>> data, vector<double> all_y_true){
    // - data is a (n x 2) vector of vectors, n = # of samples in the dataset.
    // - all_y_trues is a vector of vectors with n elements.
    //   Elements in all_y_trues correspond to those in data.

    printf("\n\n Start training network....\n\n");

    double learn_rate = 0.1;
    int    max_epochs = 1000; // Number of times to loop through the entire dataset

    for (int epoch = 0; epoch < max_epochs; ++epoch){
        for (size_t i = 0; i < data.size() ; ++i){
            vector<double> x      = data[i]; 
            double         y_true = all_y_true[i];

            // --- Do a feedforward (we'll need these values later)
            double sum_h1 = w1 * x[0] + w2 * x[1] + b1;
            double h1 = sigmoid(sum_h1);

            double sum_h2 = w3 * x[0] + w4 * x[1] + b2;
            double h2 = sigmoid(sum_h2);

            double sum_o1 = w5 * h1 + w6 * h2 + b3;
            double o1 = sigmoid(sum_o1);
            double y_pred = o1;

            // --- Calculate partial derivatives.
            // --- Naming: d_L_d_w1 represents "partial L / partial w1"
            double d_L_d_ypred = -2 * (y_true - y_pred);

            // Neuron o1
            double d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1);
            double d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1);
            double d_ypred_d_b3 = deriv_sigmoid(sum_o1);

            double d_ypred_d_h1 = w5 * deriv_sigmoid(sum_o1);
            double d_ypred_d_h2 = w6 * deriv_sigmoid(sum_o1);

            // Neuron h1
            double d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1);
            double d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1);
            double d_h1_d_b1 = deriv_sigmoid(sum_h1);

            // Neuron h2
            double d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2);
            double d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2);
            double d_h2_d_b2 = deriv_sigmoid(sum_h2);

            // --- Update weights and biases
            // Neuron h1
            w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
            w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
            b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

            // Neuron h2
            w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
            w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
            b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

            // Neuron o1
            w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
            w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
            b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3;

        }
        // --- Calculate total loss at the end of each epoch
        if (epoch % 10 == 0){
            vector<double> y_preds = {};

            for (size_t i = 0; i < data.size() ; ++i){
                vector<double> x      = data[i];
                y_preds.push_back(feedForward(x));
            }
            
            double loss = mse_loss(all_y_true, y_preds);
            printf("Epoch %d loss: %.3lf\n", epoch, loss);
        }
    }

    printf("\n\n ...training network ended.\n\n");
}

bool OurNeuralNetwork::testOurNeuralNetwork(void){

    vector<vector<double>> data = { { -2, -1},  // Alice
                                    { 25,  6},  // Bob
                                    { 17,  4},  // Charlie
                                    {-15, -6}   // Diana 
                                    };
                                    
    vector<double> all_y_trues = {  1,  // Alice
                                    0,  // Bob
                                    0,  // Charlie
                                    1   // Diana 
                                    };

    // Train our neural network!
    OurNeuralNetwork network;
    network.train(data, all_y_trues);

    // Make some predictions
    vector<double> emily = {-7, -3};  // 128 pounds, 63 inches
    vector<double> frank = {20,  2};  // 155 pounds, 68 inches
    printf("Make some predictions:\n");
    printf("   Emily pred: %.3lf  ..... y_true 1 - Female \n", network.feedForward(emily));  // 0.951 - F
    printf("   Frank pred: %.3lf  ..... y_true 0 - Male   \n", network.feedForward(frank));  // 0.039 - M

    return true;
}

void runTests(void){
    vector<bool> allTests;
    int counter = 0;

    allTests.push_back(Neuron::testNeuron());
    counter++;

    allTests.push_back(OurTestNeuralNetwork::testOurTestNeuralNetwork());
    counter++;

    allTests.push_back(OurNeuralNetwork::testOurNeuralNetwork());
    counter++;

    if (all_of( allTests.begin(), allTests.end(), [](bool t){return t;} ))  // Lambda function
        cout << endl << "ALL " << counter << " TEST'S PASSED !!!" << endl;
    else
        cout << endl << "Some test's FAILED !!!" << endl;
}

int main(int argc, char* argv[]) {
    cout << "Neural network from scratch in C++..." << endl;

    runTests();

    cout << "End..." << endl;
}
