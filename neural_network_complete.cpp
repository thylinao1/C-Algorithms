 //NEURAL NETWORK WITH BACKPROPAGATION 

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// NEURAL NETWORK CLASS

class NeuralNetwork {
private:
    // Network architecture
    int input_size_;
    int hidden1_size_;
    int hidden2_size_;
    int output_size_;
    
    // Network parameters 
    Matrix W1_, W2_, W3_;
    Vector b1_, b2_, b3_;
    
    // activations
    // a0 = input, a1 = first hidden output, a2 = second hidden output, a3 = final output
    // z1, z2, z3 = weighted sums before activation
    Matrix a0_, z1_, a1_, z2_, a2_, z3_, a3_;
    
    // Random number generator for weight initialization
    std::mt19937 rng_;
    
    // Activation functions

    //Sigmoid activation function
    static Matrix sigma(const Matrix& z) {
        return (1.0 + (-z.array()).exp()).inverse().matrix();}
    
    // Derivative of sigmoid
    static Matrix d_sigma(const Matrix& z) {
        return ((z.array() / 2.0).cosh().pow(-2) / 4.0).matrix();}
    
    // Forward Propagation
    
     //Forward pass through the network
    void network_function(const Matrix& input) {
        // Store input
        a0_ = input;
        // First hidden layer
        z1_ = W1_ * a0_; 
        z1_.colwise() += b1_;  
        a1_ = sigma(z1_); 
        // Second hidden layer
        z2_ = W2_ * a1_;
        z2_.colwise() += b2_;
        a2_ = sigma(z2_);
        // Output layer
        z3_ = W3_ * a2_;
        z3_.colwise() += b3_;
        a3_ = sigma(z3_); }
    
    // Backpropagation
    
    //Compute gradient for W3 
     //Using chain rule
    Matrix jacobian_W3(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);        
        J = J.array() * d_sigma(z3_).array(); 
        Matrix grad = J * a2_.transpose() / X.cols(); 
        return grad;}
    
     //Compute gradient for b3 
    Vector jacobian_b3(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);
        J = J.array() * d_sigma(z3_).array();
        Vector grad = J.rowwise().sum() / X.cols();
        return grad;}
    
    /**
     * Compute gradient for W2
     * Now we need to backpropagate through layer 3:
     */
    Matrix jacobian_W2(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);
        J = J.array() * d_sigma(z3_).array();
        J = W3_.transpose() * J;   
        J = J.array() * d_sigma(z2_).array();
        Matrix grad = J * a1_.transpose() / X.cols();
        return grad; }
    
     //Compute gradient for b2
    Vector jacobian_b2(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);
        J = J.array() * d_sigma(z3_).array();
        J = W3_.transpose() * J;
        J = J.array() * d_sigma(z2_).array();
        Vector grad = J.rowwise().sum() / X.cols();
        return grad;}
    
     //Compute gradient for W1 
    Matrix jacobian_W1(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);
        J = J.array() * d_sigma(z3_).array();
        J = W3_.transpose() * J;   
        J = J.array() * d_sigma(z2_).array();
        J = W2_.transpose() * J;           
        J = J.array() * d_sigma(z1_).array();
        Matrix grad = J * a0_.transpose() / X.cols();
        return grad;}
    
     //Compute gradient for b1
    Vector jacobian_b1(const Matrix& X, const Matrix& Y) {
        network_function(X);
        
        Matrix J = 2.0 * (a3_ - Y);
        J = J.array() * d_sigma(z3_).array();
        J = W3_.transpose() * J;
        J = J.array() * d_sigma(z2_).array();
        J = W2_.transpose() * J;
        J = J.array() * d_sigma(z1_).array();
        Vector grad = J.rowwise().sum() / X.cols();
        return grad;}

public:
    // Constructior
    /**
     * Create a neural network
     * @param input_size: Number of input features
     * @param hidden1_size: Number of neurons in first hidden layer
     * @param hidden2_size: Number of neurons in second hidden layer
     * @param output_size: Number of output neurons
     * @param seed: Random seed for reproducibility
     */
    NeuralNetwork(int input_size, int hidden1_size, int hidden2_size, 
                  int output_size, unsigned int seed = 42)
        : input_size_(input_size), hidden1_size_(hidden1_size),
          hidden2_size_(hidden2_size), output_size_(output_size),rng_(seed) {
        
        // Initialize weights with small random values
        // Using normal distribution with mean=0, std=0.5
        std::normal_distribution<double> dist(0.0, 0.5);
        
        W1_ = Matrix::Zero(hidden1_size_, input_size_);
        W2_ = Matrix::Zero(hidden2_size_, hidden1_size_);
        W3_ = Matrix::Zero(output_size_, hidden2_size_);
        // Fill with random values
        for (int i = 0; i < W1_.size(); ++i) W1_(i) = dist(rng_);
        for (int i = 0; i < W2_.size(); ++i) W2_(i) = dist(rng_);
        for (int i = 0; i < W3_.size(); ++i) W3_(i) = dist(rng_);
        
        // Initialize biases
        b1_ = Vector::Zero(hidden1_size_);
        b2_ = Vector::Zero(hidden2_size_);
        b3_ = Vector::Zero(output_size_);
        for (int i = 0; i < hidden1_size_; ++i) b1_(i) = dist(rng_);
        for (int i = 0; i < hidden2_size_; ++i) b2_(i) = dist(rng_);
        for (int i = 0; i < output_size_; ++i) b3_(i) = dist(rng_);}
    
    // Training
    
    /**
     * Train the neural network using gradient descent
     * 
     * @param X: Input data (features × samples)
     * @param Y: Target outputs (outputs × samples)
     * @param learning_rate: Step size for gradient descent
     * @param epochs: Number of training iterations
     * @param verbose: Print progress during training
     * 
     * @return Vector of loss values at each epoch
     */
    std::vector<double> train(const Matrix& X, const Matrix& Y,
                             double learning_rate, int epochs,bool verbose = true) {
        std::vector<double> loss_history;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            //Compute all gradients via backpropagation
            Matrix grad_W3 = jacobian_W3(X, Y);
            Vector grad_b3 = jacobian_b3(X, Y);
            Matrix grad_W2 = jacobian_W2(X, Y);
            Vector grad_b2 = jacobian_b2(X, Y);
            Matrix grad_W1 = jacobian_W1(X, Y);
            Vector grad_b1 = jacobian_b1(X, Y);
            
            //Update all parameters (gradient descent)
            W3_ -= learning_rate * grad_W3;
            b3_ -= learning_rate * grad_b3;
            W2_ -= learning_rate * grad_W2;
            b2_ -= learning_rate * grad_b2;
            W1_ -= learning_rate * grad_W1;
            b1_ -= learning_rate * grad_b1;
            
            // Compute loss (MSE)
            double current_loss = compute_cost(X, Y);
            loss_history.push_back(current_loss);
            
            // Print progress every 100 epochs
            if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
                std::cout << "Epoch " << std::setw(4) << epoch 
                          << " Loss: " << std::fixed << std::setprecision(6) 
                          << current_loss << std::endl;}}
        if (verbose) {
            std::cout << "Final loss: " << loss_history.back() << "\n" << std::endl;}
        return loss_history; }
    
     //Compute MSE loss
    double compute_cost(const Matrix& X, const Matrix& Y) {
        Matrix predictions = predict(X);
        double mse = (predictions - Y).squaredNorm() / X.cols();
        return mse; }
     //Make predictions on new data
    Matrix predict(const Matrix& X) {
        network_function(X);
        return a3_;}
};

