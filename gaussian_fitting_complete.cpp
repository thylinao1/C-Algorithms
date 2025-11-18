/**
 * GAUSSIAN DISTRIBUTION FITTING
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using Vector = Eigen::VectorXd;

constexpr double PI = 3.14159265358979323846;

/**
 * Gaussian Probability Density Function (PDF)
 * @param x: Points where to evaluate
 * @param mu: Mean of the distribution
 * @param sig: Standard deviation
 * @return: PDF values at each point
 */
Vector gaussian_pdf(const Vector& x, double mu, double sig) {
    Vector result(x.size());
    double normalization = 1.0 / (std::sqrt(2.0 * PI) * sig);
    for (int i = 0; i < x.size(); ++i) {
        double diff = x(i) - mu;
        result(i) = normalization * std::exp(-diff * diff / (2.0 * sig * sig)); }
    return result;}

//Partial derivative of Gaussian with respect to mu
Vector d_gaussian_dmu(const Vector& x, double mu, double sig) {
    Vector f = gaussian_pdf(x, mu, sig);
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i) {
        result(i) = f(i) * (x(i) - mu) / (sig * sig);}
    return result;}

//Partial derivative of Gaussian with respect to std dev
Vector d_gaussian_dsig(const Vector& x, double mu, double sig) {
    Vector f = gaussian_pdf(x, mu, sig);
    Vector result(x.size());
    for (int i = 0; i < x.size(); ++i) {
        double diff = x(i) - mu;
        result(i) = f(i) * (-1.0 / sig + diff * diff / (sig * sig * sig));} 
    return result;}


// Optimization
/**
 * Perform one step of steepest descent (gradient descent)
 * @param x: Data points
 * @param y: Observed values
 * @param mu: Current mean estimate
 * @param sig: Current std dev estimate
 * @param learning_rate: Step size (α)
 * @return: Updated (mu, sigma)
 */
std::pair<double, double> steepest_step(const Vector& x, const Vector& y,
                                        double mu, double sig, double learning_rate) {
    //current model prediction
    Vector f = gaussian_pdf(x, mu, sig);
    //  error
    Vector residual = y - f;
    // partial derivatives of f
    Vector dfdmu = d_gaussian_dmu(x, mu, sig);
    Vector dfdsig = d_gaussian_dsig(x, mu, sig);
    //  gradients of x^2 using chain rule
    double grad_mu = -2.0 * residual.dot(dfdmu);
    double grad_sig = -2.0 * residual.dot(dfdsig);
    // Update parameters
    mu -= learning_rate * grad_mu;
    sig -= learning_rate * grad_sig;
    return {mu, sig};}

 // chi-squared error
double chi_squared(const Vector& x, const Vector& y, double mu, double sig) {
    Vector f = gaussian_pdf(x, mu, sig);
    return (y - f).squaredNorm();}


// Firring function
/**
 * Fit Gaussian distribution to data
 * 
 * @param x: Data points (e.g., heights)
 * @param y: Observed frequencies/densities
 * @param mu_init: Initial guess for mean
 * @param sig_init: Initial guess for standard deviation
 * @param learning_rate: Step size for gradient descent
 * @param iterations: Number of optimization steps
 * @param verbose: Print progress
 * @return: Optimized (mu, sigma)
 */
std::pair<double, double> fit_gaussian(const Vector& x, const Vector& y,
                                       double mu_init, double sig_init,double learning_rate, int iterations,
                                       bool verbose = true) 
{
    double mu = mu_init;
    double sig = sig_init;
    
    // Store optimization history
    std::vector<std::pair<double, double>> path;
    std::vector<double> errors;
    
    path.push_back({mu, sig});
    errors.push_back(chi_squared(x, y, mu, sig));

    // Optimization 
    for (int iter = 0; iter < iterations; ++iter) {
        // Perform one step
        auto [new_mu, new_sig] = steepest_step(x, y, mu, sig, learning_rate);
        mu = new_mu;
        sig = new_sig;
        // Track progress
        path.push_back({mu, sig});
        double error = chi_squared(x, y, mu, sig);
        errors.push_back(error);}
    return {mu, sig};
}