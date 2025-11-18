//PAGERANK ALGORITHM


#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Helper Functions

// Fix dangling nodes (nodes with no outgoing links)
Matrix fix_dangling_nodes(const Matrix& L) {
    Matrix L_fixed = L;
    const int n = L.rows();
    for (int col = 0; col < n; ++col) {
        // Check if column sums to zero
        if (L.col(col).sum() < 1e-10) {
            L_fixed.col(col) = Vector::Ones(n) / n;}
    }
    return L_fixed;}

// Google matrix
Matrix build_google_matrix(const Matrix& L, double damping) {
    const int n = L.rows();
    Matrix J = Matrix::Ones(n, n);
    Matrix M = damping * L + (1.0 - damping) / n * J;
    return M;}

 // L2 distance between two vectors
double compute_distance(const Vector& a, const Vector& b) {
    return (a - b).norm();}

// Mathod 1: Power Itration

/**
 * @param L: Link matrix (column-stochastic)
 * @param damping: Damping factor (typically 0.85)
 * @param max_iterations: Stop after this many iterations
 * @param tolerance: Convergence threshold
 * @param verbose: Print progress
 * @return: PageRank vector (normalized to sum to 100)
 */
Vector pagerank_power_iteration(const Matrix& L, double damping = 0.85,
                                int max_iterations = 1000,double tolerance = 1e-6,
                                bool verbose = true) {
    auto start = std::chrono::high_resolution_clock::now();
    
    const int n = L.rows();
    //  Google matrix
    Matrix M = build_google_matrix(L, damping);
    // Initialize with uniform distribution
    Vector r = Vector::Ones(n) / n;
    Vector r_prev;
    int iterations = 0;
    // Power iteration
    for (int iter = 0; iter < max_iterations; ++iter) {
        r_prev = r;
        r = M * r;
        // Normalize
        r /= r.sum();
        // Check convergence
        double distance = compute_distance(r, r_prev);
        
        if (distance < tolerance) {
            iterations = iter + 1;
            if (verbose) {
                std::cout << "\nConverged after " << iterations 
                          << " iterations!" << std::endl;}
            break;} 
        iterations = iter + 1;}
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (verbose) {
        std::cout << "Time: " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms\n" << std::endl;}
    
    // for intrpretability
    r *= 100.0;
    return r;}

// Method 2: Eigendecomposition (a bit slower for larage graphs)
/**
 * @param L: Link matrix
 * @param verbose: Print progress
 * @return: PageRank vector (normalized to sum to 100)
 */
Vector pagerank_eigendecomposition(const Matrix& L, bool verbose = true) {
    auto start = std::chrono::high_resolution_clock::now();
    const int n = L.rows();
    
    if (verbose) {
        std::cout << "Nodes: " << n << "\n" << std::endl;}
    // Compute eigenvalues and eigenvectors
    Eigen::EigenSolver<Matrix> solver(L);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "failed" << std::endl;
        return Vector::Zero(n);}
    
    // Get eigenvalues and eigenvectors 
    Eigen::VectorXcd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXcd eigenvectors = solver.eigenvectors();
    
    // Find index of dominant eigenvalue 
    int dominant_idx = 0;
    double min_dist = std::abs(eigenvalues(0).real() - 1.0);
    for (int i = 1; i < n; ++i) {
        double dist = std::abs(eigenvalues(i).real() - 1.0);
        if (dist < min_dist) {
            min_dist = dist;
            dominant_idx = i;}}
    if (verbose) {
        std::cout << "Dominant eigenvalue: " << std::fixed 
                  << std::setprecision(6) << eigenvalues(dominant_idx).real() 
                  << std::endl;}
    
    // Extract principal eigenvector
    Vector r(n);
    for (int i = 0; i < n; ++i) {
        r(i) = eigenvectors(i, dominant_idx).real();}
    // Make all entries positive and normalize
    r = r.cwiseAbs();
    r /= r.sum();
    r *= 100.0;
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (verbose) {
        std::cout << "Time: " << time_ms << " ms\n" << std::endl;} 
    return r;
}