// Created by Omar Oubari <omar.oubari@inserm.fr>
//
// Information: loading datasets and creating lightweight corsesets

#pragma once

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <random>

// external dependency
#include "blaze/Blaze.h"
#include "third_party/numpy.hpp"
#include "third_party/filesystem.hpp"

template <typename T>
void build_coreset(dataset<T>& set, int Nprime, int nthreads) {
    
    std::vector<std::mt19937> mt(nthreads);
    Tp threads(nthreads);
    
    auto N = set.shape.first;
    auto D = set.shape.second;
    
    blaze::DynamicVector<T, blaze::rowVector> u(D, static_cast<T>(0.0));
    blaze::DynamicVector<T, blaze::rowVector> q(N);
    
    auto coreset = blaze::DynamicMatrix<T, blaze::rowMajor>(Nprime, set.shape.second);
    set.weight.resize(Nprime);

    // compute mean
    if (!blaze::isEmpty(set.data)) {
        for (int n = 0; n < N; n++) {
            u += blaze::row(set.data, n);
        }
    } else {
        // if no data then stream from vector of files
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto file: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(file, tmp_data);
            
            // loop through temp data matrix and add to u without using too much memory
            for (auto i=0; i<tmp_data.rows(); ++i) {
                u += blaze::row(tmp_data, i);
            }
            
            // clear matrix
            blaze::clear(tmp_data);
        }
    }
    u *= static_cast<T>(1.0)/N;
    
    // compute proposal distribution
    if (!blaze::isEmpty(set.data)) {
        threads.parallel(N, [&] (int n, int t) {
            q[n] = blaze::sqrNorm(blaze::row(set.data, n) - u);
        });
    } else {
        size_t shift = 0;
        blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
        for (auto file: set.files) {
            // read npy file
            loadBlazeFromNumpy<T>(file, tmp_data);
            
            // fill in q matrix without using much memory
            auto temp_N = static_cast<int>(tmp_data.rows());
            threads.parallel(temp_N, [&] (int i, int t) {
                size_t n = i+shift;
                q[n] = blaze::sqrNorm(blaze::row(tmp_data, i) - u);
            });
            
            // shift by the last number of data points
            shift += temp_N;
            
            
            // clear temporary containers
            blaze::clear(tmp_data);
        }
    }
    
    T sum = static_cast<T>(0.0);
    for (int n = 0; n < N; n++) {
        sum += q[n];
    }

    threads.parallel(N, [&] (int n, int t) {
        q[n] = static_cast<T>(0.5) * (q[n] / sum + static_cast<T>(1.0) / N);
    });


    std::random_device r;
    for (int t = 0; t < threads.size(); t++) {
        std::seed_seq seq{r(), r(), r(), r(), r(), r(), r(), r()};
        mt[t].seed(seq);
        mt[t].discard(1e3);
    }
                      
    std::discrete_distribution<int> dst(q.begin(), q.end());
    if (!blaze::isEmpty(set.data)) {
        threads.parallel(Nprime, [&] (int m, int t) {
            // get sample and fill coreset
            int n = dst(mt[t]);
            blaze::row(coreset, m) = blaze::row(set.data, n);
            set.weight[m] = static_cast<T>(1.0) / (q[n] * Nprime);
        });
    } else {
        threads.parallel(Nprime, [&] (int m, int t) {
            // get sample
            int n = dst(mt[t]);
            
            // find index of closest element to n such than element < n. this will help us identify which file contains the index n
            auto upper = std::upper_bound(set.file_search.begin(), set.file_search.end(), n);
            auto idx = std::distance(set.file_search.begin(), upper) - 1;

            // read npy file
            blaze::DynamicMatrix<T, blaze::rowMajor> tmp_data;
            loadBlazeFromNumpy<T>(set.files[idx], tmp_data);

            // fill coreset
            blaze::row(coreset, m) = blaze::row(tmp_data, n-set.file_search[idx]);
            set.weight[m] = static_cast<T>(1.0) / (q[n] * Nprime);
        });
    }
    
    // replace data with coreset
    blaze::clear(set.data);
    set.data = coreset;
    set.data.shrinkToFit();
    set.shape.first = Nprime;
}
