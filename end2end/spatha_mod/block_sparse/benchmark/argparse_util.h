#pragma once

#include <string>
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <random>
#include <chrono>       // timeseed


void parseSpmmArgs(const int argc, const char** argv,
    int &m, int &n, int &k, float &density, unsigned &seed,
    int &pattern_code, int &block_sz, int &meta_block_sz, bool verbose=false)
{
    std::string Usage =
        "\tRequired cmdline args:\n\
        --m [M]\n\
        --n [N]\n\
        --k [K]\n\
        --d [density between 0~1]\n\
        --sparsity-type [block-2in4] \n\
        --block-size [32/64/128/256] \n\
        --meta-block-size [16] \n\
        Optional cmdline args: \n\
        --random: set a random seed. by default 2022 for every run.\n\
    \n";
    // default
    m = 0; n = 0; k = 0; density = 0.5f; seed = 2022;
    meta_block_sz = 0; block_sz = 0; pattern_code = -1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m") && i!=argc-1) {
            m = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--n") && i!=argc-1) {
            n = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--k") && i!=argc-1) {
            k = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--d") && i!=argc-1) {
            density = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--block-size") && i!=argc-1) {
            block_sz = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--meta-block-size") && i!=argc-1) {
            meta_block_sz = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--sparsity-type") && i!=argc-1) {
            if (!strcmp(argv[i+1], "block-2in4"))
                pattern_code = 0;
        }
        else if (!strcmp(argv[i], "--random") ) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
    }

    std::stringstream log;

    if (m == 0 || n==0 || k==0 ) {
        std::cerr << Usage;
        std::cerr << "Forget to set m,n or k? \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (block_sz<32) {
        std::cerr << Usage;
        std::cerr << "Unsupported block size. Choose in [32/64/128/256]. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (meta_block_sz!=16) {
        std::cerr << Usage;
        std::cerr << "Unsupported block size. Choose in [16]. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (pattern_code == -1) {
        std::cerr << Usage;
        std::cerr << "sparsity-pattern is not given or unsupported.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if (verbose) {
        std::cerr << log.str();
    }
}