/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>


void parseArgs(const int argc, const char** argv, int &m, int &n, int &k, float &density, int &block_sz, int &meta_block_sz, int &spmm_code, int &gemm_code, int &acc_code, int &pattern_code, int &precision_code, unsigned &seed, bool &check, int &bm, int &bn, int &bk, int &wm, int &wn, int &wk, int &mm, int &mn, int &mk, int &nstage, int &nn_row, int &mm_row, bool &load_pattern, std::string &path_to_dset, bool verbose=false)
{
    std::string Usage =
        "\tRequired cmdline args:\n\
        --sparsity-type [csr/cvs/n-to-m] \n\
        --spmm [CLASP/sputnik/spatha/cuSparseLt]\n\
        --gemm [cublas]\n\
        --precision [single, half, int8]\n\
        --m [M]\n\
        --k [K]\n\
        --n [N]\n\
        --path_to_dset [file]: load sparse matrix from file\n\
        --d [density between 0~1]\n\
        Optional cmdline args: \n\
        --block-size [2/4/8/32/64/128] \n\
        --meta-block-size [16] \n\
        --nn_row: n value on N:M format\n\
        --mm_row: m value on N:M format\n\
        --acc_t\n\
        --bm\n\
        --bn\n\
        --bk\n\
        --wm\n\
        --wn\n\
        --wk\n\
        --mm\n\
        --mn\n\
        --mk\n\
        --nstage\n\
        --random: set a random seed. by default 2022 for every run.\n\
        --check: check results correctness\n\
    \n";

    // default
    m = 0; n = 0; k = 0; density = 0.f; seed = 2022;
    block_sz = 1; spmm_code=-1; gemm_code=-1; pattern_code = -1; precision_code=-1; meta_block_sz=0; acc_code=0;
    check=false;
    bm=0, bn=0, bk=0;
    wm=0, wn=0; wk=0;
    mm=0, mn=0, mk=0;
    nstage=0;
    nn_row=0; mm_row=0;
    load_pattern = false;

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
        else if (!strcmp(argv[i], "--bm") && i!=argc-1) {
            bm = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--bn") && i!=argc-1) {
            bn = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--bk") && i!=argc-1) {
            bk = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--wm") && i!=argc-1) {
            wm = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--wn") && i!=argc-1) {
            wn = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--wk") && i!=argc-1) {
            wk = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--mm") && i!=argc-1) {
            mm = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--mn") && i!=argc-1) {
            mn = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--mk") && i!=argc-1) {
            mk = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--nstage") && i!=argc-1) {
            nstage = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--nn_row") && i!=argc-1) {
            nn_row = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--mm_row") && i!=argc-1) {
            mm_row = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--block-size") && i!=argc-1) {
            block_sz = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--meta-block-size") && i!=argc-1) {
            meta_block_sz = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--path_to_dset") && i!=argc-1) {
            load_pattern=true;
            path_to_dset = argv[i+1];
        }
        else if (!strcmp(argv[i], "--spmm") && i!=argc-1) {
            if (!strcmp(argv[i+1], "CLASP")){
                spmm_code = 0;
            } else if (!strcmp(argv[i+1], "sputnik")){
                spmm_code = 1;
            } else if (!strcmp(argv[i+1], "spatha")){
                spmm_code = 2;
            } else if (!strcmp(argv[i+1], "cuSparseLt")){
                spmm_code = 3;
            }
        }
        else if (!strcmp(argv[i], "--gemm") && i!=argc-1) {
            if (!strcmp(argv[i+1], "cuBlas")){
                gemm_code = 0;
            } else if (!strcmp(argv[i+1], "cuBlasLt")){
                gemm_code = 1;
            }
        }
        else if (!strcmp(argv[i], "--sparsity-type") && i!=argc-1) {
            if (!strcmp(argv[i+1], "csr")){
                pattern_code = 0;
            } else if (!strcmp(argv[i+1], "cvs")){
                pattern_code = 1;
            } else if (!strcmp(argv[i+1], "n-to-m")){
                pattern_code = 2;
            }
        }
        else if (!strcmp(argv[i], "--precision") && i!=argc-1) {
            if (!strcmp(argv[i+1], "single")){
                precision_code = 0;
            } else if (!strcmp(argv[i+1], "half")){
                precision_code = 1;
            } else if (!strcmp(argv[i+1], "int8")){
                precision_code = 3;
            }
        }
        else if (!strcmp(argv[i], "--acc_t") && i!=argc-1) {
            if (!strcmp(argv[i+1], "fp32")){
                acc_code = 0;
            } else if (!strcmp(argv[i+1], "fp16")){
                acc_code = 1;
            }
        }
        else if (!strcmp(argv[i], "--check") ) {
            check = true;
        } else if (!strcmp(argv[i], "--random") ) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
    }

    std::stringstream log;

    log     << "\narguments: m = " << m
            <<            "\nn = " << n
            <<            "\nk = " << k
            <<      "\ndensity = " << density
            <<         "\nseed = " << seed;

    log         << "\nspmm-kernel = "      << spmm_code
                << "\ngemm-kernel = "      << gemm_code
                << "\nsparsity-pattern = " << pattern_code
                << "\nprecision = "        << precision_code
                << "\ncheck results ? "   << check;

    log     <<   "\n" ;

    if (m == 0 || n==0 || k==0 || density==0.f) {
        std::cerr << Usage;
        std::cerr << "Forget to set m,n,k,density or the path to an input file? \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if (spmm_code==0 && (((block_sz!=2) && (block_sz!=4) && (block_sz!=8)) || pattern_code!=1)) {
        std::cerr << Usage;
        std::cerr << "Unsupported block size. Block size for CLASP must be 2,4,8. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if (spmm_code==1 && (pattern_code!=0 || block_sz!=1) ) {
        std::cerr << Usage;
        std::cerr << "Unsupported sparse format. Sparse format for Sputnik must be csr. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if (pattern_code == -1) {
        std::cerr << Usage;
        std::cerr << "sparsity-pattern is not given or unsupported.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if(acc_code==1 && spmm_code<2){
        precision_code=2;
    } else if((acc_code!=0) || (acc_code==1 && spmm_code>1)){
        std::cerr << Usage;
        std::cerr << "accumulator type unsupported.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }

    if (verbose) {
        std::cerr << log.str();
    }
}