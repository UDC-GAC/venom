# VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8084447.svg)](https://doi.org/10.5281/zenodo.8084447)

The V:N:M (VENOM) format enables the execution of arbitrary N:M ratios on SPTCs, which natively only support 2:4 patterns (50% sparsity). To efficiently exploit VENOM, we propose **Spatha** 🗡️, a high-performance sparse-library for DL routines. We ran all the experiments on NVIDIA RTX 3090 GPU. The software requirements to reproduce the artifact are: CUDA Toolkit 11.5 or 11.7, cuSparseLt v.0.3.0, Python 3.10, PyTorch 1.13.1 and cmake 3.16.3.

<p align="center"><img align="center" width="280" src="logo.jpeg"/></p>

# Reproduction with container
## Step 1: Download and run the container
### Option 1: download an already-built docker image
```bash
wget https://zenodo.org/record/8084447/files/venom_container.tar.gz
docker load -i venom_container.tar.gz
docker run -it –-gpus all venom_container
```

### Option 2: build the container from scratch
```bash
git clone --recurse-submodules git@github.com:UDC-GAC/venom.git && cd venom
docker build -t venom_container .
docker run -it --gpus all --name <your_container_name> venom_container
```

## Step 2: Compile and run the experiments

Compilation is already inlined in the scripts provided, so you can jump directly to (1) if you plan to follow the artifact scripts. However, the instructions to build and install the code are the following:

Build and install the centralized benchmarking tool:
```bash
cd /projects/venom/
mkdir build && cd build
# about 1 minute
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF && make -j 16
```

Three compiling options are defined to build the following kernel versions:
- -DBASELINE: baseline Spatha implementation for 2:4 sparsity
- -DIDEAL_KERNEL: Spatha N:M implementation without column-loc structure overhead (ideal situation)
- -DOUT_32B: Spatha N:M implementation with 32-bit storage instructions. By default 128-bit instructions are used.

Note: If you find a problem like this:
```
Policy "CMP0104" is not known to this version of CMake
```
Please, comment this line ```cmake_policy(SET CMP0104 OLD)``` in ```include/sputnik/CMakeLists.txt```

Build and install VENOM as a Python module:
```bash
cd end2end
# about 1 minute
./install.sh
```

(1) To reproduce the results on Fig 9
```bash
cd /projects/venom/

# about 1 hour
./benchmark/run_ablation1.sh

python plot/run_ablation1.py
```

(2) To reproduce the results on Fig 10

```bash
cd /projects/venom/

# about 5 minutes
./benchmark/run_ablation2.sh

python plot/run_ablation2.py
```

(3) To reproduce the results on Fig 12

```bash
cd /projects/venom/

# about 20 minutes
./benchmark/run_baseline_a.sh
./benchmark/run_baseline_b.sh

python plot/run_baseline_a.py
python plot/run_baseline_b.py
```

(4) To reproduce the results on Fig 13

```bash
cd /projects/venom/

# about 2 hours
./benchmark/run_spmm_spatha.sh

python plot/run_spmm_spatha.py
```

(5) To reproduce the results on Fig 15

```bash
conda activate end2end
# about 10 minutes
./end2end/run_inference.sh
python3 plot/run_inference.py
```

(6) To reproduce the results on Fig 11
```bash
conda activate end2end
# about 6 minutes
python3 benchmark/energy.py
```

(7) Since reproducing results on Table 2 can take a significant amount of time, we provide three different scripts to alleviate this process
```bash
conda activate sparseml_artf
cd sparseml
# Script that contains a subset of the experiments with the most aggressive configurations using the pair-wise version of the sparsifier
# about 4 days
./sparseml_SS1.sh
# Script that contains all the sparsity-format configurations but relaxed with pair-wise version of the sparsifier
# about 10 days
./sparseml_SS2.sh
# Script that contains all the sparsity-format configurations and performs the exhaustive search process
# about 25 days
./sparseml_SS3.sh
```
Note: each script in ```integrations/huggingface-transformers/scripts``` has two execution possibilities. Please, uncomment the first line if you want to use a single-GPU, or the second one with the total number of GPUs available for multiple-GPU execution.
```bash
#single-GPU
CUDA_VISIBLE_DEVICES=0 python3.10 src/sparseml/transformers/question_answering.py \
#multi-GPU (3 in this example)
python3.10 -m torch.distributed.launch --nproc_per_node=3 src/sparseml/transformers/question_answering.py \
````

## Step 3: check plots
```bash
cd /projects/venom/result
scp *.pdf username@hostmachine:/host/path/target
```

# Reproduction with source code
## Step 1: Prepare code and setup python environments
```bash
git clone --recurse-submodules git@github.com:UDC-GAC/venom.git && cd venom
```

Setup environments:
```bash
conda create -y --name end2end
conda activate end2end
conda install pytorch cudatoolkit torchvision torchaudio pytorch-cuda==11.7 -c pytorch -c nvidia
pip install pybind11 matplotlib pandas seaborn shapely holoviews
cd end2end/sten
pip install .
conda deactivate
```
```bash
cd sparseml
conda env create -f sparseml.yml
conda activate sparseml_artf
python3.10 -m pip install -e .
python3.10 uninstall transformers
python3.10 -m pip install https://github.com/neuralmagic/transformers/releases/download/v1.5/transformers-4.23.1-py3-none-any.whl datasets scikit-learn seqeval pulp
conda deactivate
```

## Step 2&3: Suppose the source code is in the path ```/projects/venom```. Then, follow the same ```Step 2&3``` instructions as described for docker containers

# How to use. Examples:

## Spatha 🗡️
```
./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m 1024 --k 4096 --n 4096 --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check
```

```
./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 16 --m 1024 --k 4096 --n 4096 --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check
```
## cuSparseLt
```
./src/benchmark_spmm --sparsity-type csr --spmm cuSparseLt --gemm cuBlas --precision half --m 1024 --k 4096 --n 768 --d 0.5 --check
```
## CLASP
```
./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 16 --m 1024 --k 256 --n 256 --d 0.2 --check
```

## Publication

VENOM is published in SC'23. To cite our work:
```bibtex
@inproceedings{castroR2023venom,
  author = {Roberto L. Castro, Andrei Ivanov, Diego Andrade, Tal Ben-Nun, Basilio B. Fraguela, and Torsten Hoefler},
  title = {VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores},
  booktitle = {Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC'23)},
  location = {Denver, CO},
  publisher = {IEEE Press},
  series = {SC'23},
  year = {2023}
}
```

## License
Apache-2.0 License

-- Roberto López Castro
--
