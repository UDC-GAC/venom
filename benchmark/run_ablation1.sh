

#! /bin/bash

mkdir -p /projects/venom/result
log_file=/projects/venom/result/ablation1.csv

echo "benchmark.gemm.nm"

echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $log_file

shapes="
1024,768,4096 \
1024,1536,4096 \
1024,2304,4096 \
1024,3072,4096 \
1024,3840,4096 \
1024,4608,4096 \
1024,5376,4096 \
1024,6144,4096 \
1024,6912,4096 \
1024,7680,4096 \
1024,8448,4096 \
1024,9216,4096 \
1024,9984,4096 \
1024,10752,4096 \
1024,11520,4096 \
1024,12288,4096"

mkdir build && cd build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF && make -j 16

for shape in $shapes; do
    IFS=","; set -- $shape
    m=$1; k=$2; n=$3

    ./src/benchmark_spmm --sparsity-type csr --spmm cuSparseLt --gemm cuBlas --precision half --m $m --k $k --n $n --d 0.5 --check >> $log_file
done;

#uncomment the following configurations in order to use 128-bit SMEM stores

cfgs_min1="
128,64,32,32,64,32,16,8,32,2, \
128,64,32,32,64,32,16,8,32,3, \
128,64,32,32,64,32,16,8,32,4, \
128,64,32,128,64,32,16,8,32,2, \
128,64,32,128,64,32,16,8,32,3, \
128,64,32,128,64,32,16,8,32,4, \
128,64,32,64,64,32,16,8,32,2, \
128,64,32,64,64,32,16,8,32,3, \
128,64,32,64,64,32,16,8,32,4"

make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=ON -DOUT_32B=OFF && make -j 16

IFS=" ";
for mm_row in 10 20 40 100; do
    for shape in $shapes; do
        IFS=","; set -- $shape
        m=$1; k=$2; n=$3

        IFS=" ";
        for cfg in $cfgs_min1; do
            IFS=","; set -- $cfg

            bm=$1; bn=$2; bk=$3; wm=$4; wn=$5; wk=$6; mm=$7; mn=$8; mk=$9; nstage=${10};

            ./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row $mm_row --m $m --k $k --n $n --d 0.5 --bm $bm --bn $bn --bk $bk --wm $wm --wn $wn --wk $wk --mm $mm --mn $mn --mk $mk --nstage $nstage >> $log_file
        done;
        IFS=" ";
    done;
    IFS=" ";
done;

sed -i "s/2,sm/7,sm/" $log_file

make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF && make -j 16

IFS=" ";
for mm_row in 10 20 40 100; do
    for shape in $shapes; do
        IFS=","; set -- $shape
        m=$1; k=$2; n=$3

        IFS=" ";
        for cfg in $cfgs_min1; do
            IFS=","; set -- $cfg

            bm=$1; bn=$2; bk=$3; wm=$4; wn=$5; wk=$6; mm=$7; mn=$8; mk=$9; nstage=${10};

            ./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row $mm_row --m $m --k $k --n $n --d 0.5 --bm $bm --bn $bn --bk $bk --wm $wm --wn $wn --wk $wk --mm $mm --mn $mn --mk $mk --nstage $nstage --check >> $log_file
        done;
        IFS=" ";
    done;
    IFS=" ";
done;

#uncomment the following configurations in order to use 32-bit SMEM stores

#cfgs_min1="
#64,32,32,64,32,32,16,8,32,2 \
#64,32,32,64,32,32,16,8,32,3 \
#64,32,32,64,32,32,16,8,32,4 \
#64,64,32,64,32,32,16,8,32,2 \
#64,64,32,64,32,32,16,8,32,3 \
#64,64,32,64,32,32,16,8,32,4 \
#64,64,32,32,32,32,16,8,32,2, \
#64,64,32,32,32,32,16,8,32,3, \
#64,64,32,32,32,32,16,8,32,4, \
#128,32,32,128,32,32,16,8,32,2 \
#128,64,32,64,32,32,16,8,32,2 \
#128,64,32,64,32,32,16,8,32,3 \
#128,64,32,64,32,32,16,8,32,4 \
#128,64,32,64,32,32,16,8,32,5 \
#128,64,32,64,32,32,16,8,32,6 \
#128,64,32,128,32,32,16,8,32,2 \
#128,64,32,128,32,32,16,8,32,3 \
#128,64,32,128,32,32,16,8,32,4 \
#128,128,32,64,32,32,16,8,32,2 \
#128,128,32,64,32,32,16,8,32,4 \
#128,128,32,64,64,32,16,8,32,2 \
#128,128,32,64,64,32,16,8,32,4 \
#128,128,32,128,64,32,16,8,32,2 \
#128,128,32,128,128,32,16,8,32,2 \
#128,128,32,128,128,32,16,8,32,4 \
#128,64,32,32,64,32,16,8,32,2, \
#128,64,32,32,64,32,16,8,32,3, \
#128,64,32,32,64,32,16,8,32,4, \
#128,64,32,64,16,32,16,8,32,2, \
#128,64,32,64,16,32,16,8,32,3, \
#128,64,32,64,16,32,16,8,32,4, \
#128,32,32,32,32,32,16,8,32,2, \
#128,32,32,32,32,32,16,8,32,3, \
#128,64,32,32,32,32,16,8,32,2, \
#128,64,32,32,32,32,16,8,32,3, \
#128,64,32,32,32,32,16,8,32,4, \
#128,128,32,32,64,32,16,8,32,2, \
#128,128,32,32,64,32,16,8,32,4, \
#128,64,32,128,64,32,16,8,32,2, \
#128,64,32,128,64,32,16,8,32,3, \
#128,64,32,128,64,32,16,8,32,4, \
#128,64,32,64,64,32,16,8,32,2, \
#128,64,32,64,64,32,16,8,32,3, \
#128,64,32,64,64,32,16,8,32,4, \
#128,128,32,64,128,32,16,8,32,2, \
#128,128,32,64,128,32,16,8,32,3, \
#128,128,32,64,128,32,16,8,32,4, \
#64,128,32,64,64,32,16,8,32,2, \
#64,128,32,64,64,32,16,8,32,3, \
#64,128,32,64,64,32,16,8,32,4, \
#64,128,32,32,64,32,16,8,32,2, \
#64,128,32,32,64,32,16,8,32,3, \
#64,128,32,32,64,32,16,8,32,4, \
#64,128,32,32,128,32,16,8,32,2, \
#64,128,32,32,128,32,16,8,32,3, \
#64,128,32,32,128,32,16,8,32,4, \
#64,128,32,64,128,32,16,8,32,2, \
#64,128,32,64,128,32,16,8,32,3, \
#64,128,32,64,128,32,16,8,32,4"