

#! /bin/bash

mkdir -p /projects/venom/result
log_file=/projects/venom/result/ablation2.csv

echo "benchmark.gemm.nm"

shapes="1024,4096,4096"

echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_row,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $log_file

cfgs_min1="
64,64,32,32,64,32,16,8,32,2 \
64,64,32,32,64,32,16,8,32,3 \
64,64,32,32,64,32,16,8,32,4 \
64,64,32,64,64,32,16,8,32,2 \
64,64,32,64,64,32,16,8,32,3 \
64,64,32,64,64,32,16,8,32,4 \
128,64,32,32,64,32,16,8,32,2 \
128,64,32,32,64,32,16,8,32,3 \
128,64,32,32,64,32,16,8,32,4 \
128,64,32,128,64,32,16,8,32,2 \
128,64,32,128,64,32,16,8,32,3 \
128,64,32,128,64,32,16,8,32,4 \
128,64,32,64,64,32,16,8,32,2 \
128,64,32,64,64,32,16,8,32,3 \
128,64,32,64,64,32,16,8,32,4 \
32,64,32,32,64,32,16,8,32,2 \
32,64,32,32,64,32,16,8,32,3 \
32,64,32,32,64,32,16,8,32,4"

mkdir build && cd build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=ON && make -j 16

for mm_row in 7 8 10 20 40 100; do
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

sed -i "s/2,sm/4,sm/" $log_file

make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF && make -j 16

for mm_row in 7 8 10 20 40 100; do
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
