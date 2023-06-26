#! /bin/bash

mkdir -p /projects/venom_artifact/result
log_file=/projects/venom_artifact/result/inference.csv

cd end2end

echo "algo,n,m,v,mean,median,std,len" > $log_file

./install_v64.sh
###############
python bert_pytorch.py -m 8 -v 64 >> $log_file

python bert_pytorch.py -m 16 -v 64 >> $log_file

python bert_pytorch.py -m 32 -v 64 >> $log_file
##
##
python gpt2_pytorch.py -m 8 -v 64 >> $log_file

python gpt2_pytorch.py -m 16 -v 64 >> $log_file

python gpt2_pytorch.py -m 32 -v 64 >> $log_file
##
##
python gpt3_pytorch.py -m 8 -v 64 >> $log_file

python gpt3_pytorch.py -m 16 -v 64 >> $log_file

python gpt3_pytorch.py -m 32 -v 64 >> $log_file

###############
./install.sh
###############
python bert_pytorch.py -m 8 -v 128 >> $log_file

python bert_pytorch.py -m 16 -v 128 >> $log_file

python bert_pytorch.py -m 32 -v 128 >> $log_file
##
##
python gpt2_pytorch.py -m 8 -v 128 >> $log_file

python gpt2_pytorch.py -m 16 -v 128 >> $log_file

python gpt2_pytorch.py -m 32 -v 128 >> $log_file
##
##
python gpt3_pytorch.py -m 8 -v 128 >> $log_file

python gpt3_pytorch.py -m 16 -v 128 >> $log_file

python gpt3_pytorch.py -m 32 -v 128 >> $log_file
###############