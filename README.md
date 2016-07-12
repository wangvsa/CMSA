# CUDA-MSA
Improved Center Star Algorithm for Multiple Sequences Alignment Based on CUDA

Usage:

./msa.out [options] input_path output_path

Options:

    -g  : use GPU only (default use both GPU and CPU)
    -c  : use CPU only (default use both GPU and CPU)
    -w <float>  : specify the workload ratio of CPU / CPU
    -b <int>    : specify the number of blocks per grid
    -t <int>    : specify the number of threads per block
    -n <int>    : specify the number of GPU devices should be used

