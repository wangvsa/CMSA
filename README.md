# CMSA
CMSA is a robust and efficient MSA system for large-scale datasets on the heterogeneous CPU/GPU platform. It performs and optimizes multiple sequence alignment automatically for users’ submitted sequences without any assumptions. CMSA adopts the co-run computation model so that both CPU and GPU devices are fully utilized. Moreover, CMSA proposes an improved center star strategy that reduces the time complexity of its center sequence selection process from O(mn^2) to O(mn).

Usage:

./msa.out [options] input_path output_path

Options:

    -g  : use GPU only (default use both GPU and CPU)
    -c  : use CPU only (default use both GPU and CPU)
    -w <float>  : specify the workload ratio of CPU / CPU
    -b <int>    : specify the number of blocks per grid
    -t <int>    : specify the number of threads per block
    -n <int>    : specify the number of GPU devices should be used

