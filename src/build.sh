nvcc main.cu cuda-nw.cu center-star.cc util.cu nw.cc global.cc -lcuda -Xcompiler -fopenmp -o msa.out
#nvcc fastautil.cc util.cu global.cc -o fastautil.out
#nvcc sort.cc util.cu global.cc -o sort.out
