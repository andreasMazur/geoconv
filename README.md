# Masterarbeit

compilation:

gcc -I/usr/include/python3.10 -I/home/andreas/programs/c_libraries test.c -lpython3.10 /home/andreas/programs/c_libraries/cblas_LINUX.a /home/andreas/programs/c_libraries/libblas.a -lm

gcc -I/usr/include/python3.10 -I/home/andreas/programs/anaconda3/envs/Masterarbeit/lib/python3.10/site-packages/numpy/core/include -I/home/andreas/programs/c_libraries c_extension.c -lpython3.10 /home/andreas/programs/c_libraries/libcblas.a /home/andreas/programs/c_libraries/libblas.a -lm


https://canvas.kth.se/courses/24933/pages/tutorial-blas-library-and-mkl
https://askubuntu.com/questions/1270161/how-to-build-and-link-blas-and-lapack-libraries-by-hand-for-use-on-cluster