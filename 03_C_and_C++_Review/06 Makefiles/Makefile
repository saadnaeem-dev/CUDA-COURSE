# Makefile Used for automatin C,C++ compilation
# Usage: make 01 runs 01 target which then executes gcc -o 01 01.c
.PHONY: 01 01_obj 01_obj_exe_run 02 03 clean # This is a special target that tells make that the listed targets (e.g., 01, 02, clean) are not actual files but commands or tasks that should always run when invoked. This prevents make from confusing a target with a file of the same name.
# @ sign prevent printing out gcc -o 01 01.c to terminal
GCC = gcc # GNU C compiler
NVCC = nvcc
CUDA_FLAGS = -arch=sm_86 # sets the architecture of the CUDA GPU to compute capability 8.6.

01: # target 01 This target compiles the 01.c file into an executable named 01 using the gcc compiler ($(GCC) refers to gcc).
	@$(GCC) -o 01 01.c


# just compiles to object file
01_obj: # This target compiles the 01.c file into an object file (01.o) instead of a complete executable. The -c flag tells gcc to compile the source into an object file without linking.
	@$(GCC) -c 01.c -o 01.o

# compiles and runs the object file (ensure 01_obj is up to 
# date by running 01_obj first if it hasn't been run yet)
# if we run make 01_obj_exe_run, its going to call 01_obj target first which converts .c into object file by @$(GCC) -c 01.c -o 01.o and here we use that object file to create an executable @$(GCC) 01.o -o 01 then finally running that executable by @./01
01_obj_exe_run: 01_obj # target: dependencies The first command (@$(GCC) 01.o -o 01) links the object file 01.o into an executable 01. The second command (@./01) runs the 01 executable. This target automates the process of first creating the object file, linking it into an executable, and then running that executable.
	@$(GCC) 01.o -o 01
	@./01

02:
	@$(GCC) -o 02 02.c

03: # This target compiles 03.cu (a CUDA source file) into an executable 03_cu using the NVIDIA CUDA Compiler (nvcc). The output is the executable 03_cu. This runs nvcc -arch=sm_86 -o 03_cu 03.cu which can be run using ./03_cu we can also do make 03
	@$(NVCC) $(CUDA_FLAGS) -o 03_cu 03.cu

clean: # make clean
	rm -f 01 02 03_cu *.o
