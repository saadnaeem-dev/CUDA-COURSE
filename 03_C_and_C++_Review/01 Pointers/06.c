#include <stdio.h>
// matrix -> arr -> integers
// similar to 01.c but with arrays

int main() {
    int arr1[] = {1, 2, 3, 4};
    int arr2[] = {5, 6, 7, 8};
    int* ptr1 = arr1;
    int* ptr2 = arr2;
    int* matrix[] = {ptr1, ptr2}; // 0, 1 < 2 matrix[0] & matrix[1] ptr1 & ptr2 points to the first element of each array respectively

    for (int i = 0; i < 2; i++) { // position i gives the memory address
        for (int j = 0; j < 4; j++) { // once at 1st row we print & ++ 4 times to get [1, 2, 3, 4]. decides how many times to run
            printf("%d ", *matrix[i]++);
        }
        printf("\n");
    }
    //    *matrix[i] dereferences the pointer matrix[i], giving you the value at the current position in that row.
    //    matrix[i]++ increments the pointer matrix[i], advancing it to the next element in the row after dereferencing it. This means that after printing the current value, the pointer moves to the next element in the row.
    //    Initially, matrix[0] points to arr1, which holds {1, 2, 3, 4}
    //    In the first iteration of the inner loop (j loop), *matrix[0] gives 1, and matrix[0]++ advances the pointer to point to 2
}