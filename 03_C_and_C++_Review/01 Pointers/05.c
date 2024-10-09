#include <stdio.h>

int main() {
    // 8 bits in a byte times 4 bytes we get 32 bits which is integer 32 type classical int 32 this is how its laid out in the memory
    int arr[] = {12, 24, 36, 48, 60}; // arr itself is a pointer to the first element of the array

    int* ptr = arr;  // ptr points to the first element of arr (default in C)

    printf("Position one: %d\n", *ptr);  // Output: 12

    for (int i = 0; i < 5; i++) {
        printf("%d\t", *ptr);
        printf("%p\t", ptr);
        printf("%p\n", &ptr);
        ptr++; // Jumping 32 bits
    }
    // Output: 
    // Position one: 12
    // disclaimer: the memory addresses won't be the same each time you run
    // 12 0x7fff773fe780
    // 24 0x7fff773fe784
    // 36 0x7fff773fe788
    // 48 0x7fff773fe78c
    // 60 0x7fff773fe790

    // notice that the pointer is incremented by 4 bytes (size of int = 4 bytes * 8 bits/bytes = 32 bits = int32) each time. 
    // ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 (4 GB) which is too small given how much memory we typically have.
    // arrays are layed out in memory in a contiguous manner (one after the other rather than at random locations in the memory grid)
    //
    //    Position one: 12
    //12      0x7ffda8186820  0x7ffda8186818 this part remains the same as its the address to the pointer
    //24      0x7ffda8186824  0x7ffda8186818
    //36      0x7ffda8186828  0x7ffda8186818
    //48      0x7ffda818682c  0x7ffda8186818
    //60      0x7ffda8186830  0x7ffda8186818


}