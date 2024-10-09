// Purpose: Demonstrate NULL pointer initialization and safe usage.

// Key points:
// 1. Initialize pointers to NULL when they don't yet point to valid data.
// 2. Check pointers for NULL before using to avoid crashes.
// 3. NULL checks allow graceful handling of uninitialized or failed allocations.

#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize pointer to NULL
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    // Check for NULL before using
    if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate memory
    ptr = malloc(sizeof(int)); // returns void pointer, the size of int. there is something there now. it doesn't have an explicit data type (32 bits 4 bytes of something)
    if (ptr == NULL) {
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation, ptr value: %p\n", (void*)ptr); // we know this exists no so we can use it for something

    // Safe to use ptr after NULL check
    *ptr = 42; // dereference to set the value for this address
    printf("5. Value at ptr: %d\n", *ptr);

    // Clean up
    free(ptr); // freeing up the pointer
    ptr = NULL;  // Set to NULL after freeing

    printf("6. After free, ptr value: %p\n", (void*)ptr); // use typecast void* to reference a NULL pointer

    // Demonstrate safety of NULL check after free
    if (ptr == NULL) {
        printf("7. ptr is NULL, safely avoided use after free\n");
    }

    return 0;
}
// We can use NULL pointers to do little tricks to make our code more robust. By checking if its null we can avoid running into unexpected errors like segmentation fault (seg fault and other weired things)
// 1. Initial ptr value: (nil)
// 2. ptr is NULL, cannot dereference
// 4. After allocation, ptr value: 0x56480a49d6b0
// 5. Value at ptr: 42
// 6. After free, ptr value: (nil)
// 7. ptr is NULL, safely avoided use after free