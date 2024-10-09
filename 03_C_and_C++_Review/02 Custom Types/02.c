// exact same as 02.cpp
#include <stdio.h>

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.1, 2.5};
    printf("size of Point: %zu\n", sizeof(Point));  // Output: 8 bytes = 4 bytes (float x) + 4 bytes (float y)
    printf("Value of p.x: %.1f\n",p.x); // float 32 numbers occupy 4 bytes

}