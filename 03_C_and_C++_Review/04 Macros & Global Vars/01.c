#include <stdio.h>

// examples for each conditional macro
// #if
// #ifdef
// #ifndef
// #elif
// #else
// #endif

#define PI 3.14159
#define AREA(r) (PI * r * r)

#ifndef radius // if radius is not defined
#define radius 7 // we define radius and set it to 7
#endif

// if elif else logic
// we can only use integer constants in #if and #elif
#if radius > 10 // is radius > 10 ? False
#define radius 10
#elif radius < 5 // false
#define radius 5
#else
#define radius 7
#endif


int main() {
    printf("Area of circle with radius %d: %f\n", radius, AREA(radius));
    // Output: Area of circle with radius 7: 153.937910
    // radius is integer type so %d AREA is float type so %f
}