#include <stdio.h>

int main(int argc, char **argv)
{
    int d = 2;
    printf("welcome to the program with a bug!\n");
    scanf("%d", &d);
    printf("You gave me: %d", d);
    return 0;
}
// (base) saad@Saad-AI-Machine:/mnt/c/Users/saadn/PycharmProjects/cuda-course/03_C_and_C++_Review/07 Debuggers$ gcc -o debug debug.c -g
// debug.c: In function ‘main’:
// debug.c:7:13: warning: format ‘%d’ expects argument of type ‘int *’, but argument 2 has type ‘int’ [-Wformat=]
//     7 |     scanf("%d", d);
//       |            ~^   ~
//       |             |   |
//       |             |   int
//       |             int *
// int* expected int provided
// (base) saad@Saad-AI-Machine:/mnt/c/Users/saadn/PycharmProjects/cuda-course/03_C_and_C++_Review/07 Debuggers$ file debug
// debug: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=75409a0bc64420a3acc36bdf6e07339cedf572e1, for GNU/Linux 3.2.0, with debug_info, not stripped
// (base) saad@Saad-AI-Machine:/mnt/c/Users/saadn/PycharmProjects/cuda-course/03_C_and_C++_Review/07 Debuggers$