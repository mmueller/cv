// Quick & dirty test suite.

#include <stdio.h>

#include "Filter.h"

int testGaussian() {
    printf("A 5x5 gaussian kernel (sigma=1):\n");
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            printf("%g\t", gaussian(x, y));
        }
        printf("\n");
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int result = 0;
    result |= testGaussian();
    return result;
}
