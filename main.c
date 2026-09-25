#include <stdio.h>

#include "Node.h"
#include "Tape.h"

int main() {
    Tape* tape = tape_alloc(16);
    Node* x = constant(2, tape);     
    Node* y = constant(4, tape);     
    Node* z = constant(-6, tape);
    
    Node* x_plus_y = add(x, y, tape);
    Node* mul_by_z = mul(x_plus_y, z, tape);
    Node* ReLU = relu(mul_by_z, tape);
    printf("ReLU((x+y) * z) = %f\n", ReLU->value);
}