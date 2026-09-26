#include <stdio.h>

#include "Node.h"
#include "Tape.h"

int main() {
    Tape* tape = tape_alloc(16);
    Node* x = node_constant(2, tape);     
    Node* y = node_constant(4, tape);     
    Node* z = node_constant(-6, tape);
    
    Node* x_plus_y = node_add(x, y, tape);
    Node* mul_by_z = node_mul(x_plus_y, z, tape);
    Node* ReLU = node_relu(mul_by_z, tape);
    printf("ReLU((x+y) * z) = %f\n", ReLU->value);
    tape_free(tape);
}