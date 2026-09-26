#include <stdlib.h>

#include "Tape.h"
#include "Node.h"

Node* node_alloc(Node* left, Node* right, Tape* tape) {
    Node* result = malloc(sizeof(Node));

    result->grad = 0;
    result->left = left;
    result->right = right;
    tape_add(tape, result);

    return result;
}

Node* node_add(Node* a, Node* b, Tape* tape) {
    Node* result = node_alloc(a, b, tape);

    result->value = a->value + b->value;
    result->operation = ADD;

    return result;
}

Node* node_mul(Node* a, Node* b, Tape* tape) {
    Node* result = node_alloc(a, b, tape);

    result->value = a->value * b->value;
    result->operation = MUL;

    return result;
}

Node* node_constant(double x, Tape* tape) {
    Node* result = node_alloc(NULL, NULL, tape);

    result->value = x;
    result->operation = CONSTANT;
    
    return result;
}

Node* node_relu(Node* x, Tape* tape) {
    Node* result = node_alloc(x, NULL, tape);
    
    result->value = x->value > 0 ? x->value : 0;
    result->operation = RELU;

    return result;
}