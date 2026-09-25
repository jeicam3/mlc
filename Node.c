#include <stdlib.h>

#include "Tape.h"
#include "Node.h"

Node* add(Node* a, Node* b, Tape* tape) {
    Node* result = malloc(sizeof(Node));

    result->value = a->value + b->value;
    result->grad = 0;
    result->left = a;
    result->right = b;

    result->operation = ADD;
    tape_add(tape, result);
    return result;
}

Node* mul(Node* a, Node* b, Tape* tape) {
    //może tutaj dać makro? - generalnie to będzie często powtarzający się kod i można to dać razem z zerowaniem grad bo to też będzie stałe
    Node* result = malloc(sizeof(Node));

    result->value = a->value * b->value;
    result->grad = 0;
    result->left = a;
    result->right = b;

    result->operation = MUL;
    tape_add(tape, result);
    return result;
}

Node* constant(double x, Tape* tape) {
    Node* result = malloc(sizeof(Node));

    result->value = x;
    result->grad = 0;
    result->left = NULL;
    result->right = NULL;

    result->operation = CONSTANT;
    tape_add(tape, result);
    return result;
}

Node* relu(Node* x, Tape* tape) {
    Node* result = malloc(sizeof(Node));
    
    result->value = x->value > 0 ? x->value : 0;
    result->grad = 0;
    result->left = x;
    result->right = NULL;

    result->operation = RELU;
    tape_add(tape, result);
    return result;
}