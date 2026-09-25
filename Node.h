#include "Op.h"

typedef struct Tape Tape;

typedef struct Node {
    Op operation;
    double value;
    double grad;
    struct Node* left;
    struct Node* right;
} Node; 

Node* add(Node* a, Node* b, Tape* tape);
Node* mul(Node* a, Node* b, Tape* tape);
Node* constant(double x, Tape* tape);
Node* relu(Node* x, Tape* tape);