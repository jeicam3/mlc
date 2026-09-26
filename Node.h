#include "Op.h"

typedef struct Tape Tape;

typedef struct Node {
    Op operation;
    double value;
    double grad;
    struct Node* left;
    struct Node* right;
} Node; 

Node* node_alloc(Node* left, Node* right, Tape* tape);
Node* node_add(Node* a, Node* b, Tape* tape);
Node* node_mul(Node* a, Node* b, Tape* tape);
Node* node_constant(double x, Tape* tape);
Node* node_relu(Node* x, Tape* tape);