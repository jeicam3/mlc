#pragma once

#include "Op.h"
#include "Graph.h"

typedef struct Node {
    Op operation;
    double value;
    double grad;
    struct Node* left;
    struct Node* right;
} Node; 

Node* node_alloc(Node* left, Node* right, Graph* graph);
Node* node_add(Node* a, Node* b, Graph* graph);
Node* node_sub(Node* a, Node* b, Graph* graph);
Node* node_mul(Node* a, Node* b, Graph* graph);
Node* node_constant(double x, Graph* graph);
Node* node_relu(Node* x, Graph* graph);
void node_print(Node* node, int depth);