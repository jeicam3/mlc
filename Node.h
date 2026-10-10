#pragma once

#include "Op.h"
#include "Graph.h"

#define OP_IMPLEMENTATION
#define GRAPH_IMPLEMENTATION

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

#ifdef NODE_IMPLEMENTATION

Node* node_alloc(Node* left, Node* right, Graph* graph) {
    Node* result = malloc(sizeof(Node));

    result->grad = 0;
    result->left = left;
    result->right = right;
    graph_add(graph, result);

    return result;
}

Node* node_add(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, graph);
    result->operation = ADD;
    return result;
}

Node* node_sub(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, graph);
    result->operation = SUB;
    return result;
}

Node* node_mul(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, graph);
    result->operation = MUL;
    return result;
}

Node* node_constant(double x, Graph* graph) {
    Node* result = node_alloc(NULL, NULL, graph);
    result->value = x;
    result->operation = CONSTANT;
    return result;
}

Node* node_relu(Node* x, Graph* graph) {
    Node* result = node_alloc(x, NULL, graph);    
    result->operation = RELU;
    return result;
}

void node_print(Node* node, int depth) {
    if (!node) return;

    for (int i = 0; i < depth; i++) {
        printf("   ");
    }
    printf("Operation: %s | Value: %.3f | Grad: %.3f\n", op_to_str(node->operation), node->value, node->grad);

    node_print(node->left, depth + 1);
    node_print(node->right, depth + 1);
}

#endif