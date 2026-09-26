#include <stdlib.h>

#include "Graph.h"
#include "Node.h"

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