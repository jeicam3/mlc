#pragma once
#include "tensorc.h"
#include <stdio.h>

typedef enum { CONSTANT, ADD, SUB, MUL, MATMUL, RELU } Op;

typedef struct Node {
    Op operation;
    Tensor* value;
    Tensor* grad;
    struct Node* left;
    struct Node* right;
} Node; 

#define GRAPH_ROOT(t) ((t)->ptrs[(t)->count-1])

typedef struct {
    Node** ptrs;
    int capacity;
    int count;
} Graph;

Graph* graph_alloc(size_t init_capacity);
void graph_free(Graph* graph);
void graph_add(Graph* graph, Node* node);
void graph_backward(Graph* graph);
void graph_forward(Graph* graph);
void graph_zero_grads(Graph* graph);
void graph_print(Graph* graph);

Node* node_alloc(Node* left, Node* right, int* shape, int ndim, Graph* graph);
Node* node_add(Node* a, Node* b, Graph* graph);
Node* node_sub(Node* a, Node* b, Graph* graph);
Node* node_mul(Node* a, Node* b, Graph* graph);
Node* node_matmul(Node* a, Node* b, Graph* graph);
Node* node_constant(float* data, int* shape, int ndim, Graph* graph);
Node* node_relu(Node* x, Graph* graph);
void node_print(Node* node, int depth);

const char* op_to_str(Op op);

#ifdef MLC_IMPLEMENTATION

const char* op_to_str(Op op) {
    switch (op) {
        case ADD: return "ADD";
        case SUB: return "SUB";
        case MUL: return "MUL";
        case MATMUL: return "MATMUL";
        case RELU: return "RELU";
        case CONSTANT: return "CONSTANT";
        default: return "UNKNOWN";
    }
}

Node* node_alloc(Node* left, Node* right, int* shape, int ndim, Graph* graph) {
    Node* result = malloc(sizeof(Node));
    result->value = tensor_alloc(shape, ndim);
    result->grad  = tensor_alloc(shape, ndim);
    result->left = left;
    result->right = right;
    graph_add(graph, result);
    return result;
}

Node* node_add(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, a->value->shape, a->value->ndim, graph);
    result->operation = ADD;
    return result;
}

Node* node_sub(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, a->value->shape, a->value->ndim, graph);
    result->operation = SUB;
    return result;
}

Node* node_mul(Node* a, Node* b, Graph* graph) {
    Node* result = node_alloc(a, b, a->value->shape, a->value->ndim, graph);
    result->operation = MUL;
    return result;
}

Node* node_matmul(Node* a, Node* b, Graph* graph) {
    int out_shape[2] = {a->value->shape[0], b->value->shape[1]};
    Node* result = node_alloc(a, b, out_shape, 2, graph);
    result->operation = MATMUL;
    return result;
}

Node* node_constant(float* data, int* shape, int ndim, Graph* graph) {
    Node* result = node_alloc(NULL, NULL, shape, ndim, graph);
    result->operation = CONSTANT;
    memcpy(result->value->data, data, result->value->size * sizeof(float));
    return result;
}

Node* node_relu(Node* x, Graph* graph) {
    Node* result = node_alloc(x, NULL, x->value->shape, x->value->ndim, graph);    
    result->operation = RELU;
    return result;
}

void node_print(Node* node, int depth) {
    if (!node) return;
    for (int i = 0; i < depth; i++) printf("   ");
    printf("Operation: %s | Value: %.3f | Grad: %.3f\n", op_to_str(node->operation), node->value->data[0], node->grad->data[0]);
    node_print(node->left, depth + 1);
    node_print(node->right, depth + 1);
}

Graph* graph_alloc(size_t init_capacity) {
    Graph* result = malloc(sizeof(Graph));
    result->capacity = init_capacity;
    result->ptrs = malloc(sizeof(Node*)*init_capacity);
    result->count = 0;
    return result;
}

void graph_free(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        tensor_free(graph->ptrs[i]->value);
        tensor_free(graph->ptrs[i]->grad);
        free(graph->ptrs[i]);
    }
    free(graph->ptrs);
    free(graph);
}

void graph_add(Graph* graph, Node* node) {
    if (graph->count == graph->capacity) {
        graph->capacity += (graph->capacity >> 1); 
        graph->ptrs = realloc(graph->ptrs, sizeof(Node*) * graph->capacity);
    }
    graph->ptrs[graph->count++] = node;
}

void graph_forward(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        Node* curr = graph->ptrs[i];
        switch (curr->operation) {
            case ADD: tensor_add(curr->value, curr->left->value, curr->right->value); break;
            case SUB: tensor_sub(curr->value, curr->left->value, curr->right->value); break;
            case MUL: tensor_mul(curr->value, curr->left->value, curr->right->value); break;
            case MATMUL: tensor_matmul(curr->value, curr->left->value, curr->right->value); break;
            case RELU: tensor_relu(curr->value, curr->left->value); break;
            case CONSTANT: break;
        }
    }
}

// Ten krok zrefaktoryzujemy przy wprowadzaniu funkcji backward do tensorc.h
void graph_backward(Graph* graph) {
    for (int j = 0; j < GRAPH_ROOT(graph)->grad->size; j++) {
        GRAPH_ROOT(graph)->grad->data[j] = 1.0f;
    }
    
    for (int i = graph->count-1; i >= 0; i--) {
        Node* curr = graph->ptrs[i];
        switch (curr->operation) {
            case ADD:
                for (int j = 0; j < curr->value->size; j++) {
                    curr->left->grad->data[j]  += curr->grad->data[j];
                    curr->right->grad->data[j] += curr->grad->data[j];
                }
                break;
            case SUB:
                for(int j = 0; j < curr->value->size; j++) {
                    curr->left->grad->data[j] += curr->grad->data[j];
                    curr->right->grad->data[j] -= curr->grad->data[j];
                }
                break;
            case MUL:
                for(int j = 0; j < curr->value->size; j++) {
                    curr->left->grad->data[j] += curr->grad->data[j] * curr->right->value->data[j];
                    curr->right->grad->data[j] += curr->grad->data[j] * curr->left->value->data[j];
                }
                break;
            case MATMUL: {
                int M = curr->left->value->shape[0];
                int K = curr->left->value->shape[1];
                int N = curr->right->value->shape[1];

                float* dC = curr->grad->data;
                float* A = curr->left->value->data;
                float* B = curr->right->value->data;
                float* dA = curr->left->grad->data;
                float* dB = curr->right->grad->data;

                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        for (int k = 0; k < K; k++) {
                            dA[m * K + k] += dC[m * N + n] * B[k * N + n];
                        }
                    }
                }

                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        for (int k = 0; k < K; k++) {
                            dB[k * N + n] += A[m * K + k] * dC[m * N + n];
                        }
                    }
                }
                break;
            }
            case RELU:
                for(int j = 0; j < curr->value->size; j++) {
                    curr->left->grad->data[j] += (curr->left->value->data[j] > 0 ? 1.0f : 0.0f) * curr->grad->data[j];
                }
                break;
            case CONSTANT: break;
        }
    }
}

void graph_zero_grads(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        int padded_size = (graph->ptrs[i]->grad->size + 7) & ~7;
        memset(graph->ptrs[i]->grad->data, 0, padded_size * sizeof(float));
    }
}

void graph_print(Graph* graph) {
    node_print(GRAPH_ROOT(graph), 0);
}

#endif