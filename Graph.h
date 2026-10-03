#pragma once

typedef struct Node Node;

#define GRAPH_ROOT(t) ((t)->ptrs[(t)->count-1])

typedef struct {
    Node** ptrs;
    int capacity;
    int count;
}Graph;

Graph* graph_alloc(size_t init_capacity);
void graph_free(Graph* graph);
void graph_add(Graph* graph, Node* node);
void graph_backward(Graph* graph);
void graph_forward(Graph* graph);
void graph_zero_grads(Graph* graph);
void graph_print(Graph* graph);