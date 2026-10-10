#include "Node.h"

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

#ifdef GRAPH_IMPLEMENTATION

Graph* graph_alloc(size_t init_capacity) {    
    Graph* result = malloc(sizeof(Graph));
    result->capacity = init_capacity;
    result->ptrs = malloc(sizeof(Node*)*init_capacity);
    result->count = 0;
    return result;
}

void graph_free(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        free(graph->ptrs[i]);
    }
    free(graph->ptrs);
    free(graph);
}

void graph_add(Graph* graph, Node* node) {
    if (graph->count == graph->capacity) {
        graph->capacity = graph->capacity + (graph->capacity >> 1); 
        graph->ptrs = realloc(graph->ptrs, sizeof(Node*) * graph->capacity);
    }
    graph->ptrs[graph->count++] = node;
}

void graph_backward(Graph* graph) {
    GRAPH_ROOT(graph)->grad = 1;
    for (int i = graph->count-1; i >= 0; i--) {
        Node* curr_node = graph->ptrs[i];
        switch (curr_node->operation) {
            case ADD:
                curr_node->left->grad += curr_node->grad;
                curr_node->right->grad += curr_node->grad;
                break;
            case SUB:
                curr_node->left->grad += curr_node->grad;
                curr_node->right->grad -= curr_node->grad;
                break;
            case MUL:
                curr_node->left->grad += curr_node->grad * curr_node->right->value;
                curr_node->right->grad += curr_node->grad * curr_node->left->value;
                break;
            case RELU:
                curr_node->left->grad += (curr_node->left->value > 0 ? 1.0 : 0.0) * curr_node->grad;
                break;
            case CONSTANT:
                break;
        }
    }
}

void graph_forward(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        Node* curr_node = graph->ptrs[i];

        switch (curr_node->operation) {
            case ADD:
                curr_node->value = curr_node->left->value + curr_node->right->value;
                break;
            case SUB:
                curr_node->value = curr_node->left->value - curr_node->right->value;
                break;
            case MUL:
                curr_node->value = curr_node->left->value * curr_node->right->value;
                break;
            case RELU:
                curr_node->value = (curr_node->left->value > 0) ? curr_node->left->value : 0;
                break;
            case CONSTANT:
                break;
        }
    }
}

void graph_zero_grads(Graph* graph) {
    for (int i = 0; i < graph->count; i++) {
        graph->ptrs[i]->grad = 0.0;
    }
}

void graph_print(Graph* graph) {
    node_print(GRAPH_ROOT(graph), 0);
}

#endif