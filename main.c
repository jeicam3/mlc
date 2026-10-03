#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include "Node.h"

double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

int main() {
    Graph* graph = graph_alloc(100);

    //training data - xor gate
    int training_data[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0
    };

    //input layer
    Node* x1 = node_constant(0, graph);
    Node* x2 = node_constant(0, graph);

    //hidden layer
    Node* w11 = node_constant(rand_weight(), graph);
    Node* w12 = node_constant(rand_weight(), graph);
    Node* b1 = node_constant(rand_weight(), graph);

    Node* w21 = node_constant(rand_weight(), graph);
    Node* w22 = node_constant(rand_weight(), graph);
    Node* b2 = node_constant(rand_weight(), graph);

    Node* n1 = node_relu(node_add(node_add(node_mul(w11, x1, graph), node_mul(w12, x2, graph), graph), b1, graph), graph);
    Node* n2 = node_relu(node_add(node_add(node_mul(w21, x1, graph), node_mul(w22, x2, graph), graph), b2, graph), graph);

    //output layer
    Node* w31 = node_constant(rand_weight(), graph);
    Node* w32 = node_constant(rand_weight(), graph);
    Node* b3 = node_constant(rand_weight(), graph);

    Node* output = node_add(node_add(node_mul(w31, n1, graph), node_mul(w32, n2, graph), graph), b3, graph);

    double learning_rate = 0.01;
    Node* y_true = node_constant(0, graph);
    Node* diff = node_sub(output, y_true, graph);
    Node* loss = node_mul(diff, diff, graph);

    double loss_sum;

    for (int epoch = 0; epoch < 3000; epoch++) {
        loss_sum = 0;
        for (int i = 0; i < 4; i++) {
            graph_zero_grads(graph);

            x1->value = training_data[i*3 + 0];
            x2->value = training_data[i*3 + 1];
            y_true->value = training_data[i*3 + 2];

            graph_forward(graph);

            graph_backward(graph);

            w11->value = w11->value - learning_rate * w11->grad;
            w12->value = w12->value - learning_rate * w12->grad;
            b1->value = b1->value - learning_rate * b1->grad;

            w21->value = w21->value - learning_rate * w21->grad;
            w22->value = w22->value - learning_rate * w22->grad;
            b2->value = b2->value - learning_rate * b2->grad;

            w31->value = w31->value - learning_rate * w31->grad;
            w32->value = w32->value - learning_rate * w32->grad;
            b3->value = b3->value - learning_rate * b3->grad;

            loss_sum += loss->value;
        }

        if (epoch % 10 == 0) {
            printf("Epoch: %d, Loss: %f\n", epoch, loss_sum);
        }
    }

    printf("\nTraining finished\n");
    printf("Final model's loss: %f\n", loss_sum);
    printf("Final model's predictions: \n");
    for (int i = 0; i < 4; i++) {
        x1->value = training_data[i*3 + 0];
        x2->value = training_data[i*3 + 1];
        y_true->value = training_data[i*3 + 2];

        graph_forward(graph);

        printf("%.1f ^ %.1f = %f (actual: %.1f)\n", x1->value, x2->value, output->value, y_true->value);
    }

    graph_print(graph);
    graph_free(graph);
    return 0;
}