#define TENSORC_IMPLEMENTATION
#include "tensorc.h"
#define MLC_IMPLEMENTATION
#include "mlc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand_weight();
    }
}

void update_weights(Node* w, float lr) {
    for (int i = 0; i < w->value->size; i++) {
        w->value->data[i] -= lr * w->grad->data[i];
    }
}

int main() {
    srand((unsigned int)time(NULL));
    Graph* graph = graph_alloc(100);

    // 1. ZBIÓR DANYCH (Batch 4 próbek)
    int x_shape[] = {4, 2};
    float x_data[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    Node* X = node_constant(x_data, x_shape, 2, graph);

    int y_shape[] = {4, 1};
    float y_data[] = {0.0f, 1.0f, 1.0f, 0.0f};
    Node* Y_TRUE = node_constant(y_data, y_shape, 2, graph);

    // 2. WARSTWA UKRYTA (2 wejścia -> 2 neurony)
    int w1_shape[] = {2, 2};
    float w1_data[4]; 
    init_random(w1_data, 4);
    Node* W1 = node_constant(w1_data, w1_shape, 2, graph);
    
    // Bias [4, 2]: Tymczasowy hack - powielamy 2 wartości biasu 4 razy, by pasowały do batcha
    int b1_shape[] = {4, 2}; 
    float b1_data[8]; 
    init_random(b1_data, 8);
    Node* B1 = node_constant(b1_data, b1_shape, 2, graph);

    // 3. WARSTWA WYJŚCIOWA (2 neurony ukryte -> 1 wyjście)
    int w2_shape[] = {2, 1};
    float w2_data[2]; 
    init_random(w2_data, 2);
    Node* W2 = node_constant(w2_data, w2_shape, 2, graph);

    // Bias [4, 1]: Powielony 1 bias dla 4 elementów batcha
    int b2_shape[] = {4, 1};
    float b2_data[4]; 
    init_random(b2_data, 4);
    Node* B2 = node_constant(b2_data, b2_shape, 2, graph);

    // 4. ARCHITEKTURA GRAFU
    Node* hidden = node_relu(node_add(node_matmul(X, W1, graph), B1, graph), graph);
    Node* pred   = node_add(node_matmul(hidden, W2, graph), B2, graph);
    
    Node* diff   = node_sub(pred, Y_TRUE, graph);
    Node* loss   = node_mul(diff, diff, graph); // MSE Loss

    // 5. PĘTLA UCZĄCA
    // Sieć 2-2-1 jest mniejsza, więc może wymagać dłuższego treningu lub większego LR.
    float learning_rate = 0.05f; 

    for (int epoch = 0; epoch < 10000; epoch++) {
        graph_zero_grads(graph);
        graph_forward(graph);
        graph_backward(graph);

        // Aktualizacja wag
        update_weights(W1, learning_rate);
        update_weights(B1, learning_rate);
        update_weights(W2, learning_rate);
        update_weights(B2, learning_rate);

        if (epoch % 1000 == 0) {
            float total_loss = 0;
            for(int i = 0; i < 4; i++) total_loss += loss->value->data[i];
            printf("Epoch: %d, Loss: %f\n", epoch, total_loss / 4.0f);
        }
    }

    printf("\nTraining finished!\n");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ^ %.1f = %.4f (Oczekiwane: %.1f)\n", 
            X->value->data[i*2], X->value->data[i*2 + 1], 
            pred->value->data[i], Y_TRUE->value->data[i]);
    }

    graph_free(graph);
}