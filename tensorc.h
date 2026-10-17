#pragma once
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Tensor;

Tensor* tensor_alloc(int* shape, int ndim);
void tensor_free(Tensor* t);

void tensor_add(Tensor* out, Tensor* a, Tensor* b);
void tensor_sub(Tensor* out, Tensor* a, Tensor* b);
void tensor_mul(Tensor* out, Tensor* a, Tensor* b);
void tensor_matmul(Tensor* out, Tensor* a, Tensor* b);
void tensor_relu(Tensor* out, Tensor* x);

#ifdef TENSORC_IMPLEMENTATION

Tensor* tensor_alloc(int* shape, int ndim) {
    Tensor* t = malloc(sizeof(Tensor));
    t->ndim = ndim;
    
    t->shape = malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    
    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->size *= shape[i];
        t->strides[i] = (i == ndim - 1) ? 1 : t->strides[i+1] * shape[i+1];
    }

    int padded_size = (t->size + 7) & ~7;
    t->data = aligned_alloc(32, padded_size * sizeof(float));
    memset(t->data, 0, padded_size * sizeof(float));

    return t;
}

void tensor_free(Tensor* t) {
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t);
}

void tensor_add(Tensor* out, Tensor* a, Tensor* b) {
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_sub(Tensor* out, Tensor* a, Tensor* b) {
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void tensor_mul(Tensor* out, Tensor* a, Tensor* b) {
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_matmul(Tensor* out, Tensor* a, Tensor* b) {
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    memset(out->data, 0, out->size * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
}

void tensor_relu(Tensor* out, Tensor* x) {
    for (int i = 0; i < out->size; i++) {
        out->data[i] = (x->data[i] > 0) ? x->data[i] : 0.0f;
    }
}

#endif