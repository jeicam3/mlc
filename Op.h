#pragma once

typedef enum {
    CONSTANT,
    ADD,
    SUB,
    MUL,
    RELU
} Op;

const char* op_to_str(Op op);