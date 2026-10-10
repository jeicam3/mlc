#pragma once

typedef enum {
    CONSTANT,
    ADD,
    SUB,
    MUL,
    RELU
} Op;

const char* op_to_str(Op op);

#ifdef OP_IMPLEMENTATION

const char* op_to_str(Op op) {
    switch (op) {
            case ADD:
                return "ADD";
                break;
            case SUB:
                return "SUB";
                break;
            case MUL:
                return "MUL";
                break;
            case RELU:
                return "RELU";
                break;
            case CONSTANT:
                return "CONSTANT";
                break;
        }
}

#endif