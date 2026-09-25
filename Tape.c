#include <stdlib.h>

#include "Tape.h"

Tape* tape_alloc(size_t init_capacity) {    
    Tape* result = malloc(sizeof(Tape));
    result->capacity = init_capacity;
    result->ptrs = malloc(sizeof(Node*)*init_capacity);
    result->count = 0;
    return result;
}

void tape_free(Tape* tape) {
    for (int i = 0; i < tape->count; i++) {
        free(tape->ptrs[i]);
    }
    free(tape->ptrs);
    free(tape);
}

void tape_add(Tape* tape, Node* node) {
    if (tape->count == tape->capacity) {
        tape->capacity = tape->capacity + (tape->capacity >> 1); 
        tape->ptrs = realloc(tape->ptrs, sizeof(Node*) * tape->capacity);
    }
    tape->ptrs[tape->count++] = node;
}