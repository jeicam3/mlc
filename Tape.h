typedef struct Node Node;

typedef struct Tape {
    Node** ptrs;
    int capacity;
    int count;
}Tape;

Tape* tape_alloc(size_t init_capacity);
void tape_free(Tape* tape);
void tape_add(Tape* tape, Node* node);