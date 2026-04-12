#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "stack.h"

// Function to initialize the stack
void initStack(Stack* stack, int size) {
    stack->leaves = (btree_p_leaf_node**)malloc(size * sizeof(btree_p_leaf_node*));
    stack->top = -1;
    stack->size = size;
}

// Check if the stack is full
bool isFull(Stack* stack) {
    return stack->top == stack->size - 1;
}

// Check if the stack is empty
bool isEmpty(Stack* stack) {
    return stack->top == -1;
}

// Pushes an item onto the stack
void push(Stack* stack, btree_p_leaf_node* leaf) {
    if (isFull(stack)) {
        stack->size *= 2;
        stack->leaves = (btree_p_leaf_node**)realloc(stack->leaves, stack->size * sizeof(btree_p_leaf_node*));
    }
    stack->leaves[++stack->top] = leaf;
}

// Pops an item from the stack
btree_p_leaf_node* pop(Stack* stack) {
    if (isEmpty(stack)) {
        
        return NULL;
    }
    return stack->leaves[stack->top--];
}

// Returns the top item of the stack without removing it
btree_p_leaf_node* peek(Stack* stack) {
    if (isEmpty(stack)) {
        printf("Stack is empty! Nothing to peek.\n");
        return NULL;
    }
    return stack->leaves[stack->top];
}

// Frees the dynamically allocated memory in the stack
void freeStack(Stack* stack) {
    // for (int i = 0; i <= stack->top; i++) {
    //     if (stack->leaves[i].addr != NULL) {
    //         free(stack->leaves[i].addr);  // Free the dynamically allocated memory
    //         stack->leaves[i].addr = NULL;  // Avoid dangling pointers
    //     }
    // }
    free(stack->leaves);  // Free the array of stack items
    stack->leaves = NULL;  // Avoid dangling pointers
    stack->top = -1;  // Reset the stack
    stack->size = 0;  // Reset the size
}
