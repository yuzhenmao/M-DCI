#ifndef STACK_H
#define STACK_H

#include <stdbool.h>  // For bool, true, false
#include "btree_p.h"

// Define the maximum size of the stack
#define MAX_SIZE 100

// Define the structure for the stack
typedef struct Stack {
    btree_p_leaf_node** leaves;  // Array of leaf nodes
    int top;                         // Index of the top element
    int size;                        // Maximum size of the stack
} Stack;

// Function declarations

/**
 * Initializes the stack.
 * 
 * @param stack Pointer to the stack to initialize.
 */
void initStack(Stack* stack, int size);

/**
 * Checks if the stack is full.
 * 
 * @param stack Pointer to the stack.
 * @return true if the stack is full, false otherwise.
 */
bool isFull(Stack* stack);

/**
 * Checks if the stack is empty.
 * 
 * @param stack Pointer to the stack.
 * @return true if the stack is empty, false otherwise.
 */
bool isEmpty(Stack* stack);

/**
 * Pushes an element onto the stack.
 * 
 * @param stack Pointer to the stack.
 * @param value The value to push onto the stack.
 */
void push(Stack* stack, btree_p_leaf_node* leaf);

/**
 * Pops an element from the stack.
 * 
 * @param stack Pointer to the stack.
 * @return The value popped from the stack, or -1 if the stack is empty.
 */
btree_p_leaf_node* pop(Stack* stack);

/**
 * Returns the top element of the stack without removing it.
 * 
 * @param stack Pointer to the stack.
 * @return The value at the top of the stack, or -1 if the stack is empty.
 */
btree_p_leaf_node* peek(Stack* stack);

/**
 * Frees the dynamically allocated memory in the stack.
 * 
 * @param stack Pointer to the stack to free.
 */
void freeStack(Stack* stack);

#endif  // STACK_H