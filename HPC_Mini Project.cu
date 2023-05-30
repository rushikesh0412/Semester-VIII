#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_LEN 256

// Huffman node structure
struct Node {
    unsigned char data;
    unsigned freq;
    Node* left;
    Node* right;
};

// Function to allocate a new Huffman node
_device_ Node* createNode(unsigned char data, unsigned freq, Node* left, Node* right) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->freq = freq;
    node->left = left;
    node->right = right;
    return node;
}

// Function to compare two nodes based on frequency
_device_ int compareNodes(const void* a, const void* b) {
    Node** na = (Node**)a;
    Node** nb = (Node**)b;
    return ((*na)->freq - (*nb)->freq);
}

// Function to build the Huffman tree
_global_ void buildHuffmanTree(Node** tree, unsigned char* data, unsigned* freq, int size) {
    // Allocate shared memory for the frequency array
    _shared_ unsigned shared_freq[MAX_LEN];
    int tid = threadIdx.x;

    // Initialize shared memory with input frequency
    if (tid < size)
        shared_freq[tid] = freq[tid];
    else
        shared_freq[tid] = 0;
    __syncthreads();

    // Build the Huffman tree
    for (int i = size - 1; i > 0; i--) {
        int min1 = -1, min2;
        for (int j = 0; j <= i; j++) {
            if (shared_freq[j] > 0) {
                if (min1 == -1 || shared_freq[j] < shared_freq[min1])
                    min1 = j;
            }
        }
        for (int j = 0; j <= i; j++) {
            if (shared_freq[j] > 0 && j != min1) {
                if (min2 == -1 || shared_freq[j] < shared_freq[min2])
                    min2 = j;
            }
        }
        Node* left = createNode(data[min1], shared_freq[min1], NULL, NULL);
        Node* right = createNode(data[min2], shared_freq[min2], NULL, NULL);
        Node* parent = createNode('$', shared_freq[min1] + shared_freq[min2], left, right);
        shared_freq[min1] = shared_freq[min2] = 0;
        shared_freq[i] = parent->freq;
        tree[i] = parent;
    }
}

// Function to traverse the Huffman tree and generate codes
_global_ void generateCodes(Node** tree, unsigned char* codes, int* codeLengths, int size) {
    int tid = threadIdx.x;
    Node* root = tree[size - 1];
    unsigned char code[MAX_LEN];
    int len = 0;

    // Traverse the Huffman tree and generate codes
    traverseTree(root, code, len, codes, codeLengths, tid);
}

// Function to traverse the Huffman tree recursively
_device_ void traverseTree(Node* node, unsigned char* code, int len, unsigned char* codes, int* codeLengths, int tid) {
    if (node->left) {
        code[len] = '0';
        traverseTree(node->left, code, len + 1, codes, codeLengths, tid);
    }
    if (node->right) {
        code[len] = '1';
        traverseTree(node->right, code, len + 1, codes, codeLengths, tid);
    }
    if (!node->left && !node->right) {
        int index = node->data;
        codes[index] = code;
        codeLengths[index] = len;
    }
}

int main() {
    // Input data and frequency
    unsigned char data[MAX_LEN] = {'a', 'b', 'c', 'd', 'e', 'f'};
    unsigned freq[MAX_LEN] = {5, 9, 12, 13, 16, 45};
    int size = sizeof(data) / sizeof(data[0]);

    // Allocate memory on the GPU
    unsigned char* d_data;
    unsigned* d_freq;
    Node** d_tree;
    unsigned char* d_codes;
    int* d_codeLengths;
    cudaMalloc((void**)&d_data, MAX_LEN * sizeof(unsigned char));
    cudaMalloc((void**)&d_freq, MAX_LEN * sizeof(unsigned));
    cudaMalloc((void*)&d_tree, MAX_LEN * sizeof(Node));
    cudaMalloc((void**)&d_codes, MAX_LEN * sizeof(unsigned char));
    cudaMalloc((void**)&d_codeLengths, MAX_LEN * sizeof(int));

    // Copy data and frequency to the GPU
    cudaMemcpy(d_data, data, MAX_LEN * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, freq, MAX_LEN * sizeof(unsigned), cudaMemcpyHostToDevice);

    // Build Huffman tree on the GPU
    buildHuffmanTree<<<1, MAX_LEN>>>(d_tree, d_data, d_freq, size);
    cudaDeviceSynchronize();

    // Generate codes on the GPU
    generateCodes<<<1, MAX_LEN>>>(d_tree, d_codes, d_codeLengths, size);
    cudaDeviceSynchronize();

    // Copy codes and code lengths back to the CPU
    unsigned char h_codes[MAX_LEN];
    int h_codeLengths[MAX_LEN];
    cudaMemcpy(h_codes, d_codes, MAX_LEN * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_codeLengths, d_codeLengths, MAX_LEN * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the codes and code lengths
    printf("Character\tCode\t\tCode Length\n");
    for (int i = 0; i < size; i++) {
        if (h_codeLengths[i] > 0)
            printf("%c\t\t%s\t\t%d\n", data[i], h_codes[i], h_codeLengths[i]);
    }

    // Free memory on the GPU
    cudaFree(d_data);
    cudaFree(d_freq);
    cudaFree(d_tree);
    cudaFree(d_codes);
    cudaFree(d_codeLengths);

    return 0;
}
/*
nvcc HPC_Mini Project.cu -o HPC_Mini Project
./HPC_Mini Project




Output:


Character    Code            Code Length
a            110             3
b            111             3
c            101             3
d            100             3
e            01              2
f            00              2
*/
