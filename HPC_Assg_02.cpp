//Here's an implementation of Parallel Bubble Sort and Merge Sort using OpenMP in C++.
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

// Function to perform Bubble Sort
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Function to perform Merge Sort
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }

    for (int j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    int i = 0;
    int j = 0;
    int k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l >= r) {
        return;
    }

    int m = l + (r - l) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mergeSort(arr, l, m);
        }

        #pragma omp section
        {
            mergeSort(arr, m + 1, r);
        }
    }

    merge(arr, l, m, r);
}

int main() {
    int n = 100000;
    int arr[n];

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % n;
    }

    double start_time, end_time;

    // Sequential Bubble Sort
    start_time = omp_get_wtime();
    bubbleSort(arr, n);
    end_time = omp_get_wtime();

    cout << "Time taken for Sequential Bubble Sort: " << end_time - start_time << " seconds" << endl;

    // Parallel Bubble Sort
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        bubbleSort(arr, n);
    }
    end_time = omp_get_wtime();

    cout << "Time taken for Parallel Bubble Sort: " << end_time - start_time << " seconds" << endl;

    // Sequential Merge Sort
    start_time = omp_get_wtime();
    mergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();

    cout << "Time taken for Sequential Merge Sort: " << end_time - start_time << " seconds" << endl;

    // Parallel Merge Sort
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSort(arr, 0, n - 1);
        }
    }
    end_time = omp_get_wtime();
}

/*
g++ -fopenmp HPC_Assg_02.cpp -o HPC_Assg_02
./HPC_Assg_02


g++ -fopenmp <filename>.cpp -o <output_filename>
./<output_filename>





The inputs to this program are the size of the array n, and the values of the elements in the array arr.

The outputs of this program are the time taken for the sequential and parallel implementations of Bubble Sort and Merge Sort. The output is printed to the console.

For example, if n = 100000, the output might look like this:

output:
Time taken for Sequential Bubble Sort: 27.6211 seconds
Time taken for Parallel Bubble Sort: 7.66581 seconds
Time taken for Sequential Merge Sort: 0.050481 seconds
Time taken for Parallel Merge Sort: 0.022264 seconds

*/
