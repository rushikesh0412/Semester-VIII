// implementation of Min, Max, Sum and Average operations using Parallel Reduction in 
//C++ using OpenMP:
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    int n = 100000;
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    int min_val = arr[0];
    int max_val = arr[0];
    int sum_val = 0;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
        sum_val += arr[i];
    }

    double avg_val = (double)sum_val / n;

    cout << "Minimum value: " << min_val << endl;
    cout << "Maximum value: " << max_val << endl;
    cout << "Sum value: " << sum_val << endl;
    cout << "Average value: " << avg_val << endl;

    return 0;
}


/*
g++ -fopenmp HPC_Assg_03.cpp -o HPC_Assg_03
./HPC_Assg_03

Minimum value: 0
Maximum value: 99
Sum value: 5029653
Average value: 50.2965
*/
