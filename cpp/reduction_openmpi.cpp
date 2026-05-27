/*
 * Minimal Open MPI Array Summation Example
 *
 * Compilation:
 *   mpic++ reduction_openmpi.cpp -o reduction_openmpi.o
 *
 * Running:
 *   mpirun -np 4 ./reduction_openmpi.o
 *
 * This program distributes array summation across multiple MPI processes.
 * Each process computes a local sum, then MPI_Reduce combines all partial
 * sums on process 0.
 */

#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10;
    std::vector<int> local_array(n);

    // Each process fills its local array with values
    for (int i = 0; i < n; i++) {
        local_array[i] = rank * 100 + i;
    }

    // Local sum
    int local_sum = 0;
    for (int i = 0; i < n; i++) {
        local_sum += local_array[i];
    }

    // Global sum using MPI reduction
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        std::cout << "Global sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
