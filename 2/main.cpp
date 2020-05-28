#include <iostream>
#include <math.h>
#include <vector>
#include <complex>
#include <string.h>
#include <ctime>
#include <stdlib.h>
#include "mpi.h"

typedef std::complex<float> complexd;
typedef unsigned int uint;

using std::cout;
using std::endl;
using std::vector;

int world_rank = 0, world_size;

void OneQubitEvolution(vector<complexd> &in, vector<complexd> &out, complexd U[2][2], uint nqubits, uint q)
{
    uint shift = nqubits - q;
    uint procNum = ((1 << (shift)) / in.size());

    if (world_rank != (procNum ^ world_rank))
    {
        MPI_Sendrecv(
            in.data(), in.size(), MPI_COMPLEX, procNum ^ world_rank, 0, out.data(), out.size(), MPI_COMPLEX, procNum ^ world_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (world_rank > (procNum ^ world_rank))
        {

            for (int i = 0; i < in.size(); ++i)
                out[i] = U[1][0] * in[i] + U[1][1] * out[i];
        }
        else
        {
            {
                for (int i = 0; i < in.size(); ++i)
                    out[i] = U[0][0] * in[i] + U[0][1] * out[i];
            }
        }
    }
    else
    {

        for (int i = 0; i < in.size(); ++i)
        {
            unsigned i0 = i & ~procNum;
            unsigned i1 = i | procNum;
            unsigned iq = (i & procNum) >> shift;
            out[i] = U[iq][0] * in[i0] + U[iq][1] * in[i1];
        }
    }
}

vector<complexd> generate(int n)
{
    long long int qsize = pow(2, n) / world_size;

    vector<complexd> V(qsize);
    double module = 0;
    time_t time_seed;
    time_seed = time(NULL);
    {
        unsigned int seed = (unsigned)time_seed;
        for (long long unsigned i = 0; i < qsize; i++)
        {
            float real = rand_r(&seed);
            float imag = rand_r(&seed);
            V[i] = complexd(real, imag);
            module += abs(V[i] * V[i]);
            module = sqrt(module);
            V[i] /= module;
        }
    }
    return V;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        cout << "Usage:mpic++ -n N" << argv[0] << " n(number of qubits) k(index qubits)";
        return -1;
    }
    uint n = atoi(argv[1]);
    if (n < 1 || n > 100)
        return -1;

    uint k = atoi(argv[2]);
    if (k > n)
    {
        cout << "n need bigger than k";
        return -1;
    }

    double startTime = MPI_Wtime();

    long long int vectorLength = pow(2, n) / world_size;
    vector<complexd> oldState = generate(n);
    vector<complexd> newState(vectorLength);
    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);

    OneQubitEvolution(oldState, newState, U, n, k);

    double workTime = MPI_Wtime() - startTime;

    double maxtime;
    MPI_Reduce(&workTime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
        cout << "Time:" << maxtime << "s" << endl;

    MPI_Finalize();
    return 0;
}
