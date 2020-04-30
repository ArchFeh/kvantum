#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <assert.h>
#include <cmath>
#include <mpi.h>
#include <sys/time.h>

using namespace std;

typedef complex<double> complexd;

void OneQubitEvolution(complexd *in, complexd *out, complexd U[2][2], int nqubits, int q)
{
    int shift = nqubits - q;
    int pow2q = 1 << (shift);
    int N = 1 << nqubits;
    {
        for (int i = 0; i < N; i++)
        {
            int i0 = i & ~pow2q;
            int i1 = i | pow2q;
            int iq = (i & pow2q) >> shift;
            out[i] = U[iq][0] * in[i0] + U[iq][1] * in[i1];
        }
    }
}

complexd *generate(int n)
{
    long long unsigned qsize = 1LLU << n;
    complexd *V = new complexd[qsize];
    double module = 0;
    time_t time_seed;
    time_seed = time(NULL);
    {
        unsigned int seed = (unsigned)time_seed;
        for (long long unsigned i = 0; i < qsize; i++)
        {
            V[i].real(rand_r(&seed) ;
            V[i].imag(rand_r(&seed) ;
            module += abs(V[i] * V[i]);
        }
        for (long long unsigned j = 0; j < qsize; j++)
        {
            V[j] /= module;
        }
    }
    return V;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 3)
    {
        cout << "Usage:" << argv[0] << " n(number of qubits) k(index qubits)";
        return 1;
    }
    int n = atoi(argv[1]); // number of qubits
    int k = atoi(argv[2]); // index qubit
    if (n < k)
    {
        cout << "n need bigger than k";
        exit(0);
    }

    struct timeval tvs, tve;
    complexd *V = generate(n);
    unsigned long long index = 1LLU << n;
    complexd *W = new complexd[index];
    complexd H[2][2];
    H[0][0] = 1 / sqrt(2);
    H[0][1] = 1 / sqrt(2);
    H[1][0] = 1 / sqrt(2);
    H[1][1] = -1 / sqrt(2);
    gettimeofday(&tvs, NULL);
    OneQubitEvolution(V, W, H, n, k);
    gettimeofday(&tve, NULL);
    cout << "n=" << n << " "
         << "k=" << k << "\n";
    cout << "Time:" << tve.tv_sec - tvs.tv_sec + (tve.tv_usec - tvs.tv_usec) / 1000000.0 << "s" << endl;
    delete[] V;
    delete[] W;
}
