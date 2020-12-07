
#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "AztecOO.h"
#include <omp.h>

// external function
void  get_neighbours( const int i, const int nx, const int ny,
    int & left, int & right,
    int & lower, int & upper);
// =========== //
// main driver //
// =========== //
int main(int argc, char *argv[])
{
    using namespace std;
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
    
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int iam = 0, np = 1;
    
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Get_processor_name(processor_name, &namelen);

      //omp_set_num_threads(4);

    #pragma omp parallel default(shared) private(iam, np)
      {
        np = omp_get_num_threads();
        iam = omp_get_thread_num();
        printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
               iam, np, rank, numprocs, processor_name);
      }
    
  // number of nodes along the x- and y-axis
  int nx = 500;
  int ny = 600;
  int NumGlobalElements = nx * ny;
  // create a linear map
  Epetra_Map Map(NumGlobalElements,0,Comm);
  // local number of rows
  int NumMyElements = Map.NumMyElements();
  // get update list
  int * MyGlobalElements = new int [NumMyElements];
  Map.MyGlobalElements( MyGlobalElements );
  // Create a Epetra_Matrix with 5 nonzero per rows
  Epetra_CrsMatrix A(Copy,Map,5);
  // Add  rows one-at-a-time
  // Need some vectors to help
  double Values[4];
  int Indices[4];
  int left, right, lower, upper;
  double diag = 4.0;
  for( int i=0 ; i<NumMyElements; ++i ) {
    int NumEntries=0;
    get_neighbours(  MyGlobalElements[i], nx, ny,
        left, right, lower, upper);
    if( left != -1 ) {
      Indices[NumEntries] = left;
      Values[NumEntries] = -1.0;
      ++NumEntries;
    }
    if( right != -1 ) {
      Indices[NumEntries] = right;
      Values[NumEntries] = -1.0;
      ++NumEntries;
    }
    if( lower != -1 ) {
      Indices[NumEntries] = lower;
      Values[NumEntries] = -1.0;
      ++NumEntries;
    }
    if( upper != -1 ) {
      Indices[NumEntries] = upper;
      Values[NumEntries] = -1.0;
      ++NumEntries;
    }
    // put the off-diagonal entries
    A.InsertGlobalValues(MyGlobalElements[i], NumEntries, Values, Indices);
    // Put in the diagonal entry
    A.InsertGlobalValues(MyGlobalElements[i], 1, &diag, MyGlobalElements+i);
  }
  // Finish up
  A.FillComplete();
  // create x and b vectors
  Epetra_Vector x(Map);
  Epetra_Vector b(Map);
  b.PutScalar(1.0);
  // ==================== AZTECOO INTERFACE ======================
  // create linear problem
  Epetra_LinearProblem Problem(&A,&x,&b);
  // create AztecOO instance
  AztecOO Solver(Problem);
  Solver.SetAztecOption( AZ_precond, AZ_Jacobi );
  Solver.SetAztecOption( AZ_output, AZ_summary);
  Solver.Iterate(1000,1E-9);
  // ==================== END OF AZTECOO INTERFACE ================
  if( Comm.MyPID() == 0 ) {
    cout << "Solver performed " << Solver.NumIters()
      << "iterations.\n";
    cout << "Norm of the true residual = " << Solver.TrueResidual() << endl;
  }
#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return(EXIT_SUCCESS);
}
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
void  get_neighbours( const int i, const int nx, const int ny,
    int & left, int & right,
    int & lower, int & upper)
{
  int ix, iy;
  ix = i%nx;
  iy = (i - ix)/nx;
  if( ix == 0 )
    left = -1;
  else
    left = i-1;
  if( ix == nx-1 )
    right = -1;
  else
    right = i+1;
  if( iy == 0 )
    lower = -1;
  else
    lower = i-nx;
  if( iy == ny-1 )
    upper = -1;
  else
    upper = i+nx;
  return;
}

