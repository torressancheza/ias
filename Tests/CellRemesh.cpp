#include <iostream>
#include <fstream>
#include <random>
#include <mpi.h>

#include "ias_Cell.h"

int main(int argc, char **argv)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using namespace ias;
    using namespace Tensor;
    using namespace std;
    MPI_Init(&argc, &argv);

    RCP<Cell> cell = rcp(new Cell);
    
    cell->generateSphereFromPlatonicSolid(3, 1.0);
    cell->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
    cell->Update();
    
    
    // mapFields(cell, cell);

    cell->remesh(0.03);
    cell->Update();
    
    cell->saveVTK("remesh.vtu");

    MPI_Finalize();
}
