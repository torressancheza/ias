#include <iostream>
#include <fstream>
#include <random>

#include "ias_Cell.h"
#include "ias_Tissue.h"
#include "ias_TissueGen.h"

int main(int argc, char **argv)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using namespace ias;
    using namespace Tensor;
    using namespace std;
    
//    RCP<Cell> cell1 = rcp(new Cell);
//
//    cell1->generateSphereFromOctahedron(3, 1.0);
//    cell1->setBasisFunctionType(BasisFunctionType::Linear);
//    cell1->addNodeField("c");
//    cell1->addCellField("P");
//    cell1->Update();
//    cell1->saveVTK("cell0.vtu");
//
//    RCP<Cell> cell2 = cell1->cellDivision(0.05);
//    cell1->getCellField("cellId") = 1;
//    cell2->getCellField("cellId") = 2;
//
//    cell1->saveVTK("cell1.vtu");
//    cell2->saveVTK("cell2.vtu");
    
    MPI_Init(&argc, &argv);

    
    RCP<TissueGen> tissueGen = rcp( new TissueGen);
    tissueGen->setBasisFunctionType(BasisFunctionType::Linear);
    tissueGen->addNodeFields({"vx","vy","vz"});
    tissueGen->addCellFields({"P","Paux"});
    RCP<Tissue> tissue = tissueGen->genRegularGridSpheres(2, 1, 1, 2.1, 2.1, 2.1, 1, 3);

    tissue->saveVTK("Division", "0");

    tissue->cellDivision({0,1}, 0.05, 0.05);
    
    tissue->saveVTK("Division", "1");
    
    MPI_Finalize();
}
