#include <iostream>
#include <fstream>
#include <random>

#include "ias_Cell.h"

int main()
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using namespace ias;
    using namespace Tensor;
    using namespace std;
    
    RCP<Cell> cell1 = rcp(new Cell);
    
    cell1->generateSphereFromOctahedron(3, 1.0);
    cell1->setBasisFunctionType(BasisFunctionType::Linear);
    cell1->addNodeField("c");
    cell1->addCellField("P");
    cell1->Update();
    cell1->saveVTK("cell0.vtu");

    RCP<Cell> cell2 = cell1->cellDivision(0.05);
    cell1->getCellField("cellId") = 1;
    cell2->getCellField("cellId") = 2;

    cell1->saveVTK("cell1.vtu");
    cell2->saveVTK("cell2.vtu");
}
