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
    
    RCP<Cell> cell = rcp(new Cell);
    
    cell->generateSphereFromOctahedron(3, 1.0);
    cell->setBasisFunctionType(BasisFunctionType::Linear);
    cell->Update();
    
    
    
    cell->remesh(0.1);
    cell->Update();
    
    cell->saveVTK("remesh.vtu");
}
