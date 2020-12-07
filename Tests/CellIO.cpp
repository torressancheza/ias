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
    
    tensor<double,2> nodePos = {{0.0,0.0,0.0},{1.0,0.0,0.0},{1.0,1.0,0.0},{2.0,1.0,0.0}};
    tensor<int,2> connec = {{0,1,2},{1,3,2}};
    
    cell->setNodePositions(nodePos);
    cell->setConnectivity(connec);
    cell->addNodeField("c");
    cell->addGlobField("P");
    cell->setBasisFunctionType(BasisFunctionType::Linear);
    
    try
    {
        cell->getNodeField("c");
    }
    catch (const runtime_error& error)
    {
        cout << error.what() << endl;
    }
    
    cell->Update();
    
    cell->getNodeField("c") = cell->getNodeField("x");
    
    cell->saveVTK("cell.vtu");
    
    cell = rcp(new Cell);
    cell->loadVTK("cell.vtu");
    cell->addGlobField("viscosity");
    cell->setBasisFunctionType(BasisFunctionType::Linear);
    cell->Update();
    
    cout << cell->getNodeFields(1) << endl;
    cell->getGlobFields()(0) = 1;
    cout << cell->getGlobFields() << endl;
    cell->getNodeField(0) *= 2;
    cout << cell->getNodeField("x") << endl;



    for(auto g: cell->getNodeFieldNames())
        cout << g << " ";
    cout << endl;
    
    for(auto g: cell->getGlobFieldNames())
        cout << g << " ";
    cout << endl;
}
