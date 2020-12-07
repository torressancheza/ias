#include <iostream>
#include <fstream>
#include <random>

#include "ias_Tissue.h"
#include "ias_TissueGen.h"

int main()
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using namespace ias;
    using namespace Tensor;
    using namespace std;
    
    MPI_Init(nullptr, nullptr);
    
    RCP<TissueGen> tissueGen = rcp( new TissueGen);
    tissueGen->setBasisFunctionType(BasisFunctionType::Linear);
    tissueGen->addNodeField("c");
    
    RCP<Tissue> tissue = tissueGen->genRegularGridSpheres(2, 2, 2, 2.1, 2.1, 2.1, 1.0, 3);
    
    tissue->saveVTK("Cell", "", 0);
    
    MPI_Finalize();
}
