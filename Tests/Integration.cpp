#include <iostream>
#include <fstream>
#include <random>
#include <omp.h>
#include <unistd.h>
#include <chrono>  // for high_resolution_clock

#include "ias_Tissue.h"
#include "ias_TissueGen.h"
#include "ias_Integration.h"

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        //[1] INPUT
        int eNN    = fill->eNN;
        
        tensor<double,2> nborFields   = fill->nborFields;

        tensor<double,1>&  globFields = fill->cellFields;

        double pressure = globFields(0);

        tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
        tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);
        tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);


        //[2.2] Geometry in current configuration
        tensor<double,1>      x = bfs * (nborFields(all,range(0,2)));
        tensor<double,2>     Dx = Dbfs.T() * (nborFields(all,range(0,2)));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        double               xn = x * normal;
        tensor<double,2> metric = Dx * Dx.T();
        tensor<double,2> imetric = metric.inv();

        tensor<double,1> v = bfs * nborFields(all,range(3,5));
        
        //[2.4] First-order derivatives of geometric quantities wrt nodal positions
        tensor<double,4> dDx        = outer(Dbfs,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross     = dDx(all,all,1,all)*antisym3D*Dx(0,all) - dDx(all,all,0,all)*antisym3D*Dx(1,all);
        tensor<double,2> djac       = 1./jac * dcross * cross;
        tensor<double,4> dmetric    = dDx * Dx.T();
                         dmetric   += dmetric.transpose({0,1,3,2});
        tensor<double,4> dmetric_CC = product(product(dmetric,imetric,{{2,0}}),imetric,{{2,0}});
        tensor<double,3> dnormal    = dcross/jac - outer(djac/(jac*jac),cross);
        
        //[2.5] Second-order derivatives of geometric quantities wrt nodal positions
        tensor<double,5> ddcross    = (dDx(all,all,1,all)*antisym3D*dDx(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                         ddcross  += ddcross.transpose({2,3,0,1,4});
        tensor<double,4> ddjac      = 1./jac * (ddcross * cross + product(dcross,dcross,{{2,2}}) - outer(djac,djac));
        tensor<double,5> ddnormal   = ddcross/jac - outer(dcross,djac/(jac*jac)).transpose({0,1,3,4,2}) - outer(djac/(jac*jac),dcross) - outer(ddjac/(jac*jac),cross) + 2.0/(jac*jac*jac) * outer(outer(djac,djac),cross);
        tensor<double,6> ddmetric   = 2.0 * product(dDx,dDx,{{3,3}}).transpose({0,1,3,4,2,5});
        
        
        //[3] OUTPUT
        tensor<double,2>& rhs_n = fill->vec_n;
        tensor<double,1>& rhs_g = fill->vec_c;
        tensor<double,4>& A_nn  = fill->mat_nn;
        tensor<double,3>& A_ng  = fill->mat_nc;
        tensor<double,3>& A_gn  = fill->mat_cn;
            
        // [3.1] Friction dissipation
        rhs_n(all,range(0,2)) += fill->w_sample * jac * outer(bfs,v);
        A_nn(all,range(0,2),all,range(0,2))  += fill->w_sample * jac * outer(bfs,outer(bfs,Identity(3))).transpose({0,2,1,3});
        
        rhs_n(all,range(0,2)) += fill->w_sample * jac * outer(bfs,normal);
        A_nn(all,range(0,2),all,range(0,2))  += fill->w_sample * jac * outer(bfs,outer(bfs,Identity(3))).transpose({0,2,1,3});
}

void interaction(Teuchos::RCP<ias::DoubleIntegralStr> inter)
{
}

int main(int argc, char *argv[])
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using namespace ias;
    using namespace Tensor;
    using namespace std;
    
    MPI_Init(&argc, &argv);
    
    RCP<TissueGen> tissueGen = rcp( new TissueGen);
    tissueGen->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
    tissueGen->addNodeField("c");
    tissueGen->addCellField("P");

    RCP<Tissue> tissue = tissueGen->genRegularGridSpheres(2, 1, 1, 2.1, 2.1, 2.1, 1.0, 5);
    tissue->calculateCellCellAdjacency(0.5);
    tissue->updateGhosts();
//    tissue->calculateInteractingElements(0.5);
    tissue->saveVTK("Cell", "");
    
    for(auto cell: tissue->getLocalCells())
        cout << cell->getCellField("cellId") << endl;


    RCP<Integration> integration = rcp(new Integration);
    integration->setTissue(tissue);
    integration->setNodeDOFs({"x","y","z"});
    integration->setCellDOFs({"P"});
    integration->setSingleIntegrand(internal);
    integration->setDoubleIntegrand(interaction);
    integration->setNumberOfIntegrationPointsSingleIntegral(1);
    integration->setNumberOfIntegrationPointsDoubleIntegral(1);
    integration->Update();

    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    
    integration->computeSingleIntegral();
//    integration->computeDoubleIntegral();

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    
    cout << elapsed.count() << endl;

    MPI_Finalize();
}
