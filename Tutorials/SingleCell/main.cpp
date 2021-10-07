//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

#include <AztecOO.h>

#include "aux.h"


int main(int argc, char **argv)
{
    using namespace Tensor;
    using namespace std;
    using namespace ias;
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    MPI_Init(&argc, &argv);
    
    //---------------------------------------------------------------------------
    // [0] Input parameters
    //---------------------------------------------------------------------------
    int nSubdiv{5};
    
    RCP<TissueGen> tissueGen = rcp( new TissueGen);
    tissueGen->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
    tissueGen->addNodeFields({"vx","vy","vz","tension","phi","v_n","vtx","vty","vtz"});
    tissueGen->addTissFields({"deltat", "error"});
    RCP<Tissue> tissue = tissueGen->genRegularGridSpheres(1, 1, 1, 0.0, 0.0, 0.0, 1, nSubdiv);
    tissue->calculateCellCellAdjacency(0.1);

    double deltat = 1.E-4;
    tissue->getTissField("deltat") = deltat;
    
    RCP<Integration> leastSquaresIntegration = rcp(new Integration);
    leastSquaresIntegration->setTissue(tissue);
    leastSquaresIntegration->setNodeDOFs({"x","y","z","tension","phi","v_n","vtx","vty","vtz"});
    leastSquaresIntegration->setSingleIntegrand(leastSquares);
    leastSquaresIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    leastSquaresIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    leastSquaresIntegration->Update();

    RCP<solvers::TrilinosAztecOO> leastSquaresLinearSolver = rcp(new solvers::TrilinosAztecOO);
    leastSquaresLinearSolver->setIntegration(leastSquaresIntegration);
    leastSquaresLinearSolver->addAztecOOParameter("solver","bicgstab");
    leastSquaresLinearSolver->addAztecOOParameter("precond","dom_decomp");
    leastSquaresLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
    leastSquaresLinearSolver->addAztecOOParameter("output","none");
    leastSquaresLinearSolver->setMaximumNumberOfIterations(5000);
    leastSquaresLinearSolver->setResidueTolerance(1.E-10);
    leastSquaresLinearSolver->Update();

    leastSquaresIntegration->fillVectorWithScalar(0.0);
    leastSquaresIntegration->fillSolutionWithScalar(0.0);
    leastSquaresIntegration->fillMatrixWithScalar(0.0);
    leastSquaresIntegration->InitialiseTissIntegralFields(0.0);
    leastSquaresIntegration->InitialiseCellIntegralFields(0.0);
    leastSquaresIntegration->computeSingleIntegral();
    leastSquaresIntegration->assemble();
    leastSquaresLinearSolver->solve();
    leastSquaresIntegration->setSolToDOFs();

    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx","vy","vz"});
    physicsIntegration->setTissIntegralFields({"error"});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    physicsIntegration->Update();

    RCP<solvers::TrilinosAztecOO> physicsLinearSolver = rcp(new solvers::TrilinosAztecOO);
    physicsLinearSolver->setIntegration(physicsIntegration);
    physicsLinearSolver->addAztecOOParameter("solver","gmres");
    physicsLinearSolver->addAztecOOParameter("precond","dom_decomp");
    physicsLinearSolver->addAztecOOParameter("subdomain_solve","ilut"); 
    physicsLinearSolver->addAztecOOParameter("tol","1.E-10"); 
    physicsLinearSolver->addAztecOOParameter("output","none");
    physicsLinearSolver->setMaximumNumberOfIterations(5000);
    physicsLinearSolver->setResidueTolerance(1.E-10);
    physicsLinearSolver->Update();
    
    RCP<solvers::NewtonRaphson> physicsNewtonRaphson = rcp(new solvers::NewtonRaphson);
    physicsNewtonRaphson->setLinearSolver(physicsLinearSolver);
    physicsNewtonRaphson->setSolutionTolerance(1.E-10);
    physicsNewtonRaphson->setResidueTolerance(1.E-10);
    physicsNewtonRaphson->setMaximumNumberOfIterations(6);
    physicsNewtonRaphson->setVerbosity(true);
    physicsNewtonRaphson->Update();

    
    physicsNewtonRaphson->solve();
    bool conv = physicsNewtonRaphson->getConvergence();
    
    for(auto cell: tissue->getLocalCells())
    {
        cell->getNodeField("vx") /= deltat;
        cell->getNodeField("vy") /= deltat;
        cell->getNodeField("vz") /= deltat;
    }
    tissue->saveVTK("solution","_n"+to_string(nSubdiv));

    cout << "L2 norm error vn = " << sqrt(tissue->getTissField("error")) << endl;

    MPI_Finalize();

    return 0;
}
