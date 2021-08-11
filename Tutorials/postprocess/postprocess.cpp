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
#include "ias_ParametrisationUpdate.h"


int main(int argc, char **argv)
{
    using namespace Tensor;
    using namespace std;
    using namespace ias;
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    MPI_Init(&argc, &argv);
    
    string resLocation;
    string resFileName;
    string outFileName;

    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);

        config.readInto(resLocation, "resLocation");
        config.readInto(resFileName, "resFileName");
        config.readInto(outFileName, "outFileName");
    }
    //---------------------------------------------------------------------------
    
    
    RCP<Tissue> tissue = rcp(new Tissue);
    tissue->loadVTK(resLocation, resFileName, BasisFunctionType::LoopSubdivision);
    tissue->Update();
    double intCL = tissue->getLocalCells()[0]->getCellField("intCL");
    double intEL = tissue->getLocalCells()[0]->getCellField("intEL");
    tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
    tissue->balanceDistribution();
    tissue->updateGhosts();
    tissue->calculateInteractingElements(intEL+3.0*intCL);

    std::vector<std::string> moments = {"M0", "M10", "M11", "M12", "M20", "M21", "M22", "M23", "M24"};
    for(auto cell: tissue->getLocalCells())
    {
        auto cellFieldnames = cell->getCellFieldNames();
        for(auto moment: moments)
        {
            if(std::find(cellFieldnames.begin(),cellFieldnames.end(),moment) == cellFieldnames.end())
                cell->addCellField(moment);
        }

        cell->Update();

        for(auto moment: moments)
            cell->getCellField(moment) = 0.0;

    }
    tissue->Update();


    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx"});
    physicsIntegration->setCellIntegralFields(moments);
    // physicsIntegration->setSingleIntegrand(leastSquaresTension);
    physicsIntegration->setSingleIntegrand(calculateMoments);
    // physicsIntegration->setDoubleIntegrand(leastSquaresTensionInteraction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    physicsIntegration->setCellDOFsInInteractions(false);
    physicsIntegration->Update();


    // RCP<solvers::TrilinosAztecOO> physicsLinearSolver = rcp(new solvers::TrilinosAztecOO);
    // physicsLinearSolver->setIntegration(physicsIntegration);
    // physicsLinearSolver->addAztecOOParameter("solver","gmres");
    // physicsLinearSolver->addAztecOOParameter("precond","dom_decomp");
    // physicsLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
    // physicsLinearSolver->addAztecOOParameter("output","none");
    // physicsLinearSolver->setMaximumNumberOfIterations(5000);
    // physicsLinearSolver->setResidueTolerance(1.E-8);
    // physicsLinearSolver->Update();

    // physicsIntegration->computeSingleIntegral();
    // physicsIntegration->computeDoubleIntegral();
    // physicsIntegration->assemble();
    // physicsLinearSolver->solve();
    // physicsIntegration->setSolToDOFs();
    
    physicsIntegration->InitialiseCellIntegralFields(0.0);
    physicsIntegration->computeSingleIntegral();
    physicsIntegration->assemble();
    physicsIntegration->InitialiseCellIntegralFields(0.0);
    physicsIntegration->computeSingleIntegral();
    physicsIntegration->assemble();


    tissue->saveVTK(outFileName,"");

    MPI_Finalize();

    return 0;
}
