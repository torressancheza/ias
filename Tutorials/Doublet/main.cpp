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

#include <ias_Belos.h>

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
    
    //---------------------------------------------------------------------------
    // [0] Input parameters
    //---------------------------------------------------------------------------
    double d{2.1};
    double R{1.0};
    int nSubdiv{3};
    
    bool restart{false};
    string resLocation;
    string resFileName;

    double     intEL = 1.E-1;
    double     intCL = 5.E-2;
    double     intSt = 1.0;
    double   tension = 1.E0;
    double  tensAsym = 0.0;
    double     kappa = 1.E-2;
    double viscosity = 1.E0;
    double frictiont = 1.E-3;
    double frictionn = 1.E-3;
    
    double totTime{1.E3};
    double deltat{1.E-2};
    double stepFac{0.95};
    double maxDeltat{1.0};

    int    nr_maxite{5};
    double nr_restol{1.E-8};
    double nr_soltol{1.E-8};

    int platonicSource{};
        
    string  fEnerName{"energies.txt"};
    ofstream fEner;
    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);

        config.readInto(       d, "d");

        config.readInto(        R, "R");

        config.readInto(  nSubdiv, "nSubdiv");
        config.readInto(  platonicSource, "platonicSource");

        config.readInto(    intEL, "intEL");
        config.readInto(    intCL, "intCL");
        config.readInto(    intSt, "intSt");
        config.readInto(  tension, "tension");
        config.readInto( tensAsym, "tensAsym");
        config.readInto(    kappa, "kappa");
        config.readInto(viscosity, "viscosity");
        config.readInto(frictiont, "frictiont");
        config.readInto(frictionn, "frictionn");
        
        config.readInto(  totTime,   "totTime");
        config.readInto(   deltat,   "deltat");
        config.readInto(  stepFac,   "stepFac");
        config.readInto(  maxDeltat, "maxDeltat");

        config.readInto(nr_maxite, "nr_maxite");
        config.readInto(nr_restol, "nr_restol");
        config.readInto(nr_soltol, "nr_soltol");

        config.readInto(fEnerName, "fEnerName");
        
        config.readInto(restart, "restart");
        config.readInto(resLocation, "resLocation");
        config.readInto(resFileName, "resFileName");
                
    }
    //---------------------------------------------------------------------------


    RCP<Tissue> tissue;
    if(!restart)
    {
        RCP<TissueGen> tissueGen = rcp( new TissueGen);
        tissueGen->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
        
        tissueGen->addNodeFields({"vx","vy","vz"});
        tissueGen->addNodeFields({"x0","y0","z0"});
        tissueGen->addNodeFields({"vx0","vy0","vz0"});
        tissueGen->addNodeFields({"xR","yR","zR"});

        tissueGen->addCellFields({"P", "P0"});
        tissueGen->addCellFields({"intEL","intCL","intSt","tension","kappa","viscosity","frictiont","frictionn"});
        tissueGen->addCellFields({"V0"});
        
        tissueGen->addTissFields({"time", "deltat"});
        tissueGen->addTissField("Ei");

        tissue = tissueGen->genRegularGridSpheres(2, 1, 1, d, 0, 0, R, nSubdiv, platonicSource);
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        tissue->balanceDistribution();
                
        tissue->getTissField("time") = 0.0;
        tissue->getTissField("deltat") = deltat;
        tissue->getTissField("Ei") = 0.0;

        for(auto cell: tissue->getLocalCells())
        { 
            cell->getCellField("intEL") = intEL;
            cell->getCellField("intCL") = intCL;
            cell->getCellField("intSt") = intSt;
            cell->getCellField("kappa") = kappa;
            if(cell->getCellField("cellId")==0)
                cell->getCellField("tension") = tension * (1+tensAsym);
            else
                cell->getCellField("tension") = tension * (1-tensAsym);
            cell->getCellField("viscosity") = viscosity;
            cell->getCellField("frictiont") = frictiont;
            cell->getCellField("frictionn") = frictionn;

            cell->getCellField("V0") = 4.0*M_PI/3.0;
        }
    }
    else
    {
        tissue = rcp(new Tissue);
        tissue->loadVTK(resLocation, resFileName, BasisFunctionType::LoopSubdivision);
        tissue->Update();
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->balanceDistribution();
        tissue->updateGhosts();
        
        deltat = tissue->getTissField("deltat");
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("vx") *= deltat;
            cell->getNodeField("vy") *= deltat;
            cell->getNodeField("vz") *= deltat;
        }
    }
    tissue->saveVTK("Cell","_t"+to_string(0));

    RCP<ParametrisationUpdate> paramUpdate = rcp(new ParametrisationUpdate);
    paramUpdate->setTissue(tissue);
    paramUpdate->setMethod(ParametrisationUpdate::Method::Eulerian);
    paramUpdate->setRemoveRigidBodyTranslation(true);
    paramUpdate->setRemoveRigidBodyRotation(true);
    paramUpdate->setDisplacementFieldNames({"vx","vy","vz"});
    paramUpdate->Update();

    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx","vy","vz"});
    physicsIntegration->setCellDOFs({"P"});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setDoubleIntegrand(interaction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(3);
    physicsIntegration->setTissIntegralFields({"Ei"});
    physicsIntegration->setCellDOFsInInteractions(false);
    physicsIntegration->Update();
    physicsIntegration->calculateInteractingGaussPoints(intEL+3.0*intCL);


    RCP<solvers::TrilinosBelos> physicsLinearSolver = rcp(new solvers::TrilinosBelos);
    physicsLinearSolver->setIntegration(physicsIntegration);
    physicsLinearSolver->setSolverType("GMRES");
    physicsLinearSolver->setMaximumNumberOfIterations(5000);
    physicsLinearSolver->setResidueTolerance(1.E-8);
    physicsLinearSolver->Update();
    
    RCP<solvers::NewtonRaphson> physicsNewtonRaphson = rcp(new solvers::NewtonRaphson);
    physicsNewtonRaphson->setLinearSolver(physicsLinearSolver);
    physicsNewtonRaphson->setSolutionTolerance(nr_soltol);
    physicsNewtonRaphson->setResidueTolerance(nr_restol);
    physicsNewtonRaphson->setMaximumNumberOfIterations(nr_maxite);
    physicsNewtonRaphson->setVerbosity(true);
    physicsNewtonRaphson->Update();

    for(auto cell: tissue->getLocalCells())
    {
        cell->getNodeField("xR") = cell->getNodeField("x");
        cell->getNodeField("yR") = cell->getNodeField("y");
        cell->getNodeField("zR") = cell->getNodeField("z");
    }

    int step{};
    double time = tissue->getTissField("time");

    fEner.open (fEnerName);
    fEner.close();

    int conv{};
    bool rec_str{};
    while(time < totTime)
    {
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("x0")  = cell->getNodeField("x");
            cell->getNodeField("y0")  = cell->getNodeField("y");
            cell->getNodeField("z0")  = cell->getNodeField("z");
            
            cell->getNodeField("vx0") = cell->getNodeField("vx");
            cell->getNodeField("vy0") = cell->getNodeField("vy");
            cell->getNodeField("vz0") = cell->getNodeField("vz");
            
            cell->getCellField("P0")    = cell->getCellField("P");
        }
        tissue->updateGhosts();
        
        if(conv)
            rec_str = max(rec_str, physicsIntegration->calculateInteractingGaussPoints(intEL+3.0*intCL));
        else if(rec_str)
        {
            physicsIntegration->recalculateMatrixStructure();
            physicsLinearSolver->recalculatePreconditioner();
            rec_str = false;
        }
        
        if(tissue->getMyPart()==0)
            cout << "Step " << step << ", time=" << time << ", deltat=" << deltat << endl;

        
        if(tissue->getMyPart()==0)
            cout << "Solving for velocities" << endl;
        
        physicsNewtonRaphson->solve();
        conv = physicsNewtonRaphson->getConvergence();
        
        if ( conv )
        {
            int nIter = physicsNewtonRaphson->getNumberOfIterations();
            
            conv = paramUpdate->UpdateParametrisation();

            
            if (conv)
            {
                if(tissue->getMyPart()==0)
                    cout << "Solved!"  << endl;
                
                time += deltat;
                tissue->getTissField("time") = time;

                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") /= deltat;
                    cell->getNodeField("vy") /= deltat;
                    cell->getNodeField("vz") /= deltat;
                }
                tissue->saveVTK("Cell","_t"+to_string(step+1));
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") *= deltat;
                    cell->getNodeField("vy") *= deltat;
                    cell->getNodeField("vz") *= deltat;
                }

                if(nIter < nr_maxite)
                {
                    deltat /= stepFac;
                    for(auto cell: tissue->getLocalCells())
                    {
                        cell->getNodeField("vx") /= stepFac;
                        cell->getNodeField("vy") /= stepFac;
                        cell->getNodeField("vz") /= stepFac;
                    }
                }
                if(deltat > maxDeltat)
                    deltat = maxDeltat;
                step++;

            }
            else
            {
                cout << "failed!" << endl;
                deltat *= stepFac;
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("x") = cell->getNodeField("x0");
                    cell->getNodeField("y") = cell->getNodeField("y0");
                    cell->getNodeField("z") = cell->getNodeField("z0");
                }
                tissue->updateGhosts();
            }
        }
        else
        {
            deltat *= stepFac;
            
            for(auto cell: tissue->getLocalCells())
            {
                cell->getNodeField("vx") = cell->getNodeField("vx0") * stepFac;
                cell->getNodeField("vy") = cell->getNodeField("vy0") * stepFac;
                cell->getNodeField("vz") = cell->getNodeField("vz0") * stepFac;
                cell->getCellField("P")  = cell->getCellField("P0");
            }
            tissue->updateGhosts();
        }        
        tissue->getTissField("deltat") = deltat;
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();        
    }
        
    MPI_Finalize();

    return 0;
}
