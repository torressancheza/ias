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
    double R{1.0};
    int nSubdiv{3};
    
    bool restart{false};
    string resLocation;
    string resFileName;

    double     intEL = 1.E-1;
    double     intCL = 5.E-2;
    double     intSt = 1.0;
    double   tension = 1.E-1;
    double     kappa = 1.E-2;
    double viscosity = 1.E0;
    double frictiont = 1.E-1;
    double frictionn = 1.E-1;
    
    double kConfin = 0.E3;
    double dConfinX = 0.0;
    double dConfinY = 0.0;
    double dConfinZ = 1.0;
    double tConfin  = 1.0;
    double cConfinX = 0.0;
    double cConfinY = 0.0;
    double cConfinZ = 0.0;
    double dt_tConfin = 0.0;

    double lifetime = 1.0;

    double totTime{1.E3};
    double deltat{1.E-2};
    double stepFac{0.95};
    double maxDeltat{1.E2};

    int    nr_maxite{5};
    double nr_restol{1.E-8};
    double nr_soltol{1.E-8};
        
    string  fEnerName{"energies.txt"};
    string  updateMethod{"eulerian"};
    ofstream fEner;
    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);


        config.readInto( kConfin, "kConfin");
        config.readInto( tConfin, "tConfin");
        config.readInto( dConfinX, "dConfinX");
        config.readInto( dConfinY, "dConfinY");
        config.readInto( dConfinZ, "dConfinZ");
        config.readInto( cConfinX, "cConfinX");
        config.readInto( cConfinY, "cConfinY");
        config.readInto( cConfinZ, "cConfinZ");
        config.readInto( dt_tConfin, "dt_tConfin");

        config.readInto(        R, "R");

        config.readInto(  nSubdiv, "nSubdiv");

        config.readInto(    intEL, "intEL");
        config.readInto(    intCL, "intCL");
        config.readInto(    intSt, "intSt");
        config.readInto(  tension, "tension");
        config.readInto(    kappa, "kappa");
        config.readInto(viscosity, "viscosity");
        config.readInto(frictiont, "frictiont");
        config.readInto(frictionn, "frictionn");
        
        config.readInto( lifetime, "lifetime");

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
        
        config.readInto(updateMethod, "updateMethod");
        
        if(updateMethod.compare("eulerian")!=0 and updateMethod.compare("ale") != 0)
            throw runtime_error("Update method not understood. It should be either \"eulerian\" or \"ALE\"");
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

        tissueGen->addCellFields({"P","Paux","P0","Paux0"});
        tissueGen->addCellFields({"A","X","Y","Z"});
        tissueGen->addCellFields({"A0","X0","Y0","Z0"});
        tissueGen->addCellFields({"Ai"});
        tissueGen->addCellFields({"Ec", "Eh", "Ei"});
        tissueGen->addCellFields({"AR"});

        tissueGen->addCellFields({"intEL","intCL","intSt","tension","kappa","viscosity","frictiont","frictionn"});
        
        tissueGen->addTissField("time");
        tissueGen->addTissField("deltat");

        tissueGen->addTissField("kConfin");
        tissueGen->addTissField("tConfin");
        tissueGen->addTissFields({"dConfinX", "dConfinY", "dConfinZ"});
        tissueGen->addTissFields({"cConfinX", "cConfinY", "cConfinZ"});

        tissue = tissueGen->genTripletSpheres(R,intEL,nSubdiv);

        for(auto cell: tissue->getLocalCells())
        {
            cout << "cell ID: " << cell->getCellField("cellId") << endl;
        }
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        tissue->balanceDistribution();
        
        tissue->calculateInteractingElements(intEL+3.0*intCL);
        
        tissue->getTissField("time") = 0.0;
        tissue->getTissField("deltat") = deltat;
        tissue->getTissField("kConfin") = kConfin;
        tissue->getTissField("tConfin") = tConfin;
        tissue->getTissField("dConfinX") = dConfinX;
        tissue->getTissField("dConfinY") = dConfinY;
        tissue->getTissField("dConfinZ") = dConfinZ;
        tissue->getTissField("cConfinX") = cConfinX;
        tissue->getTissField("cConfinY") = cConfinY;
        tissue->getTissField("cConfinZ") = cConfinZ;

        for(auto cell: tissue->getLocalCells())
        {
            cell->getCellField("intEL") = intEL;
            cell->getCellField("intCL") = intCL;
            cell->getCellField("intSt") = intSt;
            cell->getCellField("kappa") = kappa;
            cell->getCellField("tension") = tension;
            cell->getCellField("viscosity") = viscosity;
            cell->getCellField("frictiont") = frictiont;
            cell->getCellField("frictionn") = frictionn;
        }
        tissue->saveVTK("Cell","_t"+to_string(0));
    }
    else
    {
        tissue = rcp(new Tissue);
        tissue->loadVTK(resLocation, resFileName, BasisFunctionType::LoopSubdivision);
        tissue->Update();
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->balanceDistribution();
        tissue->updateGhosts();
        tissue->calculateInteractingElements(intEL+3.0*intCL);
        
        deltat = tissue->getTissField("deltat");
        
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("vx") *= 0.0;
            cell->getNodeField("vy") *= 0.0;
            cell->getNodeField("vz") *= 0.0;
        }
    }
        
    tissue->saveVTK("Cell","_t"+to_string(1));


    //FIXME: create a class for this so that the user doesn't need to see this
    vector<RCP<Tissue>> serialTissues;
    for(auto cell: tissue->getLocalCells())
    {
        RCP<Tissue> cellTissue = rcp(new Tissue(MPI_COMM_SELF));
        cellTissue->addCellToTissue(cell);
        cellTissue->setTissueFieldNames({"deltat", "ale_penalty_shear", "ale_penalty_stretch", "ale_max_shear", "ale_min_stretch", "ale_max_stretch", "Em", "nElem"});
        cellTissue->Update();
        cellTissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        cellTissue->getTissField("deltat") = 1.E-3;
        cellTissue->getTissField("ale_penalty_shear")  = 1.E0;
        cellTissue->getTissField("ale_penalty_stretch")  = 1.E0;
        cellTissue->getTissField("ale_max_shear") = 0.5;
        cellTissue->getTissField("ale_min_stretch") = 0.0;
        cellTissue->getTissField("ale_max_stretch") = 2.0;
        cellTissue->getTissField("Em") = 0.0;
        cellTissue->getTissField("nElem") = cell->getNumberOfElements();
        serialTissues.push_back(cellTissue);
    }
    
    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx","vy","vz"});
    physicsIntegration->setCellDOFs({"P"});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setDoubleIntegrand(interaction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(3);
    physicsIntegration->setCellIntegralFields({"A","X","Y","Z","A0","X0","Y0","Z0","Ai"});
    physicsIntegration->setCellDOFsInInteractions(false);
    physicsIntegration->Update();

    RCP<solvers::TrilinosAztecOO> physicsLinearSolver = rcp(new solvers::TrilinosAztecOO);
    physicsLinearSolver->setIntegration(physicsIntegration);
    physicsLinearSolver->addAztecOOParameter("solver","gmres");
    physicsLinearSolver->addAztecOOParameter("precond","dom_decomp");
    physicsLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
    physicsLinearSolver->addAztecOOParameter("output","none");
    physicsLinearSolver->setMaximumNumberOfIterations(5000);
    physicsLinearSolver->setResidueTolerance(1.E-8);
    physicsLinearSolver->Update();
    
    RCP<solvers::NewtonRaphson> physicsNewtonRaphson = rcp(new solvers::NewtonRaphson);
    physicsNewtonRaphson->setLinearSolver(physicsLinearSolver);
    physicsNewtonRaphson->setSolutionTolerance(1.E-8);
    physicsNewtonRaphson->setResidueTolerance(1.E-8);
    physicsNewtonRaphson->setMaximumNumberOfIterations(5);
    physicsNewtonRaphson->setVerbosity(true);
    physicsNewtonRaphson->Update();
    
    vector<RCP<Integration>> eulerianIntegrations;
    vector<RCP<solvers::TrilinosAztecOO>> eulerianLinearSolvers;
    vector<RCP<solvers::NewtonRaphson>> eulerianNewtonRaphsons;

    for(auto serialTissue: serialTissues)
    {
        RCP<Integration> eulerianIntegration = rcp(new Integration);
        eulerianIntegration->setTissue(serialTissue);
        eulerianIntegration->setNodeDOFs({"x","y","z"});
        eulerianIntegration->setCellDOFs({});
        if(updateMethod.compare("eulerian")==0)
            eulerianIntegration->setSingleIntegrand(eulerianUpdate);
        else
            eulerianIntegration->setSingleIntegrand(arbLagEulUpdate);
        eulerianIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
        eulerianIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
        eulerianIntegration->setTissIntegralFields({"Em"});
        eulerianIntegration->Update();
        eulerianIntegrations.push_back(eulerianIntegration);

        RCP<solvers::TrilinosAztecOO> eulerianLinearSolver = rcp(new solvers::TrilinosAztecOO);
        eulerianLinearSolver->setIntegration(eulerianIntegration);
        eulerianLinearSolver->addAztecOOParameter("solver","gmres");
        eulerianLinearSolver->addAztecOOParameter("precond","dom_decomp");
        eulerianLinearSolver->addAztecOOParameter("subdomain_solve","ilu");
        eulerianLinearSolver->addAztecOOParameter("output","none");
        eulerianLinearSolver->setMaximumNumberOfIterations(500);
        eulerianLinearSolver->setResidueTolerance(1.E-8);
        eulerianLinearSolver->Update();
        eulerianLinearSolvers.push_back(eulerianLinearSolver);

        RCP<solvers::NewtonRaphson> eulerianNewtonRaphson = rcp(new solvers::NewtonRaphson);
        eulerianNewtonRaphson->setLinearSolver(eulerianLinearSolver);
        eulerianNewtonRaphson->setSolutionTolerance(1.E-8);
        eulerianNewtonRaphson->setResidueTolerance(1.E-8);
        eulerianNewtonRaphson->setMaximumNumberOfIterations(5);
        eulerianNewtonRaphson->setVerbosity(true);
        eulerianNewtonRaphson->Update();

        eulerianNewtonRaphsons.push_back(eulerianNewtonRaphson);
    }

    int step{};
    double time = tissue->getTissField("time");

    fEner.open (fEnerName);
    fEner.close();
    
    physicsIntegration->computeSingleIntegral();
    for(auto cell: tissue->getLocalCells())
    {
        cell->getNodeField("xR") = cell->getNodeField("x");
        cell->getNodeField("yR") = cell->getNodeField("y");
        cell->getNodeField("zR") = cell->getNodeField("z");
        cell->getCellField("AR") = cell->getCellField("A");
    }

    int conv{};
    bool rec_str{};
    while(time < totTime)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("x0")  = cell->getNodeField("x");
            cell->getNodeField("y0")  = cell->getNodeField("y");
            cell->getNodeField("z0")  = cell->getNodeField("z");
            cell->getNodeField("vx0") = cell->getNodeField("vx");
            cell->getNodeField("vy0") = cell->getNodeField("vy");
            cell->getNodeField("vz0") = cell->getNodeField("vz");
            
            cell->getCellField("P0")    = cell->getCellField("P");
            cell->getCellField("Paux0") = cell->getCellField("Paux");
        }
        tissue->updateGhosts();
        
        auto finish_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_1 = finish_1 - start;

        if(conv)
            rec_str = max(rec_str,tissue->calculateInteractingElements(intEL+3.0*intCL));
        else if(rec_str)
        {
            physicsIntegration->recalculateMatrixStructure();
            physicsLinearSolver->DestroyPreconditioner();
            rec_str = false;
        }
        
        auto finish_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_2 = finish_2 - finish_1;
        
        if(tissue->getMyPart()==0)
            cout << "Step " << step << ", time=" << time << ", deltat=" << deltat << endl;

        
        if(tissue->getMyPart()==0)
            cout << "Solving for velocities" << endl;
        
        physicsNewtonRaphson->solve();
        conv = physicsNewtonRaphson->getConvergence();
        
        auto finish_3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_3 = finish_3 - finish_2;
        
        auto finish_4 = std::chrono::high_resolution_clock::now();
        auto finish_4_2 = std::chrono::high_resolution_clock::now();

        if ( conv )
        {
            int nIter = physicsNewtonRaphson->getNumberOfIterations();
            
            if(tissue->getMyPart()==0)
                cout << "Solving for displacement with " << updateMethod  << endl;

            conv = UpdateMeshes(eulerianNewtonRaphsons, updateMethod);
            
            MPI_Allreduce(MPI_IN_PLACE, &conv, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            finish_4 = std::chrono::high_resolution_clock::now();
            
            if (conv)
            {
                if(tissue->getMyPart()==0)
                    cout << "Solved!"  << endl;
                // for(auto cell: tissue->getLocalCells())
                // {
                //     cell->getNodeField("x") += cell->getCellField("X")/cell->getCellField("A") - cell->getCellField("X0")/cell->getCellField("A0");
                //     cell->getNodeField("y") += cell->getCellField("Y")/cell->getCellField("A") - cell->getCellField("Y0")/cell->getCellField("A0");
                //     cell->getNodeField("z") += cell->getCellField("Z")/cell->getCellField("A") - cell->getCellField("Z0")/cell->getCellField("A0");
                // }
                
                time += deltat;
                tissue->getTissField("time") = time;
                tissue->getTissField("tConfin") += dt_tConfin * deltat;

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

                finish_4_2 = std::chrono::high_resolution_clock::now();
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
                    cell->getCellField("Paux") = cell->getCellField("Paux0");

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
        std::chrono::duration<double> elapsed_4 = finish_4 - finish_3;
        std::chrono::duration<double> elapsed_4_2 = finish_4_2 - finish_4;
        
        tissue->getTissField("deltat") = deltat;
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        auto finish_5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_5 = finish_5 - finish_4;

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        
        if(tissue->getMyPart()==0)
        {
            cout << "Duration of time-step: " << elapsed.count() << endl;
            cout << "    "<< "Update ref: " << elapsed_1.count() << endl;
            cout << "    "<< "Calculate int elements: " << elapsed_2.count() << endl;
            cout << "    "<< "Newton Raphson for v: " << elapsed_3.count() << endl;
            cout << "    "<< "Newton Raphson for x: " << elapsed_4.count() << endl;
            cout << "    "<< "Print output: " << elapsed_4_2.count() << endl;
            cout << "    "<< "Update cell adjacency: " << elapsed_5.count() << endl;
        }
    }
    
//    tissue->closePVD();
    
    MPI_Finalize();

    return 0;
}
