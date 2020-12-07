#include <iostream>
#include <fstream>
#include <random>

#include <AztecOO.h>

#include "aux.h"


int main(int argc, char **argv)
{
    using namespace std;
    using namespace ias;
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    MPI_Init(&argc, &argv);
    
    //---------------------------------------------------------------------------
    // [0] Input parameters
    //---------------------------------------------------------------------------
    int nx{4};
    int ny{1};
    int nz{1};
    double dx{2.1};
    double dy{2.1};
    double dz{2.1};
    double R{1.0};
    int nSubdiv{3};
    
    bool restart{false};
    string resLocation;
    string resFileName;

    double     intEL = 1.E-1;
    double     intCL = 5.E-2;
    double     intSt = 1.0;
    double   tension = 1.E-1;
    double viscosity = 1.E0;
    double frictiont = 1.E-1;
    double frictionn = 1.E-1;
    
    double lifetime = 1.0;

    double totTime{1.E3};
    double deltat{1.E-2};
    double stepFac{0.95};
    double maxDeltat{1.0};

    int    nr_maxite{5};
    double nr_restol{1.E-8};
    double nr_soltol{1.E-8};
        
    string  fEnerName{"energies.txt"};
    ofstream fEner;
    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);

        config.readInto(       nx, "nx");
        config.readInto(       ny, "ny");
        config.readInto(       nz, "nz");
        
        config.readInto(       dx, "dx");
        config.readInto(       dy, "dy");
        config.readInto(       dz, "dz");

        config.readInto(        R, "R");

        config.readInto(  nSubdiv, "nSubdiv");

        config.readInto(    intEL, "intEL");
        config.readInto(    intCL, "intCL");
        config.readInto(    intSt, "intSt");
        config.readInto(  tension, "tension");
        config.readInto(viscosity, "viscosity");
        config.readInto(frictiont, "frictiont");
        config.readInto(frictionn, "frictionn");
        
        config.readInto( lifetime, "lifetime");

        config.readInto(  totTime, "totTime");
        config.readInto(   deltat, "deltat");
        config.readInto(  stepFac, "stepFac");
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
        tissueGen->addNodeField("ux");
        tissueGen->addNodeField("uy");
        tissueGen->addNodeField("uz");
        tissueGen->addNodeField("x0");
        tissueGen->addNodeField("y0");
        tissueGen->addNodeField("z0");
        tissueGen->addNodeField("ux0");
        tissueGen->addNodeField("uy0");
        tissueGen->addNodeField("uz0");
        
        tissueGen->addGlobField("P");
        tissueGen->addGlobField("Paux");
        tissueGen->addGlobField("A");
        tissueGen->addGlobField("X");
        tissueGen->addGlobField("Y");
        tissueGen->addGlobField("Z");
        tissueGen->addGlobField("A0");
        tissueGen->addGlobField("X0");
        tissueGen->addGlobField("Y0");
        tissueGen->addGlobField("Z0");
        tissueGen->addGlobField("deltat");
        tissueGen->addGlobField("intEL");
        tissueGen->addGlobField("intCL");
        tissueGen->addGlobField("intSt");
        tissueGen->addGlobField("tension");
        tissueGen->addGlobField("viscosity");
        tissueGen->addGlobField("frictiont");
        tissueGen->addGlobField("frictionn");
        tissueGen->addGlobField("P0");
        tissueGen->addGlobField("Paux0");

        tissue = tissueGen->genRegularGridSpheres(nx, ny, nz, dx, dy, dz, R, nSubdiv);
        
//        tissue->reMesh({0,1},0.002);
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        tissue->balanceDistribution();
                
//        tissue->openPVD("Cell");
    }
//    else
//    {
//        tissue = rcp(new Tissue);
//        tissue->loadVTK(resLocation, resFileName);
//        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
//        tissue->updateGhosts();
//        tissue->balanceDistribution();
//        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
//
//
//        tissue->scaleNodeField(3, 0.0);
//        tissue->scaleNodeField(4, 0.0);
//        tissue->scaleNodeField(5, 0.0);
//        tissue->updateGhosts();
//
//        tissue->openPVD("Cell");
//        tissue->saveVTK("Cell","_t"+to_string(0),0.0);
//    }

    for(auto cell: tissue->getLocalCells())
    {
        cell->getGlobField("deltat") = deltat;
        cell->getGlobField("intEL") = intEL;
        cell->getGlobField("intCL") = intCL;
        cell->getGlobField("intSt") = intSt;
        cell->getGlobField("tension") = tension;
        cell->getGlobField("viscosity") = viscosity;
        cell->getGlobField("frictiont") = frictiont;
        cell->getGlobField("frictionn") = frictionn;
    }
    tissue->saveVTK("Cell","_t"+to_string(0),0.0);


    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({3,4,5});
    physicsIntegration->setGlobalDOFs({1});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setDoubleIntegrand(interaction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    physicsIntegration->setNumberOfCellIntegrals(8);
    physicsIntegration->setNumberOfGlobalIntegrals(4);
    physicsIntegration->setGlobalVariablesInInteractions(false);
    physicsIntegration->Update();


    RCP<Integration> eulerianUpdateIntegration = rcp(new Integration);
    eulerianUpdateIntegration->setTissue(tissue);
    eulerianUpdateIntegration->setNodeDOFs({0,1,2});
    eulerianUpdateIntegration->setGlobalDOFs({2});
    eulerianUpdateIntegration->setSingleIntegrand(eulerianUpdate);
    eulerianUpdateIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    eulerianUpdateIntegration->setNumberOfIntegrationPointsDoubleIntegral(1);
    eulerianUpdateIntegration->Update();

    int step{};
    double time{};

    fEner.open (fEnerName);
    fEner.close();
    
    int ierr{};
    bool rec_str{};
    while(time < totTime)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("x0")  = cell->getNodeField("x");
            cell->getNodeField("y0")  = cell->getNodeField("y");
            cell->getNodeField("z0")  = cell->getNodeField("z");
            cell->getNodeField("ux0") = cell->getNodeField("ux");
            cell->getNodeField("uy0") = cell->getNodeField("uy");
            cell->getNodeField("uz0") = cell->getNodeField("uz");
            
            cell->getGlobField("P0")    = cell->getGlobField("P");
            cell->getGlobField("Paux0") = cell->getGlobField("Paux");

        }
        tissue->updateGhosts();
        
        if(ierr == 0)
            rec_str = tissue->calculateInteractingElements(intEL+3.0*intCL);
        else if(ierr > 0 and rec_str)
        {
            physicsIntegration->recalculateMatrixStructure();
        }

        
        if(tissue->getMyPart()==0)
            cout << "Step " << step << ", time=" << time << ", deltat=" << deltat << endl;

        
        if(tissue->getMyPart()==0)
            cout << "Solving for velocities" << endl;
        
        ierr = NewtonRaphson(physicsIntegration,nr_restol,nr_soltol,nr_maxite);

        if ( ierr > 0 )
        {
            deltat *= stepFac;
            
            for(auto cell: tissue->getLocalCells())
            {
                cell->getNodeField("ux") = cell->getNodeField("ux0") * stepFac;
                cell->getNodeField("uy") = cell->getNodeField("uy0") * stepFac;
                cell->getNodeField("uz") = cell->getNodeField("uz0") * stepFac;
                cell->getGlobField("P")  = cell->getGlobField("P0");
            }
            tissue->updateGhosts();
        }
        else
        {
            
            physicsIntegration->setCellIntegralToCellField(3, 0);
            physicsIntegration->setCellIntegralToCellField(4, 1);
            physicsIntegration->setCellIntegralToCellField(5, 2);
            physicsIntegration->setCellIntegralToCellField(6, 3);
            physicsIntegration->setCellIntegralToCellField(7, 4);
            physicsIntegration->setCellIntegralToCellField(8, 5);
            physicsIntegration->setCellIntegralToCellField(9, 6);
            physicsIntegration->setCellIntegralToCellField(10, 7);
            
            int nIter = -ierr;
            
            //---------------------------------------------------------------------------
            if(tissue->getMyPart()==0)
                cout << "Solving for displacement along the normal" << endl;
            ierr = NewtonRaphson(eulerianUpdateIntegration,nr_restol,nr_soltol,nr_maxite);

            if (ierr>0)
            {
                cout << "CAREFUL!!! Solver for auxiliary field failed!" << endl;
                deltat *= stepFac;
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("x") = cell->getNodeField("x0");
                    cell->getNodeField("y") = cell->getNodeField("y0");
                    cell->getNodeField("z") = cell->getNodeField("z0");
                    cell->getGlobField("Paux") = cell->getGlobField("Paux0");

                }
                tissue->updateGhosts();
            }
            else
            {
                time += deltat;

                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("ux") /= deltat;
                    cell->getNodeField("uy") /= deltat;
                    cell->getNodeField("uz") /= deltat;
                }
                tissue->saveVTK("Cell","_t"+to_string(step+1),time);
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("ux") *= deltat;
                    cell->getNodeField("uy") *= deltat;
                    cell->getNodeField("uz") *= deltat;
                }

                if(tissue->getMyPart() == 0)
                {
                    fEner.open (fEnerName,ios::app);
                    fEner << setprecision(8) << scientific;
                    fEner << time << " " << deltat << " " << physicsIntegration->getGlobalIntegral(0) << " " << physicsIntegration->getGlobalIntegral(1) << " " << physicsIntegration->getGlobalIntegral(2) << " " << physicsIntegration->getGlobalIntegral(3) << endl;
                    fEner.close();
                }

                if(nIter < nr_maxite)
                {
                    deltat /= stepFac;
                    for(auto cell: tissue->getLocalCells())
                    {
                        cell->getNodeField("ux") /= stepFac;
                        cell->getNodeField("uy") /= stepFac;
                        cell->getNodeField("uz") /= stepFac;
                    }
                }
                if(deltat > maxDeltat)
                    deltat = maxDeltat;
                step++;
            }
        }

        for(auto cell: tissue->getLocalCells())
            cell->getGlobField("deltat") = deltat;
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        
        if(tissue->getMyPart()==0)
            cout << "Duration of time-step: " << elapsed.count() << endl;
    }
    
//    tissue->closePVD();
    
    MPI_Finalize();

    return 0;
}
