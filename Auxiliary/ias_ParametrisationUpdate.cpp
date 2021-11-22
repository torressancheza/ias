//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <Teuchos_RCP.hpp>

#include <vtkTransform.h>
#include <vtkTransformFilter.h>

#include "Tensor.h"

#include "ias_ParametrisationUpdate.h"
#include "ias_BasicStructures.h"
#include "ias_Integration.h"
#include "ias_Amesos.h"
#include "ias_Belos.h"
#include "ias_AztecOO.h"
#include "ias_NewtonRaphson.h"

namespace ias
{
    void ParametrisationUpdate::Update()
    {
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        _checkSettings();

        switch (_method)
        {
            case Method::Lagrangian:
                _updateFunction = ParametrisationUpdate::_eulerianUpdate;
                break;
            
            case Method::Eulerian:
                // updateFunction = std::bind(&ParametrisationUpdate::_eulerianUpdate, *this, std::placeholders::_1);
                _updateFunction = ParametrisationUpdate::_eulerianUpdate;
                break;
            
            case Method::ALE:
                // updateFunction = std::bind(&ParametrisationUpdate::_arbLagEulUpdate, *this, std::placeholders::_1);
                _updateFunction = ParametrisationUpdate::_arbLagEulUpdate;
                break;

            default:
                break;
        }

        _cells.clear();
        _tissues.clear();
        _integrations.clear();
        _linearSolvers.clear();
        _newtonRaphsons.clear();

        //Create the set of local tissues per cell (this is just to have a well-defined integration)
        if(_method != Method::Lagrangian or _remove_RBT or _remove_RBR)
        {
            for(auto cell: _tissue->getLocalCells())
            {
                RCP<Tissue> cellTissue = rcp(new Tissue(MPI_COMM_SELF));
                cellTissue->addCellToTissue(cell);
                cellTissue->setTissueFieldNames({"A","X","Y","Z","A0","X0","Y0","Z0","LX","LY","LZ","Ixx","Ixy","Ixz","Iyy","Iyz","Izz"});
                cellTissue->Update();
                cellTissue->calculateCellCellAdjacency(1.0);
                _tissues.push_back(cellTissue);

                RCP<Integration> integration = rcp(new Integration);
                integration->setTissue(cellTissue);
                integration->setNodeDOFs({"x","y","z"});
                integration->setTissIntegralFields({"A","X","Y","Z","A0","X0","Y0","Z0","LX","LY","LZ","Ixx","Ixy","Ixz","Iyy","Iyz","Izz"});
                integration->setSingleIntegrand(_centreOfMass);
                integration->setNumberOfIntegrationPointsSingleIntegral(3); 
                integration->setNumberOfIntegrationPointsDoubleIntegral(1);
                integration->userAuxiliaryObjects.push_back(&_dispFieldNames);
                integration->Update();
                _integrations.push_back(integration);

                if(_method == Method::Eulerian)
                {
                    RCP<solvers::TrilinosBelos> linearSolver = rcp(new solvers::TrilinosBelos);
                    linearSolver->setIntegration(integration);
                    linearSolver->setSolverType("GMRES");
                    linearSolver->setMaximumNumberOfIterations(500);
                    linearSolver->setResidueTolerance(1.E-8);
                    linearSolver->Update();
                    _linearSolvers.push_back(linearSolver);
                }

                if(_method == Method::ALE)
                {
                    RCP<Cell> newCell = rcp(new Cell);
                    newCell->setBasisFunctionType(cell->getBasisFunctionType());
                    newCell->_connec = cell->_connec;
                    newCell->_nodeFields.resize(cell->_nodeFields.shape()[0],cell->_nodeFields.shape()[1]);
                    newCell->_nodeFields = cell->_nodeFields(Tensor::all,Tensor::range(0,2));
                    newCell->addNodeField("p");
                    newCell->addNodeField("x0");
                    newCell->addNodeField("y0");
                    newCell->addNodeField("z0");
                    newCell->addNodeField("xR");
                    newCell->addNodeField("yR");
                    newCell->addNodeField("zR");
                    newCell->Update();

                    newCell->getNodeField("x0") = newCell->getNodeField("x");
                    newCell->getNodeField("y0") = newCell->getNodeField("y");
                    newCell->getNodeField("z0") = newCell->getNodeField("z");
                    newCell->getNodeField("xR") = newCell->getNodeField("x");
                    newCell->getNodeField("yR") = newCell->getNodeField("y");
                    newCell->getNodeField("zR") = newCell->getNodeField("z");
                        
                    _cells.push_back(newCell);
                    
                    RCP<Tissue> cellTissue = rcp(new Tissue(MPI_COMM_SELF));
                    cellTissue->addCellToTissue(newCell);
                    cellTissue->setTissueFieldNames({"A","nElem","deltat","ale_penalty_shear","ale_penalty_stretch","tfriction","elastRef","Em"});
                    cellTissue->Update();
                    cellTissue->calculateCellCellAdjacency(1.0);
                    cellTissue->getTissField("deltat") = 1.E-4;
                    cellTissue->getTissField("ale_penalty_shear")  = _penaltyShear;
                    cellTissue->getTissField("ale_penalty_stretch")  = _penaltyStretch;
                    cellTissue->getTissField("tfriction") = _tfriction;
                    cellTissue->getTissField("elastRef") = _elastRef;
                    cellTissue->getTissField("Em") = 0.0;
                    cellTissue->getTissField("nElem") = cell->getNumberOfElements();
                    _tissues.push_back(cellTissue);
                                    
                    RCP<Integration> integration = rcp(new Integration);
                    integration->setTissue(cellTissue);
                    integration->setNodeDOFs({"x","y","z","p"});
                    integration->setSingleIntegrand(_updateFunction);
                    integration->setTissIntegralFields({"A","Em"});
                    integration->setNumberOfIntegrationPointsSingleIntegral(3); 
                    integration->setNumberOfIntegrationPointsDoubleIntegral(1);
                    integration->Update();
                    integration->computeSingleIntegral();
                    integration->assemble();
                    _integrations.push_back(integration);
                
                    RCP<solvers::TrilinosBelos> linearSolver = rcp(new solvers::TrilinosBelos);
                    linearSolver->setIntegration(integration);
                    linearSolver->setSolverType("GMRES");
                    linearSolver->setMaximumNumberOfIterations(500);
                    linearSolver->setResidueTolerance(1.E-8);
                    linearSolver->Update();
                    _linearSolvers.push_back(linearSolver);

                    RCP<solvers::NewtonRaphson> newtonRaphson = rcp(new solvers::NewtonRaphson);
                    newtonRaphson->setLinearSolver(linearSolver);
                    newtonRaphson->setSolutionTolerance(1.E-8);
                    newtonRaphson->setResidueTolerance(1.E-8);
                    newtonRaphson->setMaximumNumberOfIterations(5);
                    newtonRaphson->setVerbosity(false);
                    newtonRaphson->Update();
                    _newtonRaphsons.push_back(newtonRaphson);
                }    
            }
        }
        
    }

    void ParametrisationUpdate::_checkSettings()
    {
        if(_dispFieldNames.size() != 3)
            throw std::runtime_error("ParametrisationUpdate:: The number of velocity fields should be 3");

        if(_ale_param_set and _method != Method::ALE)
            if(_tissue->getMyPart() == 0)
                cout << "ParametrisationUpdate::Warning: ALE parameters have been set but ALE is not being used." << endl;
    }

    bool ParametrisationUpdate::UpdateParametrisation()
    {
        int conv{1};

        if(_remove_RBT or _remove_RBR)
        {
            using namespace Tensor;

            std::vector<double> vMPI0(4,0.0), vMPI(4,0.0), lMPI(3, 0.0), iMPI(6, 0.0);
            double &A0 = vMPI0[0], &X0 = vMPI0[1], &Y0 = vMPI0[2], &Z0 = vMPI0[3];
            double &A = vMPI[0], &X = vMPI[1], &Y = vMPI[2], &Z = vMPI[3];
            double &LX = lMPI[0], &LY = lMPI[1], &LZ = lMPI[2];
            double &Ixx = iMPI[0], &Ixy = iMPI[1], &Ixz = iMPI[2], &Iyy = iMPI[3], &Iyz = iMPI[4], &Izz = iMPI[5];

            int n{};
            for(auto integration: _integrations)
            {
                if(_method == Method::ALE and n % 2 == 1)
                {
                    n++;
                    continue;
                }

                integration->InitialiseTissIntegralFields(0.0);
                integration->InitialiseCellIntegralFields(0.0);
                integration->computeSingleIntegral();
                integration->assemble();

                A += integration->getTissue()->getTissField("A");
                X += integration->getTissue()->getTissField("X");
                Y += integration->getTissue()->getTissField("Y");
                Z += integration->getTissue()->getTissField("Z");

                A0 += integration->getTissue()->getTissField("A0");
                X0 += integration->getTissue()->getTissField("X0");
                Y0 += integration->getTissue()->getTissField("Y0");
                Z0 += integration->getTissue()->getTissField("Z0");

                LX += integration->getTissue()->getTissField("LX");
                LY += integration->getTissue()->getTissField("LY");
                LZ += integration->getTissue()->getTissField("LZ");

                Ixx += integration->getTissue()->getTissField("Ixx");
                Ixy += integration->getTissue()->getTissField("Ixy");
                Ixz += integration->getTissue()->getTissField("Ixz");
                Iyy += integration->getTissue()->getTissField("Iyy");
                Iyz += integration->getTissue()->getTissField("Iyz");
                Izz += integration->getTissue()->getTissField("Izz");
                n++;
            }
            
            MPI_Allreduce(MPI_IN_PLACE, vMPI0.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, vMPI.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, lMPI.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, iMPI.data(), 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            tensor<double,1> Xvec = {X/A, Y/A, Z/A};
            tensor<double,1> X0vec = {X0/A0, Y0/A0, Z0/A0};
            tensor<double,1> V = Xvec-X0vec;
            tensor<double,1> L = {LX, LY, LZ};
            L -= 0.5 * V * antisym3D * (Xvec+X0vec);

            tensor<double,2> I = {{Ixx, Ixy, Ixz}, {Ixy, Iyy, Iyz}, {Ixz, Iyz, Izz}};
            I += 0.5 * (A * ((Xvec*Xvec) * Identity(3) - outer(Xvec,Xvec)) + A0 * ((X0vec*X0vec) * Identity(3) - outer(X0vec,X0vec)));
            tensor<double,1> omega = I.inv() * L;
            double angle = sqrt(omega*omega);
            tensor<double, 1> axis = omega/sqrt(omega*omega);

            for(auto cell: _tissue->getLocalCells())
            {
                if(_remove_RBT)
                {
                    cell->getNodeField(_dispFieldNames[0]) -= V(0);
                    cell->getNodeField(_dispFieldNames[1]) -= V(1);
                    cell->getNodeField(_dispFieldNames[2]) -= V(2);
                }

                if(_remove_RBR)
                {
                    for(int i = 0; i < cell->getNumberOfPoints(); i++)
                    {
                        tensor<double,1> x0 = {cell->getNodeField("x")(i), cell->getNodeField("y")(i), cell->getNodeField("z")(i)};
                        tensor<double,1> v = {cell->getNodeField(_dispFieldNames[0])(i), cell->getNodeField(_dispFieldNames[1])(i), cell->getNodeField(_dispFieldNames[2])(i)};
                        tensor<double,1> x = x0+v;

                        tensor<double,1> rot = omega * antisym3D * (0.5*(x+x0)-0.5*(Xvec+X0vec));

                        cell->getNodeField(_dispFieldNames[0])(i) -= rot(0);
                        cell->getNodeField(_dispFieldNames[1])(i) -= rot(1);
                        cell->getNodeField(_dispFieldNames[2])(i) -= rot(2);
                    }
                }
            }
        }

        if(_method == Method::Lagrangian)
        {
            for(auto cell: _tissue->getLocalCells())
            {
                cell->getNodeField("x") += cell->getNodeField(_dispFieldNames[0]);
                cell->getNodeField("y") += cell->getNodeField(_dispFieldNames[1]);
                cell->getNodeField("z") += cell->getNodeField(_dispFieldNames[2]);
            }
        }
        else
        {
            switch (_method)
            {                
                case Method::Eulerian:
                {
                    for(auto linearSolver: _linearSolvers)
                    {
                        linearSolver->getIntegration()->setSingleIntegrand(_updateFunction);

                        linearSolver->getIntegration()->InitialiseTissIntegralFields(0.0);
                        linearSolver->getIntegration()->InitialiseCellIntegralFields(0.0);
                        linearSolver->getIntegration()->computeSingleIntegral();
                        linearSolver->getIntegration()->assemble();

                        linearSolver->getIntegration()->InitialiseTissIntegralFields(0.0);
                        linearSolver->getIntegration()->InitialiseCellIntegralFields(0.0);
                        linearSolver->getIntegration()->fillVectorWithScalar(0.0);
                        linearSolver->getIntegration()->fillSolutionWithScalar(0.0);
                        linearSolver->getIntegration()->fillMatrixWithScalar(0.0);
                        linearSolver->getIntegration()->computeSingleIntegral();
                        linearSolver->getIntegration()->assemble();
                        linearSolver->solve();
                        conv = linearSolver->getConvergence();
                        if(conv)
                            linearSolver->getIntegration()->setSolToDOFs();

                        linearSolver->getIntegration()->setSingleIntegrand(_centreOfMass);
                    }
                    break;
                }
                case Method::ALE:
                {
                    using namespace Tensor;


                    double stepFactor = 1.05;
                    int loc_idx{};
                    for(auto newtonRaphson: _newtonRaphsons)
                    {
                        auto linearSolver = newtonRaphson->getLinearSolver();
                        auto integration = newtonRaphson->getIntegration();
                        auto tissue = integration->getTissue();
                        auto newCell = tissue->getLocalCells()[0];
                        auto cell = _tissue->getLocalCells()[loc_idx];

                        newCell->getNodeField("x") = cell->getNodeField("x") + cell->getNodeField(_dispFieldNames[0]);
                        newCell->getNodeField("y") = cell->getNodeField("y") + cell->getNodeField(_dispFieldNames[1]);
                        newCell->getNodeField("z") = cell->getNodeField("z") + cell->getNodeField(_dispFieldNames[2]);
                        newCell->getNodeField("x0") = newCell->getNodeField("x");
                        newCell->getNodeField("y0") = newCell->getNodeField("y");
                        newCell->getNodeField("z0") = newCell->getNodeField("z");
                        newCell->getNodeField("xR") = newCell->getNodeField("x");
                        newCell->getNodeField("yR") = newCell->getNodeField("y");
                        newCell->getNodeField("zR") = newCell->getNodeField("z");

                        integration->fillVectorWithScalar(0.0);
                        integration->computeSingleIntegral();
                        integration->assemble();
                        double res{};
                        integration->getVector()->NormInf(&res);
                        int n{};

                        while( res/tissue->getTissField("deltat") > 1.E-4)                      
                        {
                            newtonRaphson->solve();
 
                            conv = newtonRaphson->getConvergence();
                            if(conv != 1)
                            {
                                tissue->getTissField("deltat") /= stepFactor;
                                if(tissue->getTissField("deltat") < 1.E-6)
                                {
                                    cout << tissue->getTissField("deltat") << endl;
                                    break;
                                }
                            }
                            else 
                            {
                                newCell->getNodeField("x0") = newCell->getNodeField("x");
                                newCell->getNodeField("y0") = newCell->getNodeField("y");
                                newCell->getNodeField("z0") = newCell->getNodeField("z");

                                if(newtonRaphson->getNumberOfIterations() <= 4 )
                                    tissue->getTissField("deltat") *= stepFactor;


                                res = newtonRaphson->getResiduals()[0]; //Forces in the first time-step of NR are only due to mesh distortion

                                n++;
                            }

                            tissue->getTissField("deltat") = std::min(tissue->getTissField("deltat"),_deltatMax);
                        }

                        if(conv)
                        {
                            cell->getNodeField("x") = newCell->getNodeField("x");
                            cell->getNodeField("y") = newCell->getNodeField("y");
                            cell->getNodeField("z") = newCell->getNodeField("z");
                        }

                        loc_idx++;
                    }
                    break;
                }

                default:
                    break;
                    
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, &conv, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        return conv;
    }

    void ParametrisationUpdate::_centreOfMass(Teuchos::RCP<ias::SingleIntegralStr> fill)
    {
        using namespace std;
        using namespace Tensor;
        
        //[1] INPUT
        int eNN    = fill->eNN;
        tensor<double,1>  bfs(fill->bfs[0].data(),eNN);
        tensor<double,2> Dbfs(fill->bfs[1].data(),eNN,2);

        tensor<double,2> nborFields   = fill->nborFields;

        std::vector<std::string>* velFieldNames = static_cast<std::vector<std::string>*>(fill->userAuxiliaryObjects[0]);

        int idx_x = fill->idxNodeField("x");
        int idx_z = fill->idxNodeField("z");
        int idx_vx = fill->idxNodeField((*velFieldNames)[0]);
        int idx_vz = fill->idxNodeField((*velFieldNames)[2]);

        //[2.1] Geometry in the configuration at previous time-step
        tensor<double,1>      x0 = bfs * nborFields(all,range(idx_x,idx_z));
        tensor<double,2>     Dx0 = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
        tensor<double,1>  cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
        double              jac0 = sqrt(cross0*cross0);
        tensor<double,1> normal0 = cross0/jac0;
        double              x0n0 = x0*normal0;

        //[2.2] Geometry in current configuration
        tensor<double,1>      x = bfs * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
        tensor<double,2>     Dx = Dbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        double               xn = x * normal;
        tensor<double,2> metric = Dx * Dx.T();

        fill->tissIntegrals(fill->idxTissIntegral("A0")) += fill->w_sample * jac0;
        fill->tissIntegrals(fill->idxTissIntegral("X0")) += fill->w_sample * jac0 * x0(0);
        fill->tissIntegrals(fill->idxTissIntegral("Y0")) += fill->w_sample * jac0 * x0(1);
        fill->tissIntegrals(fill->idxTissIntegral("Z0")) += fill->w_sample * jac0 * x0(2);
        fill->tissIntegrals(fill->idxTissIntegral("A")) += fill->w_sample * jac;
        fill->tissIntegrals(fill->idxTissIntegral("X")) += fill->w_sample * jac * x(0);
        fill->tissIntegrals(fill->idxTissIntegral("Y")) += fill->w_sample * jac * x(1);
        fill->tissIntegrals(fill->idxTissIntegral("Z")) += fill->w_sample * jac * x(2);

        tensor<double,1> L = fill->w_sample * 0.5 * ((x-x0) * antisym3D * (jac * x + jac0 * x0));
        fill->tissIntegrals(fill->idxTissIntegral("LX")) += L(0);
        fill->tissIntegrals(fill->idxTissIntegral("LY")) += L(1);
        fill->tissIntegrals(fill->idxTissIntegral("LZ")) += L(2);

        tensor<double,2> I =  fill->w_sample * 0.5 * (jac * (outer(x,x) - (x*x) * Identity(3)) + jac0 * (outer(x0,x0) - (x0*x0) * Identity(3)));
        fill->tissIntegrals(fill->idxTissIntegral("Ixx")) += I(0,0);
        fill->tissIntegrals(fill->idxTissIntegral("Iyy")) += I(1,1);
        fill->tissIntegrals(fill->idxTissIntegral("Izz")) += I(2,2);
        fill->tissIntegrals(fill->idxTissIntegral("Ixy")) += I(0,1);
        fill->tissIntegrals(fill->idxTissIntegral("Ixz")) += I(0,2);
        fill->tissIntegrals(fill->idxTissIntegral("Iyz")) += I(1,2);
    }

    void ParametrisationUpdate::_eulerianUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill)
    {
        using namespace std;
        using namespace Tensor;
        
        //[1] INPUT
        int eNN    = fill->eNN;
        tensor<double,1>  bfs(fill->bfs[0].data(),eNN);
        tensor<double,2> Dbfs(fill->bfs[1].data(),eNN,2);

        tensor<double,2> nborFields   = fill->nborFields;
        tensor<double,1> V = fill->tissFields(range(fill->idxTissField("X"),fill->idxTissField("Z")))/fill->tissFields(fill->idxTissField("A")) - fill->tissFields(range(fill->idxTissField("X0"),fill->idxTissField("Z0")))/fill->tissFields(fill->idxTissField("A0"));

        std::vector<std::string>* velFieldNames = static_cast<std::vector<std::string>*>(fill->userAuxiliaryObjects[0]);

        int idx_x = fill->idxNodeField("x");
        int idx_z = fill->idxNodeField("z");
        int idx_vx = fill->idxNodeField((*velFieldNames)[0]);
        int idx_vz = fill->idxNodeField((*velFieldNames)[2]);

        //[2.1] Geometry in the configuration at previous time-step
        tensor<double,1>      x = bfs * nborFields(all,range(idx_x,idx_z));
        tensor<double,2>     Dx = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        double               xn = x * normal;
        tensor<double,2> metric = Dx * Dx.T();

        tensor<double,1>      v  = bfs * nborFields(all,range(idx_vx,idx_vz));
        tensor<double,2>     Dx_ = Dbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
        tensor<double,1>  cross_ = Dx_(1,all) * antisym3D * Dx_(0,all);
        double             jac_  = sqrt(cross_*cross_);

        tensor<double,2>& rhs_n = fill->vec_n;
        tensor<double,4>& A_nn  = fill->mat_nn;

        rhs_n += fill->w_sample * jac * outer(bfs,x+V+((v-V)*normal)*normal);
        A_nn  += fill->w_sample * jac * outer(outer(bfs,bfs),Identity(3)).transpose({0,2,1,3});

        fill->tissIntegrals(fill->idxTissIntegral("A0")) += fill->w_sample * jac;
        fill->tissIntegrals(fill->idxTissIntegral("X0")) += fill->w_sample * jac * x(0);
        fill->tissIntegrals(fill->idxTissIntegral("Y0")) += fill->w_sample * jac * x(1);
        fill->tissIntegrals(fill->idxTissIntegral("Z0")) += fill->w_sample * jac * x(2);
        fill->tissIntegrals(fill->idxTissIntegral("A")) += fill->w_sample * jac_;
        fill->tissIntegrals(fill->idxTissIntegral("X")) += fill->w_sample * jac_ * (x(0)+v(0));
        fill->tissIntegrals(fill->idxTissIntegral("Y")) += fill->w_sample * jac_ * (x(1)+v(1));
        fill->tissIntegrals(fill->idxTissIntegral("Z")) += fill->w_sample * jac_ * (x(2)+v(2));

    }

    void ParametrisationUpdate::_arbLagEulUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill)
    {
        using namespace std;
        using namespace Tensor;
        
        //[1] INPUT
        int eNN    = fill->eNN;
        tensor<double,1>  bfs(fill->bfs[0].data(),eNN);
        tensor<double,2> Dbfs(fill->bfs[1].data(),eNN,2);
        tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);

        tensor<double,2>   nborFields = fill->nborFields;
        tensor<double,1>&  tissFields = fill->tissFields;

        tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                                {{0.0,0.0},{0.0,1.0}},
                                {{0.0,1.0},{1.0,0.0}}};
        
        double deltat    = tissFields(fill->idxTissField("deltat"));

        double tfriction = tissFields(fill->idxTissField("tfriction"));
        double elastRef = tissFields(fill->idxTissField("elastRef"));
        double ale_penalty_shear = tissFields(fill->idxTissField("ale_penalty_shear")) * deltat;
        double ale_penalty_stretch = tissFields(fill->idxTissField("ale_penalty_stretch")) * deltat;
        double nElem = tissFields(fill->idxTissField("nElem"));
        double A  = fill->tissFields(fill->idxTissField("A"));


        int idx_x = fill->idxNodeField("x");
        int idx_z = fill->idxNodeField("z");
        int idx_x0 = fill->idxNodeField("x0");
        int idx_z0 = fill->idxNodeField("z0");
        int idx_xR = fill->idxNodeField("xR");
        int idx_zR = fill->idxNodeField("zR");

        tensor<double,1>      xR = bfs * nborFields(all,range(idx_xR,idx_zR));

        //[2.1] Geometry in the configuration at previous time-step
        tensor<double,1>      x0 = bfs * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,2>     Dx0 = Dbfs.T() * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,1>  cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
        double              jac0 = sqrt(cross0*cross0);
        tensor<double,1> normal0 = cross0/jac0;
        double              x0n0 = x0*normal0;
        tensor<double,2> metric0 = Dx0 * Dx0.T();
        tensor<double,2> imetric0 = metric0.inv();
        
        //[2.2] Geometry in current configuration
        tensor<double,1>      x = bfs * nborFields(all,range(idx_x,idx_z));
        tensor<double,2>     Dx = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        tensor<double,2> metric = Dx * Dx.T();
        tensor<double,2> imetric = metric.inv();


        //Linear interpolation
        tensor<double,2> Dbfs_l = {{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}};
        array<int,3> idx = {0,1,eNN-6}; //FIXME: this only works for subdivision surfaces

        tensor<double,2> nborFields_l(3,3);
        for(int i = 0; i < 3; i++)
            nborFields_l(i,all) = nborFields(idx[i],range(idx_x,idx_z));
         
        // cout << "HOLA e1" << endl;
        // tensor<double,1>      x_l = bfs_l * nborFields_l;
        tensor<double,2>     Dx_l = Dbfs_l.T() * nborFields_l;
        tensor<double,1>  cross_l = Dx_l(1,all) * antisym3D * Dx_l(0,all);
        double              jac_l = sqrt(cross_l*cross_l);
        tensor<double,2> metric_l = Dx_l * Dx_l.T();
        tensor<double,2> imetric_l = metric_l.inv();

        //[2.4] First-order derivatives of geometric quantities wrt nodal positions
        tensor<double,4> dDx_l        = outer(Dbfs_l,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross_l     = dDx_l(all,all,1,all)*antisym3D*Dx_l(0,all) - dDx_l(all,all,0,all)*antisym3D*Dx_l(1,all);
        tensor<double,2> djac_l       = 1./jac_l * dcross_l * cross_l;
        tensor<double,4> dmetric_l    = dDx_l * Dx_l.T();
                        dmetric_l   += dmetric_l.transpose({0,1,3,2});
        tensor<double,4> dmetric_l_CC = product(product(dmetric_l,imetric_l,{{2,0}}),imetric_l,{{2,0}});
        tensor<double,3> dnormal_l    = dcross_l/jac_l - outer(djac_l/(jac_l*jac_l),cross_l);

        //[2.5] Second-order derivatives of geometric quantities wrt nodal positions
        tensor<double,5> ddcross_l    = (dDx_l(all,all,1,all)*antisym3D*dDx_l(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                        ddcross_l  += ddcross_l.transpose({2,3,0,1,4});
        tensor<double,4> ddjac_l      = 1./jac_l * (ddcross_l * cross_l + product(dcross_l,dcross_l,{{2,2}}) - outer(djac_l,djac_l));
        tensor<double,5> ddnormal_l   = ddcross_l/jac_l - outer(dcross_l,djac_l/(jac_l*jac_l)).transpose({0,1,3,4,2}) - outer(djac_l/(jac_l*jac_l),dcross_l) - outer(ddjac_l/(jac_l*jac_l),cross_l) + 2.0/(jac_l*jac_l*jac_l) * outer(outer(djac_l,djac_l),cross_l);
        tensor<double,6> ddmetric_l   = 2.0 * product(dDx_l,dDx_l,{{3,3}}).transpose({0,1,3,4,2,5});


        double angle = 2.*(M_PI)/(eNN-6); //FIXME: this only works for subdivision surfaces
        tensor<double,2> DxR = {{1.0, 0.0}, {cos(angle), sin(angle)}};
        DxR *= sqrt(2.0*A/nElem/sin(angle));// / sqrt(c1_0*c1_0+c2_0*c2_0);
        tensor<double,2> metricR = DxR * DxR.T();
        double jacR = sqrt(metricR.det());
        tensor<double,2> imetricR = metricR.inv();
        double J = jac_l/jacR;
        double I1 = product(imetricR, metric_l,{{0,0},{1,1}});
        double shear = 1.0-4.0*(J*J)/(I1*I1) > 0 ? sqrt(1.0-4.0*(J*J)/(I1*I1)) : 0.0;

        double energy{};
        tensor<double,2> rhs_l(3,3);
        tensor<double,4> A_l(3,3,3,3);

        rhs_l = 0.0;
        A_l = 0.0;

        tensor<double,2> dI1 = product(imetricR,dmetric_l,{{0,2},{1,3}});
        tensor<double,4> ddI1 = product(imetricR,ddmetric_l,{{0,4},{1,5}});
        energy += fill->w_sample * jacR * (ale_penalty_shear * (I1-2) + ale_penalty_stretch * (J-1)*(J-1))/deltat;
        rhs_l += fill->w_sample * (jacR * ale_penalty_shear * dI1 + ale_penalty_stretch * 2.0 * (J-1) * djac_l);
        A_l += fill->w_sample * (jacR * ale_penalty_shear * ddI1 + ale_penalty_stretch * 2.0 * ((J-1) * ddjac_l + outer(djac_l,djac_l)/jacR));
        
        tensor<double,2>& rhs_n = fill->vec_n;
        tensor<double,4>& A_nn  = fill->mat_nn;
        for(int i1 = 0; i1 < 3; i1++)
        {
            for(int i2 = 0; i2 < 3; i2++)
                A_nn(idx[i1],range(0,2),idx[i2],range(0,2)) += A_l(i1,all,i2,all);

            rhs_n(idx[i1],range(0,2)) += rhs_l(i1,all);
        }
        rhs_n(all,range(0,2)) += tfriction * fill->w_sample * jac0 * outer(bfs, x-x0);
        A_nn(all,range(0,2),all,range(0,2))  += tfriction * fill->w_sample * jac0 * outer(outer(bfs, bfs), Identity(3)).transpose({0,2,1,3});

        rhs_n(all,range(0,2)) += elastRef * fill->w_sample * jac0 * outer(bfs, x-xR);
        A_nn(all,range(0,2),all,range(0,2))  += elastRef * fill->w_sample * jac0 * outer(outer(bfs, bfs), Identity(3)).transpose({0,2,1,3});

        double p = bfs * fill->nborFields(all,fill->idxNodeField("p"));
        rhs_n(all,range(0,2)) += fill->w_sample * jac0 * p * outer(bfs,normal0);
        rhs_n(all,3) += fill->w_sample * jac0 * bfs * ((x-x0)*normal0);
        A_nn(all,range(0,2),all,3) += fill->w_sample * jac0 * outer(outer(bfs,normal0),bfs);
        A_nn(all,3,all,range(0,2)) += fill->w_sample * jac0 * outer(bfs,outer(bfs,normal0));
        double kappa = 1.E-6;
        double J0 = jac0/jacR;
        A_nn(all,3,all,3) += kappa * fill->w_sample * jac0 * outer(bfs,bfs);
 
        fill->tissIntegrals(fill->idxTissIntegral("A")) += fill->w_sample * jac_l;
        fill->tissIntegrals(fill->idxTissIntegral("Em")) += fill->w_sample * jac * energy;
    }

}