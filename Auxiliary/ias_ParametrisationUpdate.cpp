//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <Teuchos_RCP.hpp>

#include "ias_ParametrisationUpdate.h"
#include "ias_BasicStructures.h"
#include "ias_Integration.h"
#include "ias_AztecOO.h"
#include "ias_NewtonRaphson.h"

namespace ias
{
    void ParametrisationUpdate::Update()
    {
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        _checkSettings();

        std::function<void(Teuchos::RCP<SingleIntegralStr>)> updateFunction;
        
        switch (_method)
        {
            case Method::Lagrangian:

                break;
            
            case Method::Eulerian:
                updateFunction = std::bind(&ParametrisationUpdate::_eulerianUpdate, *this, std::placeholders::_1);
                break;
            
            case Method::ALE:
                updateFunction = std::bind(&ParametrisationUpdate::_arbLagEulUpdate, *this, std::placeholders::_1);
                break;
            default:
                break;
        }

        //Create the set of local tissues per cell (this is just to have a well-defined integration)
        for(auto cell: _tissue->getLocalCells())
        {
            RCP<Tissue> cellTissue = rcp(new Tissue(MPI_COMM_SELF));
            cellTissue->addCellToTissue(cell);
            cellTissue->Update();
            cellTissue->calculateCellCellAdjacency(1.0);
            _tissues.push_back(cellTissue);
 
            RCP<Integration> integration = rcp(new Integration);
            integration->setTissue(cellTissue);
            integration->setNodeDOFs({"x","y","z"}); //TODO: make it depend on user's input
            integration->setCellDOFs({"Paux"}); //FIXME: this is a dummy variable that we don't want to explicitly add to the cell... but how can we bypass this? Maybe remove? It seems this volume conservation is not entirely relevant. For Eulerian this would make the system linear, which is way better
            integration->setSingleIntegrand(updateFunction);
            integration->setNumberOfIntegrationPointsSingleIntegral(3); //TODO: make it a parameter to set
            integration->setNumberOfIntegrationPointsDoubleIntegral(1);
            integration->Update();
            _integrations.push_back(integration);

            RCP<solvers::TrilinosAztecOO> linearSolver = rcp(new solvers::TrilinosAztecOO);
            linearSolver->setIntegration(integration);
            linearSolver->addAztecOOParameter("solver","gmres"); //TODO: these options are pretty generic and should work in almost all cases... but the number of iterations and the tolerance should be read
            linearSolver->addAztecOOParameter("precond","dom_decomp");
            linearSolver->addAztecOOParameter("subdomain_solve","ilu");
            linearSolver->addAztecOOParameter("output","none");
            linearSolver->setMaximumNumberOfIterations(500);
            linearSolver->setResidueTolerance(1.E-8);
            linearSolver->Update();
            _linearSolvers.push_back(linearSolver);

            RCP<solvers::NewtonRaphson> newtonRaphson = rcp(new solvers::NewtonRaphson);
            newtonRaphson->setLinearSolver(linearSolver);
            newtonRaphson->setSolutionTolerance(1.E-8); //TODO: read the parameters
            newtonRaphson->setResidueTolerance(1.E-8);
            newtonRaphson->setMaximumNumberOfIterations(5);
            newtonRaphson->setVerbosity(false);
            newtonRaphson->Update();

            _newtonRaphsons.push_back(newtonRaphson);
        }
    }

    void ParametrisationUpdate::_checkSettings()
    {

    }

    void ParametrisationUpdate::UpdateParametrisation()
    {

        //TODO: estimates areas here
        bool conv{true};
        for(auto newtonRaphson: _newtonRaphsons)
        {
            newtonRaphson->solve();
            conv = newtonRaphson->getConvergence();
            if(not conv)
                break;
        }
        MPI_Allreduce(MPI_IN_PLACE, &conv, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

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

        tensor<double,1>&  globFields = fill->cellFields;
        tensor<double,1> V = globFields(range(fill->idxCellField("X"),fill->idxCellField("Z")))/globFields(fill->idxCellField("A")) - globFields(range(fill->idxCellField("X0"),fill->idxCellField("Z0")))/globFields(fill->idxCellField("A0"));

        tensor<double,1>&  tissFields = fill->tissFields;

        double deltat    = tissFields(fill->idxTissField("deltat"));
        double pressure2 = globFields(fill->idxCellField("Paux"));
        
        int idx_x = fill->idxNodeField("x");
        int idx_z = fill->idxNodeField("z");
        int idx_vx = fill->idxNodeField("vx");
        int idx_vz = fill->idxNodeField("vz");
        int idx_x0 = fill->idxNodeField("x0");
        int idx_z0 = fill->idxNodeField("z0");

        //[2.1] Geometry in the configuration at previous time-step
        tensor<double,1>      x0 = bfs * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,2>     Dx0 = Dbfs.T() * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,1>  cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
        double              jac0 = sqrt(cross0*cross0);
        tensor<double,1> normal0 = cross0/jac0;
        double              x0n0 = x0*normal0;

        //[2.2] Geometry in current configuration
        tensor<double,1>      x = bfs * nborFields(all,range(idx_x,idx_z));
        tensor<double,2>     Dx = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        double               xn = x * normal;
        tensor<double,2> metric = Dx * Dx.T();

        tensor<double,1>      v = bfs * nborFields(all,range(idx_vx,idx_vz));

        //[2.4] First-order derivatives of geometric quantities wrt nodal positions
        tensor<double,4> dDx        = outer(Dbfs,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross     = dDx(all,all,1,all)*antisym3D*Dx(0,all) - dDx(all,all,0,all)*antisym3D*Dx(1,all);
        tensor<double,2> djac       = 1./jac * dcross * cross;
        tensor<double,4> dmetric    = dDx * Dx.T();
                        dmetric   += dmetric.transpose({0,1,3,2});
        tensor<double,3> dnormal    = dcross/jac - outer(djac/(jac*jac),cross);
        
        //[2.5] Second-order derivatives of geometric quantities wrt nodal positions
        tensor<double,5> ddcross    = (dDx(all,all,1,all)*antisym3D*dDx(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                        ddcross  += ddcross.transpose({2,3,0,1,4});
        tensor<double,4> ddjac      = 1./jac * (ddcross * cross + product(dcross,dcross,{{2,2}}) - outer(djac,djac));
        tensor<double,5> ddnormal   = ddcross/jac - outer(dcross,djac/(jac*jac)).transpose({0,1,3,4,2}) - outer(djac/(jac*jac),dcross) - outer(ddjac/(jac*jac),cross) + 2.0/(jac*jac*jac) * outer(outer(djac,djac),cross);
        tensor<double,6> ddmetric   = 2.0 * product(dDx,dDx,{{3,3}}).transpose({0,1,3,4,2,5});
        
        tensor<double,2>& rhs_n = fill->vec_n;
        tensor<double,1>& rhs_g = fill->vec_c;
        tensor<double,4>& A_nn  = fill->mat_nn;
        tensor<double,3>& A_ng  = fill->mat_nc;
        tensor<double,3>& A_gn  = fill->mat_cn;

        rhs_n += fill->w_sample * jac0 * outer(bfs,x-(x0+V+((v-V)*normal0)*normal0));
        A_nn  += fill->w_sample * jac0 * outer(outer(bfs,bfs),Identity(3)).transpose({0,2,1,3});
        
        rhs_n          += fill->w_sample * pressure2 * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
        rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jac0 * x0n0)/3.0);

        A_nn           += fill->w_sample * pressure2 * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
        A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
        A_gn = A_ng.transpose({2,0,1});
    }

    void ParametrisationUpdate::_arbLagEulUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill)
    {
        using namespace std;
        using namespace Tensor;
        
        //[1] INPUT
        int eNN    = fill->eNN;
        tensor<double,1>  bfs(fill->bfs[0].data(),eNN);
        tensor<double,2> Dbfs(fill->bfs[1].data(),eNN,2);
        // tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);

        tensor<double,2> nborFields   = fill->nborFields;

        tensor<double,1>&  globFields = fill->cellFields;
        tensor<double,1> V = globFields(range(fill->idxCellField("X"),fill->idxCellField("Z")))/globFields(fill->idxCellField("A")) - globFields(range(fill->idxCellField("X0"),fill->idxCellField("Z0")))/globFields(fill->idxCellField("A0"));
        
        tensor<double,1>&  tissFields = fill->tissFields;

        tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                                 {{0.0,0.0},{0.0,1.0}},
                                 {{0.0,1.0},{1.0,0.0}}};
        
        //FIXME: what to do with this?
        double deltat    = tissFields(fill->idxTissField("deltat"));
        double pressure2 = globFields(fill->idxCellField("Paux"));

        //FIXME: this should be read from the user's input
        int idx_x = fill->idxNodeField("x");
        int idx_z = fill->idxNodeField("z");
        int idx_vx = fill->idxNodeField("vx");
        int idx_vz = fill->idxNodeField("vz");
        int idx_x0 = fill->idxNodeField("x0");
        int idx_z0 = fill->idxNodeField("z0");
        int idx_xR = fill->idxNodeField("xR");
        int idx_zR = fill->idxNodeField("zR");

        //FIXME: we can't expect to have this information; we should compute it and store it here
        double A  = globFields(fill->idxCellField("A"));
        double AR = globFields(fill->idxCellField("AR"));

        double minStretch = _minStretch * A/AR;
        double maxStretch = _maxStretch * A/AR;

        //[2.1] Geometry in the configuration at previous time-step
        tensor<double,1>      x0 = bfs * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,2>     Dx0 = Dbfs.T() * nborFields(all,range(idx_x0,idx_z0));
        // tensor<double,2>    DDx0 = DDbfs.T() * nborFields(all,range(idx_x0,idx_z0));
        tensor<double,1>  cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
        double              jac0 = sqrt(cross0*cross0);
        tensor<double,1> normal0 = cross0/jac0;
        double              x0n0 = x0*normal0;
        tensor<double,2> metric0 = Dx0 * Dx0.T();
        tensor<double,2> imetric0 = metric0.inv();
        
        // tensor<double,2> curva0  = (DDx0 * normal0) * voigt;
        // double H0 = product(curva0,imetric0,{{0,0},{1,1}});
        // double K0 = curva0.det() * imetric0.det();
        // double c1_0 = H0 + sqrt(H0*H0-K0);
        // double c2_0 = H0 - sqrt(H0*H0-K0);

        tensor<double,2> n0n0 = outer(normal0,normal0);
        tensor<double,2> proj0 = Identity(3);
        proj0 -= n0n0;

        tensor<double,1>      v = bfs * nborFields(all,range(idx_vx,idx_vz));
        
        //[2.2] Geometry in current configuration
        tensor<double,1>      x = bfs * nborFields(all,range(idx_x,idx_z));
        tensor<double,2>     Dx = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
        tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
        double             jac  = sqrt(cross*cross);
        tensor<double,1> normal = cross/jac;
        double               xn = x * normal;
        tensor<double,2> metric = Dx * Dx.T();
        
        //[2.4] First-order derivatives of geometric quantities wrt nodal positions
        tensor<double,4> dDx        = outer(Dbfs,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross     = dDx(all,all,1,all)*antisym3D*Dx(0,all) - dDx(all,all,0,all)*antisym3D*Dx(1,all);
        tensor<double,2> djac       = 1./jac * dcross * cross;
        tensor<double,4> dmetric    = dDx * Dx.T();
                        dmetric   += dmetric.transpose({0,1,3,2});
        tensor<double,4> dmetric_C0C0 = product(product(dmetric,imetric0,{{2,0}}),imetric0,{{2,0}});
        tensor<double,3> dnormal    = dcross/jac - outer(djac/(jac*jac),cross);
        
        //[2.5] Second-order derivatives of geometric quantities wrt nodal positions
        tensor<double,5> ddcross    = (dDx(all,all,1,all)*antisym3D*dDx(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                        ddcross  += ddcross.transpose({2,3,0,1,4});
        tensor<double,4> ddjac      = 1./jac * (ddcross * cross + product(dcross,dcross,{{2,2}}) - outer(djac,djac));
        tensor<double,5> ddnormal   = ddcross/jac - outer(dcross,djac/(jac*jac)).transpose({0,1,3,4,2}) - outer(djac/(jac*jac),dcross) - outer(ddjac/(jac*jac),cross) + 2.0/(jac*jac*jac) * outer(outer(djac,djac),cross);
        tensor<double,6> ddmetric   = 2.0 * product(dDx,dDx,{{3,3}}).transpose({0,1,3,4,2,5});
        
        tensor<double,2>& rhs_n = fill->vec_n;
        tensor<double,1>& rhs_g = fill->vec_c;
        tensor<double,4>& A_nn  = fill->mat_nn;
        tensor<double,3>& A_ng  = fill->mat_nc;
        tensor<double,3>& A_gn  = fill->mat_cn;
        
        //Error in normal displacements (substracting rigid body motions, which we'll add later)
        rhs_n += fill->w_sample * jac0 * (((x-x0)-(v-V)) * normal0) * outer(bfs,normal0);
        A_nn  += fill->w_sample * jac0 * outer(outer(bfs,normal0),outer(bfs,normal0));
        
        tensor<double,2>     DxR = Dbfs.T() * nborFields(all,range(idx_xR,idx_zR));
        tensor<double,2> metricR = DxR * DxR.T();
        double jacR = sqrt(metricR.det());
        tensor<double,2> imetricR = metricR.inv();
        double J = jac/jacR;
        double I1 = product(imetricR, metric,{{0,0},{1,1}});
        double shear = 1.0-4.0*(J*J)/(I1*I1) > 0 ? sqrt(1.0-4.0*(J*J)/(I1*I1)) : 0.0;

        double shear_factor_rhs{0.0};
        double shear_factor_matrix{0.0};
        if(shear > _maxShear)
        {
            tensor<double,2> dI1 = product(imetricR,dmetric,{{0,2},{1,3}});
            tensor<double,4> ddI1 = product(imetricR,ddmetric,{{0,4},{1,5}});
            tensor<double,2> dshear = 4.0/shear * ((J*J)/(I1*I1*I1) * dI1 - (J/jacR/(I1*I1)) * djac);
            tensor<double,4> ddshear = 4.0/shear * (-3.0 * (J*J)/(I1*I1*I1*I1) * outer(dI1,dI1) + (J*J)/(I1*I1*I1) * ddI1 + 2.0 * (J/jacR/(I1*I1*I1)) * (outer(dI1,djac)+outer(djac,dI1)) - (1./(jacR*jacR)/(I1*I1)) * outer(djac,djac) - (J/jacR/(I1*I1)) * ddjac) - (1./shear) * outer(dshear,dshear);
            
            shear_factor_rhs = (shear-_maxShear) * (shear-_maxShear) / 3.0;
            shear_factor_matrix = 2.0 * (shear-_maxShear) / 3.0;
            rhs_n += fill->w_sample * _penaltyShear * jacR * shear_factor_rhs * (dshear * proj0); 
            A_nn  += fill->w_sample * _penaltyShear * jacR * shear_factor_rhs * product(ddshear, proj0,{{1,0}}).transpose({0,3,1,2});
            A_nn  += fill->w_sample * _penaltyShear * jacR * shear_factor_matrix * outer(dshear * proj0,dshear);
        }

        double jac_factor_rhs{0.0};
        double jac_factor_matrix{0.0};
        if(J < minStretch)
        {
            jac_factor_rhs = -10.0 * (minStretch-J) * (minStretch-J) / 3.0;
            jac_factor_matrix = 10.0 * 2.0 * (minStretch-J) / 3.0;
        }
        else if(J > maxStretch)
        {
            jac_factor_rhs = (J-maxStretch) * (J-maxStretch) / 3.0;
            jac_factor_matrix = 2.0 * (J-maxStretch) / 3.0;
        }
        rhs_n += fill->w_sample * _penaltyStretch * jac_factor_rhs * (djac * proj0);
        A_nn  += fill->w_sample * _penaltyStretch * jac_factor_rhs * product(ddjac, proj0, {{1,0}}).transpose({0,3,1,2});
        A_nn  += fill->w_sample * _penaltyStretch * jac_factor_matrix * outer((djac* proj0)/jacR,djac);

        //[2.3] Rate-of-deformation tensor
        tensor<double,2> rodt    = 0.5 * (metric-metric0);
        tensor<double,2> rodt_CC = imetric0 * rodt * imetric0;
        rhs_n += fill->w_sample * jac0 * _viscosity * product(product(dmetric,proj0,{{1,0}}),rodt_CC,{{1,0},{2,1}});
        A_nn  += fill->w_sample * jac0 * _viscosity * (0.5 * product(product(dmetric,proj0,{{1,0}}),dmetric_C0C0,{{1,2},{2,3}}) + product(product(ddmetric, proj0,{{1,0}}).transpose({0,5,1,2,3,4}),rodt_CC,{{4,0},{5,1}}));
        

        rhs_n          += fill->w_sample * pressure2 * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
        rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jac0 * x0n0)/3.0);

        A_nn           += fill->w_sample * pressure2 * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
        A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
        A_gn = A_ng.transpose({2,0,1});
    }

}