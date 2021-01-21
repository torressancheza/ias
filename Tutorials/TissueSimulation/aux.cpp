//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include "aux.h"

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;
    using Teuchos::RCP;
    //[1] INPUT
    int eNN    = fill->eNN;
    
    tensor<double,2>& nborFields = fill->nborFields;
    tensor<double,1>& cellFields = fill->cellFields;
    tensor<double,1>& tissFields = fill->tissFields;

    double pressure = cellFields(fill->idxCellField("P"));

    tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
    tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);
    tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);
    
    double deltat    = tissFields(fill->idxTissField("deltat"));
    
    double tension   = cellFields(fill->idxCellField("tension")) * deltat;
    double kappa     = cellFields(fill->idxCellField("kappa")) * deltat;
    double viscosity = cellFields(fill->idxCellField("viscosity"));
    double frictiont = cellFields(fill->idxCellField("frictiont"));
    double frictionn = cellFields(fill->idxCellField("frictionn"));

    tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                              {{0.0,0.0},{0.0,1.0}},
                              {{0.0,1.0},{1.0,0.0}}};
    
    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    int idx_vx = fill->idxNodeField("vx");
    int idx_vz = fill->idxNodeField("vz");

    //[2] CALCULATIONS
    //[2.1] Geometry in the configuration at previous time-step
    tensor<double,1>       x0 = bfs * nborFields(all,range(idx_x,idx_z));
    tensor<double,2>      Dx0 = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
    tensor<double,1>   cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
    double               jac0 = sqrt(cross0*cross0);
    tensor<double,1>  normal0 = cross0/jac0;
    tensor<double,2>  metric0 = Dx0 * Dx0.T();
    tensor<double,2> imetric0 = metric0.inv();
    double               x0n0 = x0 * normal0;

    //[2.2] Geometry in current configuration
    tensor<double,1>      x =  bfs * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
    tensor<double,2>     Dx =  Dbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
    tensor<double,2>    DDx = DDbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
    tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
    double             jac  = sqrt(cross*cross);
    tensor<double,1> normal = cross/jac;
    double               xn = x * normal;
    tensor<double,2> metric = Dx * Dx.T();
    tensor<double,2> imetric = metric.inv();
    tensor<double,2> curva  = (DDx * normal) * voigt;
    double H = product(curva,imetric,{{0,0},{1,1}});
        
    tensor<double,1> v = bfs * nborFields(all,range(3,5));
    
    //[2.3] Rate-of-deformation tensor
    tensor<double,2> rodt    = 0.5 * (metric-metric0);
    tensor<double,2> rodt_CC = imetric0 * rodt * imetric0;
    
    //[2.4] First-order derivatives of geometric quantities wrt nodal positions
    tensor<double,4> dDx        = outer(Dbfs,Identity(3)).transpose({0,2,1,3});
    tensor<double,4> dDDx       = outer(DDbfs,Identity(3)).transpose({0,2,1,3});
    tensor<double,3> dcross     = dDx(all,all,1,all)*antisym3D*Dx(0,all) - dDx(all,all,0,all)*antisym3D*Dx(1,all);
    tensor<double,2> djac       = 1./jac * dcross * cross;
    tensor<double,4> dmetric    = dDx * Dx.T();
                     dmetric   += dmetric.transpose({0,1,3,2});
    tensor<double,4> dmetric_C0C0 = product(product(dmetric,imetric0,{{2,0}}),imetric0,{{2,0}});
    tensor<double,4> dmetric_CC = product(product(dmetric,imetric,{{2,0}}),imetric,{{2,0}});
    tensor<double,3> dnormal    = dcross/jac - outer(djac/(jac*jac),cross);
    tensor<double,4> dcurva     = (dDDx * normal) * voigt + (dnormal * DDx.T()) * voigt;
    tensor<double,2> dH         = product(dcurva,imetric,{{2,0},{3,1}}) - product(dmetric_CC,curva,{{2,0},{3,1}});
    
    //[2.5] Second-order derivatives of geometric quantities wrt nodal positions
    tensor<double,5> ddcross    = (dDx(all,all,1,all)*antisym3D*dDx(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                     ddcross  += ddcross.transpose({2,3,0,1,4});
    tensor<double,4> ddjac      = 1./jac * (ddcross * cross + product(dcross,dcross,{{2,2}}) - outer(djac,djac));
    tensor<double,5> ddnormal   = ddcross/jac - outer(dcross,djac/(jac*jac)).transpose({0,1,3,4,2}) - outer(djac/(jac*jac),dcross) - outer(ddjac/(jac*jac),cross) + 2.0/(jac*jac*jac) * outer(outer(djac,djac),cross);
    tensor<double,6> ddmetric   = 2.0 * product(dDx,dDx,{{3,3}}).transpose({0,1,3,4,2,5});
    tensor<double,6> ddimetric =  -product(product(ddmetric,imetric,{{4,0}}),imetric,{{4,0}}) + product(product(dmetric,dmetric_CC,{{2,2}}),imetric,{{2,0}}) + product(product(dmetric,imetric,{{2,0}}),dmetric_CC,{{2,2}}).transpose({0,1,3,4,2,5});
    tensor<double,6> ddcurva    = product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}) + product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}).transpose({2,3,0,1,4,5}) + (ddnormal * DDx.T()) * voigt;
    tensor<double,4> ddH = product(ddcurva,imetric,{{4,0},{5,1}}) - product(dcurva,dmetric_CC,{{2,2},{3,3}}) - product(dmetric_CC,dcurva,{{2,2},{3,3}}) + product(ddimetric,curva,{{4,0},{5,1}});
    
    //[3] OUTPUT
    tensor<double,2>& rhs_n = fill->vec_n;
    tensor<double,1>& rhs_g = fill->vec_c;
    tensor<double,4>& A_nn  = fill->mat_nn;
    tensor<double,3>& A_ng  = fill->mat_nc;
    tensor<double,3>& A_gn  = fill->mat_cn;
    
    tensor<double,2> n0n0 = outer(normal0,normal0);
    tensor<double,2> proj0 = Identity(3);
    proj0 -= n0n0;
    
    double vn = v*normal0;
    tensor<double,1> vt = proj0 * v;
    
    //Bending energy
    rhs_n += fill->w_sample *  kappa * H * (jac * dH + 0.5 * H * djac);
    A_nn  += fill->w_sample *  kappa * (jac * (H * ddH + outer(dH,dH)) + H * outer(dH,djac) + H * outer(djac,dH) + 0.5 * H * H * ddjac);

    // [3.1] Friction dissipation
    rhs_n += fill->w_sample * frictiont * jac0 * outer(bfs,proj0*v);
    A_nn  += fill->w_sample * frictiont * jac0 * outer(bfs,outer(bfs,proj0)).transpose({0,2,1,3});

    rhs_n += fill->w_sample * frictionn * jac0 * outer(bfs,vn*normal0);
    A_nn  += fill->w_sample * frictionn * jac0 * outer(bfs,outer(bfs,n0n0)).transpose({0,2,1,3});
    
    // [3.2] Shear (+ dilatational) dissipation
    rhs_n += fill->w_sample * viscosity * jac0 * product(dmetric,rodt_CC,{{2,0},{3,1}});
    A_nn  += fill->w_sample * viscosity * jac0 * (0.5 * product(dmetric,dmetric_C0C0,{{2,2},{3,3}}) + product(ddmetric,rodt_CC,{{4,0},{5,1}}));
    
    // [3.3] Active tension power
    rhs_n += fill->w_sample * tension *  djac;
    A_nn  += fill->w_sample * tension * ddjac;

//    rhs_n += fill->w_sample * jac0 * tension * product(dmetric,imetric0,{{2,0},{3,1}});
//    A_nn  += fill->w_sample * jac0 * tension * product(ddmetric,imetric0,{{4,0},{5,1}});

    // [3.4] Volume constraint
    rhs_n          += fill->w_sample * pressure * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jac0 * x0n0)/3.0);
        
    A_nn           += fill->w_sample * pressure * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
    A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    A_gn = A_ng.transpose({2,0,1});
    
    fill->cellIntegrals(fill->idxCellIntegral("A")) += fill->w_sample * jac;
    fill->cellIntegrals(fill->idxCellIntegral("X")) += fill->w_sample * jac * x(0);
    fill->cellIntegrals(fill->idxCellIntegral("Y")) += fill->w_sample * jac * x(1);
    fill->cellIntegrals(fill->idxCellIntegral("Z")) += fill->w_sample * jac * x(2);
    
    fill->cellIntegrals(fill->idxCellIntegral("A0")) += fill->w_sample * jac0;
    fill->cellIntegrals(fill->idxCellIntegral("X0")) += fill->w_sample * jac0 * x0(0);
    fill->cellIntegrals(fill->idxCellIntegral("Y0")) += fill->w_sample * jac0 * x0(1);
    fill->cellIntegrals(fill->idxCellIntegral("Z0")) += fill->w_sample * jac0 * x0(2);
    
//    fill->globIntegrals[0] += fill->w_sample * jac0 * 0.5 * (frictiont * vt*vt + frictionn * vn*vn) / deltat/ deltat;
//    fill->globIntegrals[1] += fill->w_sample * jac0 * viscosity * product(rodt_CC,rodt,{{0,0},{1,1}}) / deltat / deltat;
////    fill->globIntegrals[2] += fill->w_sample * (jac/jac0-1.0) * tension / deltat / deltat;
//    fill->globIntegrals[2] += fill->w_sample * jac * tension / deltat;
}

void interaction(Teuchos::RCP<ias::DoubleIntegralStr> inter)
{
    using namespace std;
    using namespace Tensor;

    tensor<double,1> globFields = inter->fillStr1->cellFields;
    
    double deltat    = inter->fillStr1->tissFields(inter->fillStr1->idxTissField("deltat"));

    double r0        = globFields(inter->fillStr1->idxCellField("intEL"));
    double w         = globFields(inter->fillStr1->idxCellField("intCL"));
    double D         = globFields(inter->fillStr1->idxCellField("intSt")) * deltat;
    
    //First calculate distance to see if we can drop the points
    int eNN_1    = inter->fillStr1->eNN;

    tensor<double,2> nborFields_1   = inter->fillStr1->nborFields;
    tensor<double,1> bfs_1(inter->fillStr1->bfs[0].data(),eNN_1);
    
    int eNN_2    = inter->fillStr2->eNN;
    
    tensor<double,2> nborFields_2  = inter->fillStr2->nborFields;
    tensor<double,1>  bfs_2(inter->fillStr2->bfs[0].data(),eNN_2);

    tensor<double,1> x_1  = bfs_1 * (nborFields_1(all,range(0,2))+nborFields_1(all,range(3,5)));
    tensor<double,1> x_2  = bfs_2 * (nborFields_2(all,range(0,2))+nborFields_2(all,range(3,5)));
    double dist = sqrt((x_1-x_2)*(x_1-x_2));
    
    if(dist<3.0*w+r0)
    {
        //INPUT
        tensor<double,2> Dbfs_1(inter->fillStr1->bfs[1].data(),eNN_1,2);
        tensor<double,2> Dbfs_2(inter->fillStr2->bfs[1].data(),eNN_2,2);

        //OUTPUT
        tensor<double,2>& rhs_n_1 = inter->fillStr1->vec_n;
        tensor<double,4>& A_nn_1  = inter->fillStr1->mat_nn;

        //OUTPUT
        tensor<double,2>& rhs_n_2 = inter->fillStr2->vec_n;
        tensor<double,4>& A_nn_2  = inter->fillStr2->mat_nn;

        tensor<double,4>& A_nn_12 = inter->mat_n1n2;
        tensor<double,4>& A_nn_21 = inter->mat_n2n1;
        
        tensor<double,2> Dx_1 = Dbfs_1.T() * (nborFields_1(all,range(0,2))+nborFields_1(all,range(3,5)));
        tensor<double,1> cross_1 = Dx_1(1,all) * antisym3D * Dx_1(0,all);
        double jac_1 = sqrt(cross_1*cross_1);

        tensor<double,2> Dx_2 = Dbfs_2.T() * (nborFields_2(all,range(0,2))+nborFields_2(all,range(3,5)));
        tensor<double,1> cross_2 = Dx_2(1,all) * antisym3D * Dx_2(0,all);
        double jac_2 = sqrt(cross_2*cross_2);

        double w1 = inter->fillStr1->w_sample;
        double w2 = inter->fillStr2->w_sample;
        double ww = w1*w2;
        
        double   pot =   ModMorsePotential(D,r0,w,r0,r0+3.0*w,dist);
        double  dpot =  dModMorsePotential(D,r0,w,r0,r0+3.0*w,dist);
        double ddpot = ddModMorsePotential(D,r0,w,r0,r0+3.0*w,dist);

        tensor<double,4> dDx_1        = outer(Dbfs_1,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross_1     = dDx_1(all,all,1,all)*antisym3D*Dx_1(0,all) - dDx_1(all,all,0,all)*antisym3D*Dx_1(1,all);
        tensor<double,2> djac_1       = 1./jac_1 * dcross_1 * cross_1;
        tensor<double,5> ddcross_1    = (dDx_1(all,all,1,all)*antisym3D*dDx_1(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                         ddcross_1   += ddcross_1.transpose({2,3,0,1,4});
        tensor<double,4> ddjac_1      = 1./jac_1 * (ddcross_1 * cross_1 + product(dcross_1,dcross_1,{{2,2}}) - outer(djac_1,djac_1));

        tensor<double,2> aux_1   = outer(bfs_1,x_1-x_2);
        
        tensor<double,4> dDx_2        = outer(Dbfs_2,Identity(3)).transpose({0,2,1,3});
        tensor<double,3> dcross_2     = dDx_2(all,all,1,all)*antisym3D*Dx_2(0,all) - dDx_2(all,all,0,all)*antisym3D*Dx_2(1,all);
        tensor<double,2> djac_2       = 1./jac_2 * dcross_2 * cross_2;
        tensor<double,5> ddcross_2    = (dDx_2(all,all,1,all)*antisym3D*dDx_2(all,all,0,all).transpose({2,0,1})).transpose({0,1,3,4,2});
                         ddcross_2   += ddcross_2.transpose({2,3,0,1,4});
        tensor<double,4> ddjac_2      = 1./jac_2 * (ddcross_2 * cross_2 + product(dcross_2,dcross_2,{{2,2}}) - outer(djac_2,djac_2));
        
        tensor<double,2> aux_2   = outer(bfs_2,x_2-x_1);

        rhs_n_1 += ww  * jac_2 * ( jac_1 * dpot/dist * aux_1 + pot * djac_1) ;
        rhs_n_2 += ww  * jac_1 * ( jac_2 * dpot/dist * aux_2 + pot * djac_2) ;

        //matrix
        A_nn_1  += ww * jac_2 * ( pot * ddjac_1 + dpot/dist * outer(aux_1,djac_1) + dpot/dist * outer(djac_1,aux_1) + jac_1 * ((ddpot-dpot/dist)/(dist*dist) * outer(aux_1,aux_1) + dpot/dist * outer(bfs_1,outer(bfs_1,Identity(3))).transpose({0,2,1,3}) ));
        A_nn_2  += ww * jac_1 * ( pot * ddjac_2 + dpot/dist * outer(aux_2,djac_2) + dpot/dist * outer(djac_2,aux_2) + jac_2 * ((ddpot-dpot/dist)/(dist*dist) * outer(aux_2,aux_2) + dpot/dist * outer(bfs_2,outer(bfs_2,Identity(3))).transpose({0,2,1,3}) ));
        
        A_nn_12 += ww * ( pot * outer(djac_1,djac_2) + jac_1 * dpot/dist * outer(aux_1,djac_2) + jac_2 * dpot/dist * outer(djac_1,aux_2) + jac_1 * jac_2 * ((ddpot-dpot/dist)/(dist*dist) * outer(aux_1,aux_2) - dpot/dist * outer(bfs_1,outer(bfs_2,Identity(3))).transpose({0,2,1,3}) ));
        A_nn_21 += ww * ( pot * outer(djac_2,djac_1) + jac_2 * dpot/dist * outer(aux_2,djac_1) + jac_1 * dpot/dist * outer(djac_2,aux_1) + jac_1 * jac_2 * ((ddpot-dpot/dist)/(dist*dist) * outer(aux_2,aux_1) - dpot/dist * outer(bfs_2,outer(bfs_1,Identity(3))).transpose({0,2,1,3}) ));
        
        inter->fillStr1->cellIntegrals(inter->fillStr1->idxCellIntegral("Ai")) += inter->fillStr1->w_sample * jac_1;
        inter->fillStr2->cellIntegrals(inter->fillStr2->idxCellIntegral("Ai")) += inter->fillStr2->w_sample * jac_2;
    }
}



void eulerianUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill)
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

    
//    rhs_n += fill->w_sample * jac0 * outer(bfs,x-(x0+v));
//    A_nn  += fill->w_sample * jac0 * outer(outer(bfs,bfs),Identity(3)).transpose({0,2,1,3});
//    
    
    rhs_n          += fill->w_sample * pressure2 * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jac0 * x0n0)/3.0);

    A_nn           += fill->w_sample * pressure2 * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
    A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    A_gn = A_ng.transpose({2,0,1});
}

void arbLagEulUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;
    
    //[1] INPUT
    int eNN    = fill->eNN;
    tensor<double,1>  bfs(fill->bfs[0].data(),eNN);
    tensor<double,2> Dbfs(fill->bfs[1].data(),eNN,2);
    tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);

    tensor<double,2> nborFields   = fill->nborFields;

    tensor<double,1>&  globFields = fill->cellFields;
    tensor<double,1> V = globFields(range(fill->idxCellField("X"),fill->idxCellField("Z")))/globFields(fill->idxCellField("A")) - globFields(range(fill->idxCellField("X0"),fill->idxCellField("Z0")))/globFields(fill->idxCellField("A0"));

//    V = 0.0; //FIXME: eliminate me
    
    tensor<double,1>&  tissFields = fill->tissFields;

    
    tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                              {{0.0,0.0},{0.0,1.0}},
                              {{0.0,1.0},{1.0,0.0}}};
    
    double deltat    = tissFields(fill->idxTissField("deltat"));
    double pressure2 = globFields(fill->idxCellField("Paux"));
    double aleStrength = 0.01 * deltat;

    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    int idx_vx = fill->idxNodeField("vx");
    int idx_vz = fill->idxNodeField("vz");
    int idx_x0 = fill->idxNodeField("x0");
    int idx_z0 = fill->idxNodeField("z0");
    
    //[2.1] Geometry in the configuration at previous time-step
    tensor<double,1>      x0 = bfs * nborFields(all,range(idx_x0,idx_z0));
    tensor<double,2>     Dx0 = Dbfs.T() * nborFields(all,range(idx_x0,idx_z0));
    tensor<double,2>    DDx0 = DDbfs.T() * nborFields(all,range(idx_x0,idx_z0));
    tensor<double,1>  cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
    double              jac0 = sqrt(cross0*cross0);
    tensor<double,1> normal0 = cross0/jac0;
    double              x0n0 = x0*normal0;
    tensor<double,2> metric0 = Dx0 * Dx0.T();
    tensor<double,2> imetric0 = metric0.inv();
    tensor<double,2> curva0  = (DDx0 * normal0) * voigt;
    double H0 = product(curva0,imetric0,{{0,0},{1,1}});
    double K0 = curva0.det() * imetric0.det();
    double c1_0 = H0 + sqrt(H0*H0-K0);
    double c2_0 = H0 - sqrt(H0*H0-K0);
    
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
    double strength_normal = 100.0;
    double error_n = fill->w_sample * jac0 * strength_normal * 0.5 * (((x-x0)-(v-V)) * normal0) * (((x-x0)-(v-V)) * normal0);
    rhs_n += fill->w_sample * jac0 * strength_normal * (((x-x0)-(v-V)) * normal0) * outer(bfs,normal0);
    A_nn  += fill->w_sample * jac0 * strength_normal * outer(outer(bfs,normal0),outer(bfs,normal0));
    
    //Distortion energy (try to get equilateral triangles everywhere)
    double tanh_steepness = 50.0;
    double min_shear = 0.5;

    tensor<double,2> n0n0 = outer(normal0,normal0);
    tensor<double,2> proj0 = Identity(3);
    proj0 -= n0n0;
    
    tensor<double,2> DxR = {{1.0,0.0},{0.5,sqrt(3)/2.}};
    tensor<double,2> metricR = DxR * DxR.T();
    double jacR = sqrt(metricR.det());
    tensor<double,2> imetricR = metricR.inv();
    double J = jac/jacR;
    double I1 = product(imetricR, metric,{{0,0},{1,1}});
    double shear = sqrt(1.0-4.0*(J*J)/(I1*I1));
    tensor<double,2> dI1 = product(imetricR,dmetric,{{0,2},{1,3}});
    tensor<double,4> ddI1 = product(imetricR,ddmetric,{{0,4},{1,5}});
    tensor<double,2> dshear = 4.0/shear * ((J*J)/(I1*I1*I1) * dI1 - (J/jacR/(I1*I1)) * djac);
    tensor<double,4> ddshear = 4.0/shear * (-3.0 * (J*J)/(I1*I1*I1*I1) * outer(dI1,dI1) + (J*J)/(I1*I1*I1) * ddI1 + 2.0 * (J/jacR/(I1*I1*I1)) * (outer(dI1,djac)+outer(djac,dI1)) - (1./(jacR*jacR)/(I1*I1)) * outer(djac,djac) - (J/jacR/(I1*I1)) * ddjac) - (1./shear) * outer(dshear,dshear);

    double auxtanh = tanh(tanh_steepness*(shear-min_shear));
    double energy = 0.25 * aleStrength * (auxtanh+1.0) * shear * shear;
    double aux_rhs = 0.5 * (auxtanh+1.0) * shear + 0.25 * shear * shear * (1.0-auxtanh*auxtanh) * tanh_steepness;
    rhs_n += fill->w_sample * aleStrength * aux_rhs * (dshear * proj0); //The projector makes sure that these forces only apply to the tangent displacements
    A_nn  += fill->w_sample * aleStrength * aux_rhs * product(ddshear, proj0,{{1,0}}).transpose({0,3,1,2});
    double aux_mat = 0.5 * (auxtanh+1.0) + shear * (1.0-auxtanh*auxtanh) * tanh_steepness - 0.5 * shear * shear * auxtanh * (1.0-auxtanh*auxtanh) * tanh_steepness * tanh_steepness;
    A_nn  += fill->w_sample * aleStrength * aux_mat * outer(dshear * proj0,dshear);

//    cout << aux_rhs << endl;
//    cout << shear << " " << auxtanh+1.0 << endl;
//    cout << shear << endl;
    // Shear (+ dilatational) dissipation
    
//    tensor<double,2> DxR_ = {{2.0,0.0},{0.0,2.0}};
//    tensor<double,2> metricR_ = DxR_ * DxR_.T();
//    double J_ = sqrt(metricR_.det())/jacR;
//    double I1_ = product(imetricR, metricR_,{{0,0},{1,1}});
//    double shear_ = sqrt(1.0-4.0*(J_*J_)/(I1_*I1_));
//    cout << shear_ << endl;
    
//    //[2.3] Rate-of-deformation tensor
//    tensor<double,2> rodt    = 0.5 * (metric-metric0);
//    tensor<double,2> rodt_CC = imetric0 * rodt * imetric0;
//    rhs_n += fill->w_sample * jac0 * product(product(dmetric,proj0,{{1,0}}),rodt_CC,{{1,0},{2,1}});
//    A_nn  += fill->w_sample * jac0 * (0.5 * product(product(dmetric,proj0,{{1,0}}),dmetric_C0C0,{{1,2},{2,3}}) + product(product(ddmetric, proj0,{{1,0}}).transpose({0,5,1,2,3,4}),rodt_CC,{{4,0},{5,1}}));
    
//    // [3.3] Active tension power
//    rhs_n += fill->w_sample * aleStrength * 0.01 * (c1_0*c1_0+c2_0*c2_0) * globFields(fill->idxCellField("A")) *  djac * proj0;
//    A_nn  += fill->w_sample * aleStrength * 0.01 * (c1_0*c1_0+c2_0*c2_0) * globFields(fill->idxCellField("A")) * product(ddjac,proj0,{{1,0}}).transpose({0,3,1,2});
    
    rhs_n += fill->w_sample * jac0 * outer(bfs,proj0*(x-x0));
    A_nn  += fill->w_sample * jac0 * outer(bfs,outer(bfs,proj0)).transpose({0,2,1,3});
    
    rhs_n          += fill->w_sample * pressure2 * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jac0 * x0n0)/3.0);

    A_nn           += fill->w_sample * pressure2 * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
    A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    A_gn = A_ng.transpose({2,0,1});
}
