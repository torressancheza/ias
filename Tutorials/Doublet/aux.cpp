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
    double deltat    = tissFields(fill->idxTissField("deltat"));
    
    double tension   = cellFields(fill->idxCellField("tension")) * deltat;
    double viscosity = cellFields(fill->idxCellField("viscosity"));
    double frictiont = cellFields(fill->idxCellField("frictiont"));
    double frictionn = cellFields(fill->idxCellField("frictionn"));
    
    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    int idx_xR = fill->idxNodeField("xR");
    int idx_zR = fill->idxNodeField("zR");
    int idx_vx = fill->idxNodeField("vx");
    int idx_vz = fill->idxNodeField("vz");

    //[2] CALCULATIONS
    tensor<double,1>       xR = bfs * nborFields(all,range(idx_xR,idx_zR));
    tensor<double,2>      DxR = Dbfs.T() * nborFields(all,range(idx_xR,idx_zR));
    tensor<double,1>   crossR = DxR(1,all) * antisym3D * DxR(0,all);
    double               jacR = sqrt(crossR*crossR);
    tensor<double,1>  normalR = crossR/jacR;
    double               xRnR = xR * normalR;

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
    tensor<double,1>  cross = Dx(1,all) * antisym3D * Dx(0,all);
    double             jac  = sqrt(cross*cross);
    tensor<double,1> normal = cross/jac;
    double               xn = x * normal;
    tensor<double,2> metric = Dx * Dx.T();
    tensor<double,2> imetric = metric.inv();
        
    tensor<double,1> v = bfs * nborFields(all,range(idx_vx,idx_vz));
    
    //[2.3] Rate-of-deformation tensor
    tensor<double,2> rodt    = 0.5 * (metric-metric0);
    tensor<double,2> rodt_CC = imetric0 * rodt * imetric0;
    
    //[2.4] First-order derivatives of geometric quantities wrt nodal positions
    tensor<double,4> dDx        = outer(Dbfs,Identity(3)).transpose({0,2,1,3});
    tensor<double,3> dcross     = dDx(all,all,1,all)*antisym3D*Dx(0,all) - dDx(all,all,0,all)*antisym3D*Dx(1,all);
    tensor<double,2> djac       = 1./jac * dcross * cross;
    tensor<double,4> dmetric    = dDx * Dx.T();
                     dmetric   += dmetric.transpose({0,1,3,2});
    tensor<double,4> dmetric_C0C0 = product(product(dmetric,imetric0,{{2,0}}),imetric0,{{2,0}});
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
    
    tensor<double,2> n0n0 = outer(normal0,normal0);
    tensor<double,2> proj0 = Identity(3);
    proj0 -= n0n0;
    
    double vn = v*normal0;
    tensor<double,1> vt = proj0 * v;

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

    // [3.4] Volume constraint
    rhs_n          += fill->w_sample * pressure * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    rhs_g(0)       += deltat * (fill->w_sample * (jac * xn-jacR * xRnR)/3.0);
        
    A_nn           += fill->w_sample * pressure * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
    A_ng(all,all,0) += fill->w_sample            * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    A_gn = A_ng.transpose({2,0,1});
    
    // [3.5] Bending energy (only if higher order)
    bool higher_order = fill->bfs.size()>2;
    if(higher_order)
    {
        tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                                {{0.0,0.0},{0.0,1.0}},
                                {{0.0,1.0},{1.0,0.0}}};

        double kappa     = cellFields(fill->idxCellField("kappa")) * deltat;
        tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);
        tensor<double,2>    DDx = DDbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
        tensor<double,2> curva  = (DDx * normal) * voigt;
        double H = product(curva,imetric,{{0,0},{1,1}});
        tensor<double,4> dDDx       = outer(DDbfs,Identity(3)).transpose({0,2,1,3});

        tensor<double,4> dcurva     = (dDDx * normal) * voigt + (dnormal * DDx.T()) * voigt;
        tensor<double,2> dH         = product(dcurva,imetric,{{2,0},{3,1}}) - product(dmetric_CC,curva,{{2,0},{3,1}});
        
        tensor<double,6> ddimetric =  -product(product(ddmetric,imetric,{{4,0}}),imetric,{{4,0}}) + product(product(dmetric,dmetric_CC,{{2,2}}),imetric,{{2,0}}) + product(product(dmetric,imetric,{{2,0}}),dmetric_CC,{{2,2}}).transpose({0,1,3,4,2,5});
        tensor<double,6> ddcurva    = product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}) + product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}).transpose({2,3,0,1,4,5}) + (ddnormal * DDx.T()) * voigt;
        tensor<double,4> ddH = product(ddcurva,imetric,{{4,0},{5,1}}) - product(dcurva,dmetric_CC,{{2,2},{3,3}}) - product(dmetric_CC,dcurva,{{2,2},{3,3}}) + product(ddimetric,curva,{{4,0},{5,1}});
        
        rhs_n += fill->w_sample *  kappa * H * (jac * dH + 0.5 * H * djac);
        A_nn  += fill->w_sample *  kappa * (jac * (H * ddH + outer(dH,dH)) + H * outer(dH,djac) + H * outer(djac,dH) + 0.5 * H * H * ddjac);
    }
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

        inter->fillStr1->tissIntegrals(inter->fillStr1->idxTissIntegral("Ei")) += ww * jac_1 * jac_2;
    }
}