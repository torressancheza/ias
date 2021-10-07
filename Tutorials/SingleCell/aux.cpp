//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <cmath>
#include "aux.h"

void leastSquares(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;
    using Teuchos::RCP;
    //[1] INPUT
    int eNN    = fill->eNN;
    
    tensor<double,2>& nborFields = fill->nborFields;
    tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
    tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);   

    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    tensor<double,1>       x0 = bfs * nborFields(all,range(idx_x,idx_z));
    tensor<double,2>      Dx0 = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
    tensor<double,1>   cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
    double               jac0 = sqrt(cross0*cross0);
    tensor<double,2>  metric0 = Dx0 * Dx0.T();
    tensor<double,2> imetric0 = metric0.inv();

    double tension{};
    double phi{};
    double vn{};
    computeFieldsSpherHarm(x0, tension, phi, vn);

    //[3] OUTPUT
    tensor<double,2>& rhs_n = fill->vec_n;
    tensor<double,4>& A_nn  = fill->mat_nn; 

    rhs_n(all,range(0,2)) += fill->w_sample * jac0 * outer(bfs,x0/sqrt(x0*x0));
    A_nn(all,range(0,2),all,range(0,2))  += fill->w_sample * jac0 * outer(bfs,outer(bfs,Identity(3))).transpose({0,2,1,3});

    rhs_n(all,3) += fill->w_sample * jac0 * bfs * tension;
    A_nn(all,3,all,3)  += fill->w_sample * jac0 * outer(bfs,bfs);
    rhs_n(all,4) += fill->w_sample * jac0 * bfs * phi;
    A_nn(all,4,all,4)  += fill->w_sample * jac0 * outer(bfs,bfs);
    rhs_n(all,5) += fill->w_sample * jac0 * bfs * vn;
    A_nn(all,5,all,5)  += fill->w_sample * jac0 * outer(bfs,bfs);

    rhs_n(all,range(6,8)) += fill->w_sample * jac0 * phi * (Dbfs * imetric0 * Dx0);
    A_nn(all,range(6,8),all,range(6,8))  += fill->w_sample * jac0 * outer(bfs,outer(bfs,Identity(3))).transpose({0,2,1,3});

}

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;
    using Teuchos::RCP;
    //[1] INPUT
    int eNN    = fill->eNN;
    
    tensor<double,2>& nborFields = fill->nborFields;
    tensor<double,1>& tissFields = fill->tissFields;

    tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
    tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);    
    double deltat    = tissFields(fill->idxTissField("deltat"));
    
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
    tensor<double,4>& A_nn  = fill->mat_nn;

    // [3.1] Friction dissipation
    rhs_n += fill->w_sample * friction * jac0 * outer(bfs,v);
    A_nn  += fill->w_sample * friction * jac0 * outer(bfs,outer(bfs,Identity(3))).transpose({0,2,1,3});

    // [3.2] Shear (+ dilatational) dissipation
    rhs_n += fill->w_sample * jac0 * 0.5 * product(dmetric,rodt_CC,{{2,0},{3,1}});
    A_nn  += fill->w_sample * jac0 * 0.5 * (0.5 * product(dmetric,dmetric_C0C0,{{2,2},{3,3}}) + product(ddmetric,rodt_CC,{{4,0},{5,1}}));
    
    double tension{};
    double phi{};
    double vn{};
    computeFieldsSpherHarm(x0, tension, phi, vn);

    rhs_n += fill->w_sample * tension * deltat *  djac;
    A_nn  += fill->w_sample * tension * deltat * ddjac;

    // [3.4] Volume constraint
    rhs_n          -= fill->w_sample * pressure * deltat / 3.0 * (djac * xn + jac * outer(bfs,normal) + jac * dnormal * x);
    A_nn           -= fill->w_sample * pressure * deltat / 3.0 * (ddjac * xn + outer(djac,outer(bfs,normal)) + outer(outer(bfs,normal),djac) + outer(djac,dnormal*x) + outer(dnormal*x,djac) + jac * outer(dnormal,bfs).transpose({0,1,3,2}) +  jac * outer(bfs,dnormal).transpose({0,3,1,2}) + jac * ddnormal * x);
    
    // [3.5] Bending energy (only if higher order)
    bool higher_order = fill->bfs.size()>2;
    if(higher_order)
    {
        tensor<double,3> voigt = {{{1.0,0.0},{0.0,0.0}},
                                  {{0.0,0.0},{0.0,1.0}},
                                  {{0.0,1.0},{1.0,0.0}}};

        tensor<double,2> DDbfs(fill->bfs[2].data(),eNN,3);
        tensor<double,2>    DDx = DDbfs.T() * (nborFields(all,range(idx_x,idx_z))+nborFields(all,range(idx_vx,idx_vz)));
        tensor<double,2> curva  = -(DDx * normal) * voigt;
        double H = product(curva,imetric,{{0,0},{1,1}});
        tensor<double,4> dDDx   = outer(DDbfs,Identity(3)).transpose({0,2,1,3});

        tensor<double,4> dcurva     = -((dDDx * normal) * voigt + (dnormal * DDx.T()) * voigt);
        tensor<double,2> dH         = product(dcurva,imetric,{{2,0},{3,1}}) - product(dmetric_CC,curva,{{2,0},{3,1}});
        
        tensor<double,6> ddimetric =  -product(product(ddmetric,imetric,{{4,0}}),imetric,{{4,0}}) + product(product(dmetric,dmetric_CC,{{2,2}}),imetric,{{2,0}}) + product(product(dmetric,imetric,{{2,0}}),dmetric_CC,{{2,2}}).transpose({0,1,3,4,2,5});
        tensor<double,6> ddcurva    = -(product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}) + product(product(dDDx, dnormal,{{3,2}}),voigt,{{2,0}}).transpose({2,3,0,1,4,5}) + (ddnormal * DDx.T()) * voigt);
        tensor<double,4> ddH = product(ddcurva,imetric,{{4,0},{5,1}}) - product(dcurva,dmetric_CC,{{2,2},{3,3}}) - product(dmetric_CC,dcurva,{{2,2},{3,3}}) + product(ddimetric,curva,{{4,0},{5,1}});
        
        rhs_n += fill->w_sample * kappa * deltat * (H-C0) * (jac * dH + 0.5 * (H-C0) * djac);
        A_nn  += fill->w_sample * kappa * deltat * (jac * ((H-C0) * ddH + outer(dH,dH)) + (H-C0) * outer(dH,djac) + (H-C0) * outer(djac,dH) + 0.5 * (H-C0) * (H-C0) * ddjac);
    }


    fill->tissIntegrals(fill->idxTissIntegral("error")) += fill->w_sample * jac * ((v*normal)/deltat-vn)*((v*normal)/deltat-vn);
}
