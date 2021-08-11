//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include "aux.h"

void leastSquaresTension(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;
    using Teuchos::RCP;
    //[1] INPUT
    int eNN    = fill->eNN;
    
    tensor<double,2>& nborFields = fill->nborFields;
    tensor<double,1>& cellFields = fill->cellFields;
    tensor<double,1>& tissFields = fill->tissFields;

    double tension   = cellFields(fill->idxCellField("tension"));

    tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
    tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);    
    double deltat    = tissFields(fill->idxTissField("deltat"));
    
        
    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    //[2.1] Geometry in the configuration at previous time-step
    tensor<double,1>       x0 = bfs * nborFields(all,range(idx_x,idx_z));
    tensor<double,2>      Dx0 = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
    tensor<double,1>   cross0 = Dx0(1,all) * antisym3D * Dx0(0,all);
    double               jac0 = sqrt(cross0*cross0);
    tensor<double,1>  normal0 = cross0/jac0;
    tensor<double,2>  metric0 = Dx0 * Dx0.T();
    tensor<double,2> imetric0 = metric0.inv();
    double               x0n0 = x0 * normal0;
    //[3] OUTPUT
    tensor<double,2>& rhs_n = fill->vec_n;
    tensor<double,4>& A_nn  = fill->mat_nn;

    // [3.1] Friction dissipation
    rhs_n(all,0) += fill->w_sample * jac0 * bfs * tension;
    A_nn (all,0,all,0) += fill->w_sample * jac0 * outer(bfs,bfs);
}

void leastSquaresTensionInteraction(Teuchos::RCP<ias::DoubleIntegralStr> inter)
{
    using namespace std;
    using namespace Tensor;

    tensor<double,1> globFields = inter->fillStr1->cellFields;
    
    double deltat    = inter->fillStr1->tissFields(inter->fillStr1->idxTissField("deltat"));

    double r0        = globFields(inter->fillStr1->idxCellField("intEL"));
    double w         = globFields(inter->fillStr1->idxCellField("intCL"));
    double D         = globFields(inter->fillStr1->idxCellField("intSt"));
    
    //First calculate distance to see if we can drop the points
    int eNN_1    = inter->fillStr1->eNN;

    tensor<double,2> nborFields_1   = inter->fillStr1->nborFields;
    tensor<double,1> bfs_1(inter->fillStr1->bfs[0].data(),eNN_1);
    
    int eNN_2    = inter->fillStr2->eNN;
    
    tensor<double,2> nborFields_2  = inter->fillStr2->nborFields;
    tensor<double,1>  bfs_2(inter->fillStr2->bfs[0].data(),eNN_2);

    tensor<double,1> x_1  = bfs_1 * nborFields_1(all,range(0,2));
    tensor<double,1> x_2  = bfs_2 * nborFields_2(all,range(0,2));
    double dist = sqrt((x_1-x_2)*(x_1-x_2));
    
    if(dist<3.0*w+r0)
    {
        //INPUT
        tensor<double,2> Dbfs_1(inter->fillStr1->bfs[1].data(),eNN_1,2);
        tensor<double,2> Dbfs_2(inter->fillStr2->bfs[1].data(),eNN_2,2);

        //OUTPUT
        tensor<double,2>& rhs_n_1 = inter->fillStr1->vec_n;
        tensor<double,2>& rhs_n_2 = inter->fillStr2->vec_n;
        
        tensor<double,2> Dx_1 = Dbfs_1.T() * nborFields_1(all,range(0,2));
        tensor<double,1> cross_1 = Dx_1(1,all) * antisym3D * Dx_1(0,all);
        double jac_1 = sqrt(cross_1*cross_1);

        tensor<double,2> Dx_2 = Dbfs_2.T() * nborFields_2(all,range(0,2));
        tensor<double,1> cross_2 = Dx_2(1,all) * antisym3D * Dx_2(0,all);
        double jac_2 = sqrt(cross_2*cross_2);

        double w1 = inter->fillStr1->w_sample;
        double w2 = inter->fillStr2->w_sample;
        double ww = w1*w2;
        
        double   pot =   ModMorsePotential(D,r0,w,r0,r0+3.0*w,dist);

        rhs_n_1(all,0) += ww * jac_1 * jac_2 * pot * bfs_1;
        rhs_n_2(all,0) += ww * jac_1 * jac_2 * pot * bfs_2;
    }
}

void calculateMoments(Teuchos::RCP<ias::SingleIntegralStr> fill)
{
    using namespace std;
    using namespace Tensor;

    //[1] INPUT
    int eNN    = fill->eNN;
    
    tensor<double,2>& nborFields = fill->nborFields;
    tensor<double,1>& cellFields = fill->cellFields;
    tensor<double,1>& tissFields = fill->tissFields;

    tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
    tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN,2);    
    
        
    int idx_x = fill->idxNodeField("x");
    int idx_z = fill->idxNodeField("z");
    //[2.1] Geometry in the configuration at previous time-step
    tensor<double,1>        x = bfs * nborFields(all,range(idx_x,idx_z));
    tensor<double,2>       Dx = Dbfs.T() * nborFields(all,range(idx_x,idx_z));
    tensor<double,1>    cross = Dx(1,all) * antisym3D * Dx(0,all);
    double                jac = sqrt(cross*cross);
    tensor<double,1>   normal = cross/jac;
    tensor<double,2>   metric = Dx * Dx.T();
    tensor<double,2>  imetric = metric.inv();
    double                 xn = x * normal;


    double w_sample = fill->w_sample;

    //This will not work in the first round. One needs to first run this and then rerun it
    double A = fill->cellFields(fill->idxCellField("M0"));
    double Xx = fill->cellFields(fill->idxCellField("M10"));
    double Xy = fill->cellFields(fill->idxCellField("M11"));
    double Xz = fill->cellFields(fill->idxCellField("M12"));
    if(A > 1.E-8 and (abs(Xx)>1.E-8 or abs(Xy) > 1.E-8 or abs(Xz) > 1.E-8))
    {
        tensor<double,1> X = {Xx/A,Xy/A,Xz/A};
        x -= X;
    }
    //Zeroth moment
    fill->cellIntegrals(fill->idxCellIntegral("M0")) += w_sample * jac;

    //First moment
    fill->cellIntegrals(fill->idxCellIntegral("M10")) += w_sample * jac * x(0);
    fill->cellIntegrals(fill->idxCellIntegral("M11")) += w_sample * jac * x(1);
    fill->cellIntegrals(fill->idxCellIntegral("M12")) += w_sample * jac * x(2);
    
    //Second moment
    fill->cellIntegrals(fill->idxCellIntegral("M20")) += w_sample * jac * (x(0) * x(0) - x(1) * x(1));
    fill->cellIntegrals(fill->idxCellIntegral("M21")) += w_sample * jac * (x(0) * x(0) - x(2) * x(2));
    fill->cellIntegrals(fill->idxCellIntegral("M22")) += w_sample * jac * x(0) * x(1);
    fill->cellIntegrals(fill->idxCellIntegral("M23")) += w_sample * jac * x(0) * x(2);
    fill->cellIntegrals(fill->idxCellIntegral("M24")) += w_sample * jac * x(1) * x(2);

    //Volume
    fill->cellIntegrals(fill->idxCellIntegral("V")) += w_sample * jac * (x*normal)/3.0;

}