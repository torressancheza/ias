#ifndef _BasicStructures_h
#define _BasicStructures_h

#include "Tensor.h"

namespace ias
{
//    struct AuxIntegralStr
//    {
//        std::vector<double> dparam;
//        std::vector<std::vector<double>> dparam_cell;
//    };

    struct SingleIntegralStr
    {
        int     pDim;
        int     elemID;
        int     sampID;
        int     eNN;
        int     cellID;
        double  w_sample;
        
        Tensor::tensor<double,2> nborFields{};
        Tensor::tensor<double,1> globFields{};

        std::vector<std::vector<double>> bfs{};

        std::vector<double> globIntegrals{};
        std::vector<double> cellIntegrals{};

        Tensor::tensor<double,2> vec_n{};
        Tensor::tensor<double,1> vec_g{};

        Tensor::tensor<double,4> mat_nn{};
        Tensor::tensor<double,3> mat_ng{};
        Tensor::tensor<double,3> mat_gn{};
        Tensor::tensor<double,2> mat_gg{};

    };

    struct DoubleIntegralStr
    {
        Teuchos::RCP<SingleIntegralStr> fillStr1{Teuchos::null};
        Teuchos::RCP<SingleIntegralStr> fillStr2{Teuchos::null};
        
        //Interactions
        Tensor::tensor<double,4> mat_n1n2{};
        Tensor::tensor<double,4> mat_n2n1{};
        
        Tensor::tensor<double,3> mat_n2g1{};
        Tensor::tensor<double,3> mat_n1g2{};
        Tensor::tensor<double,3> mat_g2n1{};
        Tensor::tensor<double,3> mat_g1n2{};
        
        Tensor::tensor<double,2> mat_g2g1{};
        Tensor::tensor<double,2> mat_g1g2{};
    };
}

#endif  // _BasicStructures_h
