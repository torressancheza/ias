//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <vector>
#include <iostream>

#include "ias_LinearElements.h"

namespace ias
{

    LinearElements::LinearElements(Tensor::tensor<int,2> connec)
    {
            int nDim = connec.shape()[1]-1;
            if(nDim<1 or nDim>2)
                throw std::runtime_error("The dimension must be 1 or 2 but " + std::to_string(nDim) + " was given");
            _nDim = nDim;
            
            _adjyEN_i.push_back(0);
            for(int e = 0; e < connec.shape()[0]; e++)
            {
                _adjyEN_i.push_back(_adjyEN_i[e]+connec.shape()[1]);
        
                for(int i = 0; i < connec.shape()[1]; i++)
                    _adjyEN_j.push_back(connec(e,i));
            }
            
            _types = {0};
            _etype.resize(connec.shape()[0]);
            std::fill(_etype.begin(),_etype.end(),0);
    }

    std::vector<std::vector<double>>  LinearElements::computeBasisFunctions (std::vector<double> u, int e)
    {
        using namespace std;
        
        e = 0;
        vector<std::vector<double>> ders;
        
        ders.resize(2);
        ders[0].resize(_nDim+1);
        ders[1].resize((_nDim+1)*_nDim);

        // Compute the basis functions for each case
        switch ( _nDim )
        {

            case 1:
                
                ders[0][0] = (1.0-u[0]);
                ders[0][1] = u[0];

                ders[1][0] = -1.0;
                ders[1][1] =  1.0;

                break;
            case 2:
                
                ders[0][0] = 1.0-u[0]-u[1];
                ders[0][1] = u[0];
                ders[0][2] = u[1];
                
                ders[1][0] = -1.0;
                ders[1][1] = -1.0;
                ders[1][2] =  1.0;
                ders[1][3] =  0.0;
                ders[1][4] =  0.0;
                ders[1][5] =  1.0;
                
                break;
            default:
                throw runtime_error("LinearElements::ComputeBasisFunctions: dimension different from 1 or 2.");
        }
        
        return ders;
    }
}
