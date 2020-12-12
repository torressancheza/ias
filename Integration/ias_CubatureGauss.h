//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _CubatureGauss_h
#define _CubatureGauss_h

#include <vector>

namespace ias
{

    /*!
     * \brief Implementation of the gaussian cubature at any type of mesh.
    */
    class CubatureGauss
    {
        public:

            CubatureGauss(int iPts, int dim);
            ~CubatureGauss() = default;
        
            std::vector<std::vector<double>>& getXsamples()
            {
                return _xSamples;
            }

            std::vector<double>& getWsamples()
            {
                return _wSamples;
            }

        private:
        
            int _iPts;
            int _pDim;
        
            void _checkSettings();

            //Cubature rules
            void _lineRule();
            void _quadratureRule();
            void _triangularRule();
                
            std::vector<std::vector<double>> _xSamples;
            std::vector<double> _wSamples;

    };
}

#endif // _CubatureGauss_h
