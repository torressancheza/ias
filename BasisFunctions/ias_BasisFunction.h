//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _BasisFunctions_h
#define _BasisFunctions_h

#include <vector>
#include <iostream>

namespace ias
{
    /*! @class BasisFunction
     *  @brief Virtual class for general Basis functions */
    class BasisFunction
    {
        public:
            BasisFunction(){};
            virtual ~BasisFunction(){};
            
            /** @name Output
            *  @{ */
            /*! @brief Compute basis functions of an element*/
            virtual std::vector<std::vector<double>> computeBasisFunctions (std::vector<double> u, int e) = 0;
            /*! @brief Compute basis functions of a type*/
            virtual std::vector<std::vector<double>> computeBasisFunctionsType (std::vector<double> xi, int type) = 0;
            
            /*! @brief Get number of neighbours of a given element*/
            int getNumberOfNeighbours(int e)
            {    return _adjyEN_i[e+1]-_adjyEN_i[e];    }
            /*! @brief Pointer to the list of neighbours of element e*/
            int* getNeighbours(int e)
            {    return &_adjyEN_j[_adjyEN_i[e]];    }
            int getMaxNumberOfNeighbours()
            {
                int max = 0;
                for(size_t e = 0; e <_adjyEN_i.size()-1; e++)
                {
                    int eNN = getNumberOfNeighbours(e);
                    if(max < eNN) max = eNN;
                }
                return max;
            }
            /*! @brief Get a vector with all types of basis functions*/
            std::vector<int>& getTypes()
            {    return _types;    }
            /*! @brief Get the type of a given element*/
            int getElementType(int e)
            {    return _etype[e];    }
            /*! @brief Get the parametric dimension*/
            virtual int getParametricDimension() = 0;
            /** @} */

        protected:
            std::vector<int> _adjyEN_i; ///< _adjyEN_i[e] indices the element in _adjyEN_j where neighbours of element e start
            std::vector<int> _adjyEN_j; ///<List with neighbours for each element stored consecutively for each element. A neighbour of element e is a node whose associated function is not zero at e.
            
            std::vector<int> _types; ///<Basis function types, e.g. one type for linear elements, as many as valences in Loop Subdivision.
            std::vector<int> _etype; ///<Type for each element
    };

    /*! @class BasisFunctionType
     *  @brief To define the different basis function types available in the library */
    enum class BasisFunctionType
    {
        Undefined = 0,
        Linear = 1,
        LoopSubdivision = 2
    };
}
#endif //_BasisFunctions_h
