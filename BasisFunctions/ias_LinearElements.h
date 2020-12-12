//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _LinearElements_h
#define _LinearElements_h

#include <vector>
#include <string>

#include "Tensor.h"
#include "ias_BasisFunction.h"

namespace ias
{
    /*! @class LinearElements
     *  @brief Computes linear basis functions for a lline or triangle*/
    class LinearElements : public BasisFunction
    {
        public:
                
            /** @name Constructor/destructor
            *  @{ */
            /*! @brief Constructor
             *  @param nDim dimension of the simplex (either 1 or 2) */
            LinearElements(Tensor::tensor<int,2> connec);
            /*! @brief Destructor */
            ~LinearElements(){}
            /** @} */

            /** @name Copy/move constructors/assignments
            *  @{ */
            /*! @brief Copy contructor deleted   */
            LinearElements(const LinearElements&)             = delete;
            /*! @brief Copy assignment deleted */
            LinearElements& operator=(const LinearElements&)  = delete;
            /*! @brief Move constructor defaulted */
            LinearElements(LinearElements&&)                  = default;
            /*! @brief Copy constructor defaulted */
            LinearElements& operator=(LinearElements&&)       = default;
            /** @} */


            /** @name Output
            *  @{ */
            /*! @brief Compute basis functions of an element*/
            std::vector<std::vector<double>> computeBasisFunctions (std::vector<double> u, int e) override;
            /*! @brief Compute basis functions of a type*/
            std::vector<std::vector<double>> computeBasisFunctionsType (std::vector<double> u, int type) override
            {
                assert(type == 0);
                return computeBasisFunctions(u,0);
            };
            /*! @brief Get the parametric dimension*/
            int getParametricDimension() override
            {    return _nDim;    };

            /** @} */

        private:
            int _nDim; ///<Dimension of the simplex (1 or 2)
    };
}
#endif // _LinearElements_h
