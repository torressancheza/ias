#ifndef _LoopSubdivision_h
#define _LoopSubdivision_h

#include "ias_BasisFunction.h"
#include "Tensor.h"

namespace ias
{
    /*! @class LoopSubdivision
     *  @brief Computes basis functions associated to a Loop Subdivision Surface. The algorithm employed here is adapted from
     *         Stam, J. (1999). Evaluation of Loop subdivision surfaces. SIGGRAPH’99 Course Notes. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.4347&rep=rep1&type=pdf */
    class LoopSubdivision : public BasisFunction
    {
        public:
                
            /** @name Constructor/destructor
            *  @{ */
            /*! @brief Constructor
             *  @param connec connectivity of the mesh */
            LoopSubdivision(Tensor::tensor<int,2> connec);

            /*! @brief Destructor */
            ~LoopSubdivision(){}
            /** @} */

            /** @name Copy/move constructors/assignments
            *  @{ */
            /*! @brief Copy contructor deleted   */
            LoopSubdivision(const LoopSubdivision&)             = delete;
            /*! @brief Copy assignment deleted */
            LoopSubdivision& operator=(const LoopSubdivision&)  = delete;
            /*! @brief Move constructor defaulted */
            LoopSubdivision(LoopSubdivision&&)                  = default;
            /*! @brief Copy constructor defaulted */
            LoopSubdivision& operator=(LoopSubdivision&&)       = default;
            /** @} */

            /** @name Output
            *  @{ */
            /*! @brief Compute basis functions of an element*/
            std::vector<std::vector<double>> computeBasisFunctions (std::vector<double> xi, int e) override
            {    return _etype[e]+4 == 6 ? _computeBoxSplines(xi) : _computeIrrBasisFunctions(xi, _etype[e]+4);   }
            /*! @brief Compute basis functions of a type*/
            std::vector<std::vector<double>> computeBasisFunctionsType (std::vector<double> xi, int type) override
            {   return type+4 == 6 ? _computeBoxSplines(xi) : _computeIrrBasisFunctions(xi, type+4);    };
            /*! @brief Get the parametric dimension*/
            int getParametricDimension() override
            {    return 2;    };
            /** @} */
        
        private:
            double _eps = 1.E-8;
            /*! @brief Compute basis functions on an irregular element characterised by valance eval*/
            std::vector<std::vector<double>> _computeIrrBasisFunctions (std::vector<double> xi, int eval);
            /*! @brief Compute basis functions on a regular element (val = 6)*/
            std::vector<std::vector<double>> _computeBoxSplines (std::vector<double> xi);
    };
}

#endif //_LoopSubdivision_h
