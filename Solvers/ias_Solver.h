#ifndef _solvers_h
#define _solvers_h


#include <Epetra_LinearProblem.h>
#include "ias_Integration.h"

namespace ias
{
    namespace solvers
    {
        class Solver
        {
            public:
                /** @name Constructor/destructor
                *  @{ */
                /*! @brief Constructor */
                Solver(){};
                /*! @brief Destructor */
                ~Solver(){};
                /** @} */
                
                /** @name Setters (prior to Update())
                *  @{ */
                /*! @brief Set Integration object*/
                void setIntegration(Teuchos::RCP<Integration> integration)
                {    _integration = integration;    }
                /*! @brief Set the maximum number of iterations allowed in the solver*/
                void setMaximumNumberOfIterations(int maxIter)
                {    _maxIter = maxIter;    }
                /*! @brief Set the tolerance in the solution*/
                void setResidueTolerance(double resTol)
                {    _resTol = resTol;    }
                /** @} */
            
                /** @name Update
                *  @{ */
                /*! @brief Use the information given by the user through setters to initialise (or resize) internal variables */
                void Update()
                {
                    if(_maxIter == 0)
                        throw std::runtime_error("The maximum number of iterations is 0. Have you set it?");
                    
                    if(_resTol < 0)
                        throw std::runtime_error("The tolerance of the residual is negative. Have you set it?");
                    
                    if(_integration == Teuchos::null)
                        throw std::runtime_error("The integration object has not been set.");
                };
                /** @} */
            
                /** @name Solve
                *  @{ */
                /*! @brief Solve the system */
                virtual void solve() = 0;
                /** @} */
            
                /** @name Getters
                *  @{ */
                /*! @brief Get convergence status */
                bool getConvergence()
                {    return _converged;    }
                int getNumberOfIterations()
                {    return _nIter;    }
                Teuchos::RCP<Integration> getIntegration()
                {    return _integration;    }
                /** @} */

            protected:
                Teuchos::RCP<Integration> _integration = Teuchos::null; ///<Integration object. This is used to fill the linear system
                double _resTol{-1.0}; ///<Solver tolerance
                int    _maxIter{}; ///<Maximum number of iterations
            
                bool   _converged{}; ///<Convergence?
                int    _nIter{}; //Number of iterations
        };
    }
}

#endif //_solvers_h
