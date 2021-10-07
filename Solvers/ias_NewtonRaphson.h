//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _NewtonRaphson_h
#define _NewtonRaphson_h

#include "ias_Integration.h"
#include "ias_Solver.h"

namespace ias
{
    namespace solvers
    {
        class NewtonRaphson : public Solver
        {
            public:

                /** @name Constructor/destructor
                *  @{ */
                /*! @brief Constructor */
                NewtonRaphson(){};
                /*! @brief Destructor */
                ~NewtonRaphson(){};
                /** @} */
            
                /** @name Copy/move constructors/assignments
                *  @{ */
                /*! @brief Copy contructor deleted   */
                NewtonRaphson(const NewtonRaphson&)             = delete;
                /*! @brief Copy assignment deleted */
                NewtonRaphson& operator=(const NewtonRaphson&)  = delete;
                /*! @brief Move constructor defaulted */
                NewtonRaphson(NewtonRaphson&&)                  = default;
                /*! @brief Copy constructor defaulted */
                NewtonRaphson& operator=(NewtonRaphson&&)       = default;
                /** @} */
            
                /** @name Update
                *  @{ */
                /*! @brief Use the information given by the user through setters to initialise (or resize) internal variables */
                void Update();
                /** @} */
            
                /** @name Solve
                *  @{ */
                /*! @brief Solve the system */
                void solve();
                /** @} */
            
                /** @name Setters (prior to Update())
                *  @{ */
                /*! @brief Set the maximum number of iterations allowed in the solver*/
                void setLinearSolver(Teuchos::RCP<Solver> linearSolver)
                {    _linearSolver = linearSolver;    }
                /*! @brief Set the maximum number of iterations allowed in the solver*/
                void setSolutionTolerance(double solTol)
                {    _solTol = solTol;    }
                /*! @brief Set if the convergence is relative to the first iteration*/
                void setConvergenceToRelativeTolerance(bool convRel)
                {    _convRel = convRel;    }
                /*! @brief Set if the solver should be verbose*/
                void setVerbosity(bool verbose)
                {    _verbose = verbose;    }
                /** @} */

                /*! @brief Get residuals*/
                std::vector<double> getResiduals()
                {    return _ressError;    }

                /*! @brief Get residuals*/
                Teuchos::RCP<Solver> getLinearSolver()
                {    return _linearSolver;    }
            
            private:
                Teuchos::RCP<Solver> _linearSolver = Teuchos::null; ///<linear solver
                double _solTol{};
                bool   _convRel{};
                bool   _verbose{};
                std::vector<double> _ressError;
        };
    }
}

#endif //_NewtonRaphson_h
