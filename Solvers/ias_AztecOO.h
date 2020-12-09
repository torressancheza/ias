#ifndef _aztecoo_h
#define _aztecoo_h

#include <vector>
#include <string>

#include <AztecOO.h>


#include "ias_Solver.h"

namespace ias
{
    namespace solvers
    {
    /*! @class TrilinosAztecOO
     *  @brief Wrapper for the AztecOO solver from Trilinos (see manual for options:
     *         https://prod.sandia.gov/techlib-noauth/access-control.cgi/2004/043796.pdf )*/
        class TrilinosAztecOO : public Solver
        {
            public:

                /** @name Constructor/destructor
                *  @{ */
                /*! @brief Constructor */
                TrilinosAztecOO(){};
                /*! @brief Destructor */
                ~TrilinosAztecOO(){};
                /** @} */
            
                /** @name Copy/move constructors/assignments
                *  @{ */
                /*! @brief Copy contructor deleted   */
                TrilinosAztecOO(const TrilinosAztecOO&)             = delete;
                /*! @brief Copy assignment deleted */
                TrilinosAztecOO& operator=(const TrilinosAztecOO&)  = delete;
                /*! @brief Move constructor defaulted */
                TrilinosAztecOO(TrilinosAztecOO&&)                  = default;
                /*! @brief Copy constructor defaulted */
                TrilinosAztecOO& operator=(TrilinosAztecOO&&)       = default;
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
            
                /** @name Setters
                *  @{ */
                /*! @brief Set the maximum number of iterations allowed in the solver*/
                void addAztecOOParameter(std::string nameParam, std::string param)
                {
                    _nameParams.push_back(nameParam);
                    _params.push_back(param);
                }
                void DestroyPreconditioner()
                {    _solver->DestroyPreconditioner();    }
                /** @} */
            
            private:
                std::vector<std::string> _nameParams;
                std::vector<std::string> _params;
                Teuchos::RCP<AztecOO> _solver  = Teuchos::null;
        };
    }
}

#endif //_aztecoo_h
