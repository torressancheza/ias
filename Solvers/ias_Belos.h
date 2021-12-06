//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _belos_h
#define _belos_h

#include <vector>
#include <string>

#include <Epetra_LinearProblem.h>
#include <EpetraExt_readEpetraLinearSystem.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>

#include <Ifpack_Preconditioner.h>
#include <Ifpack.h>

#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

#include "ias_Solver.h"

namespace ias
{
    namespace solvers
    {
    /*! @class Belos
     *  @brief Wrapper for the Belos solver from Trilinos */
        class TrilinosBelos : public Solver
        {
            public:

                /** @name Constructor/destructor
                *  @{ */
                /*! @brief Constructor */
                TrilinosBelos(){};
                /*! @brief Destructor */
                ~TrilinosBelos(){};
                /** @} */
            
                /** @name Copy/move constructors/assignments
                *  @{ */
                /*! @brief Copy contructor deleted   */
                TrilinosBelos(const TrilinosBelos&)             = delete;
                /*! @brief Copy assignment deleted */
                TrilinosBelos& operator=(const TrilinosBelos&)  = delete;
                /*! @brief Move constructor defaulted */
                TrilinosBelos(TrilinosBelos&&)                  = default;
                /*! @brief Copy constructor defaulted */
                TrilinosBelos& operator=(TrilinosBelos&&)       = default;
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
                void addBelosParameter(std::string nameParam, std::string param)
                {
                    _nameParams.push_back(nameParam);
                    _params.push_back(param);
                }
                void addIfpackParameter(std::string nameParam, std::string param)
                {
                    _ifpackNameParams.push_back(nameParam);
                    _ifpackParams.push_back(param);
                }
                /*! @brief Set the solver type*/
                void setSolverType(std::string solverType)
                {    _solverType = solverType;    }

                /*! @brief Set the preconditioner type*/
                void setPrecondType(std::string precondType)
                {    _precondType = precondType;    }

                void recalculatePreconditioner()
                {    _recPrec = true;}
                /** @} */
            
            private:
                std::vector<std::string> _nameParams;
                std::vector<std::string> _params;
                std::vector<std::string> _ifpackNameParams;
                std::vector<std::string> _ifpackParams;
                std::string _solverType = "GMRES";
                std::string _precondType = "ILU";

                typedef double                ST;
                typedef Epetra_MultiVector    MV;
                typedef Epetra_Operator       OP;

                Teuchos::RCP<Ifpack_Preconditioner> _precond = Teuchos::null;
                Teuchos::RCP<Belos::EpetraPrecOp> _belosPrecond = Teuchos::null;
                Teuchos::RCP<Belos::LinearProblem<ST, MV, OP >> _problem = Teuchos::null;
                Teuchos::RCP<Belos::SolverManager<ST, MV, OP >> _solver = Teuchos::null;
                bool _recPrec{true};
                Teuchos::RCP<Teuchos::ParameterList>  _ifpackParamList; 
        };
    }
}

#endif //_belos_h
