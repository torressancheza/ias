//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************


#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <AztecOO.h>
#include <Amesos_BaseSolver.h>

#include <BelosLinearProblem.hpp>
#include <BelosSolverManager.hpp>

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockTFQMRSolMgr.hpp>
#include <BelosGCRODRSolMgr.hpp>
#include <BelosRCGSolMgr.hpp>

#include <Ifpack_Preconditioner.h>
#include <Ifpack.h>

#include "ias_Belos.h"

namespace ias
{
    namespace solvers
    {
        void TrilinosBelos::Update()
        {
            using namespace std;
            using Teuchos::RCP;
            using Teuchos::rcp;
            
            try
            {
                Solver::Update();
            }
            catch (const runtime_error& error)
            {
                string what = error.what();
                throw runtime_error("TrilinosBelos::Update: " + what);
            }
            
            if(_nameParams.size() != _params.size())
                throw runtime_error("TrilinosBelos::Update: the vectors containing the name of the Belos parameters and their values have a different size.");
            
            auto A = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetMatrix());
            auto X = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetLHS());
            auto B = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetRHS());
            _problem = Teuchos::rcp(new Belos::LinearProblem<ST, MV, OP>(A, X, B));

            for(size_t i = 0; i < _ifpackParams.size(); i++)
            {
                //Try to transform the string into a number
                const char *param = _ifpackParams[i].c_str();
                char* p;
                int converted =  (param, &p, 10);
                if (*p) 
                    _ifpackParamList.set(_ifpackNameParams[i], param);
                else 
                    _ifpackParamList.set(_ifpackNameParams[i], converted);
            }

            _problem->setLeftPrec(_belosPrecond);

            RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList);
            params->set("Maximum Iterations", _maxIter); // Maximum number of iterations allowed
            params->set("Convergence Tolerance", _resTol);  // Relative convergence tolerance requested
            for(size_t i = 0; i < _params.size(); i++)
            {
                //Try to transform the string into a number
                const char *param = _params[i].c_str();
                char* p;
                int converted = strtol(param, &p, 10);
                if (*p) 
                    params->set(_nameParams[i], param);
                else 
                    params->set(_nameParams[i], converted);
            }

            Belos::SolverFactory<ST, MV, OP> factory;
            _solver = factory.create (_solverType, params);
            
            _recPrec = true;        
        }

        void TrilinosBelos::solve()
        {    
            using namespace std;
            using Teuchos::RCP;
            using Teuchos::rcp;

            auto A = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetMatrix());
            auto X = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetLHS());
            auto B = Teuchos::rcpFromRef(*_integration->getLinearProblem()->GetRHS());

            if(_recPrec)
            {
                Ifpack Factory;
                _precond = rcp(Factory.Create(_precondType, A.getRawPtr(), 0));
                _precond->SetParameters(_ifpackParamList);
                _precond->Initialize();
                _precond->Compute();
                _belosPrecond = rcp (new Belos::EpetraPrecOp (_precond));
            }

            _problem->setOperator(A);
            _problem->setProblem(X, B);
            _problem->setLeftPrec(_belosPrecond); 

            _solver->setProblem(_problem);

            Belos::ReturnType ret;
            try
            {
                ret = _solver->solve();
            }
            catch(Belos::StatusTestError& err)
            {
                cout << err.what() << endl;
            }
            _converged = ret == Belos::Converged;
            _nIter = _solver->getNumIters();
        }
    }
}
