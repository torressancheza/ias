//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <Teuchos_ParameterList.hpp>

#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include "ias_Amesos.h"

namespace ias
{
    namespace solvers
    {
        void TrilinosAmesos::Update()
        {
            using namespace std;
            using Teuchos::RCP;
            using Teuchos::rcp;
            
            _maxIter = 1;
            _resTol = 1.E-8;
            try
            {
                Solver::Update();
            }
            catch (const runtime_error& error)
            {
                string what = error.what();
                throw runtime_error("TrilinosAmesos::Update: " + what);
            }
            
            if(_nameParams.size() != _params.size())
               throw runtime_error("TrilinosAmesos::Update: the vectors containing the name of the Amesos parameters and their values have a different size.");
                        
            
            Teuchos::ParameterList params;
            for(size_t i = 0; i < _params.size(); i++)
                params.set(_nameParams[i], _params[i].c_str());
            _solver = rcp(_AmesFact.Create("Klu", *_integration->getLinearProblem()));
            _solver->SetParameters(params);

        }
    
        void TrilinosAmesos::solve()
        {
            using Teuchos::rcp;

            _solver->SymbolicFactorization();
            _solver->NumericFactorization();
        
            int err = _solver->Solve();

            _converged = err == 0;
            _nIter = 0;
        }
    }
}
