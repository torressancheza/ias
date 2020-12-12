//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <Teuchos_ParameterList.hpp>
#include <AztecOOParameterList.hpp>


#include "ias_AztecOO.h"

namespace ias
{
    namespace solvers
    {
        void TrilinosAztecOO::Update()
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
                throw runtime_error("TrilinosAztecOO::Update: " + what);
            }
            
            if(_nameParams.size() != _params.size())
               throw runtime_error("TrilinosAztecOO::Update: the vectors containing the name of the AztecOO parameters and their values have a different size.");
                        
            _solver = rcp(new AztecOO);
            
            Teuchos::ParameterList params;
            for(size_t i = 0; i < _params.size(); i++)
                params.set(_nameParams[i], _params[i].c_str());
            _solver->SetParameters(params);
        }
    
        void TrilinosAztecOO::solve()
        {
            using Teuchos::rcp;
            _solver->SetProblem(*_integration->getLinearProblem());
//            _solver->DestroyPreconditioner();
            _solver->Iterate(_maxIter, _resTol);
            
            auto* status = _solver->GetAztecStatus();
            _converged = status[AZ_why] == 0;
            _nIter = _solver->NumIters();
        }
    }
}
