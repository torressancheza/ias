//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iomanip>
#include "ias_NewtonRaphson.h"

namespace ias
{

    namespace solvers
    {
        void NewtonRaphson::Update()
        {
            using namespace std;
            
            if(_linearSolver == Teuchos::null)
                throw runtime_error("The linear solver has not been set");
            
            if(_integration == Teuchos::null)
                _integration = _linearSolver->getIntegration();
            else if(_integration != _linearSolver->getIntegration())
                throw runtime_error("NewtonRaphson::Update: The integration object does not match that of the linear solver.");

            try
            {
                Solver::Update();
            }
            catch (const runtime_error& error)
            {
                string what = error.what();
                throw runtime_error("NewtonRaphson::Update: " + what);
            }
        }
    
        void NewtonRaphson::solve()
        {
            using namespace std;

            int ierr{0};
            int  iter{0};
            bool conv{false};

            double normR0{};
            double normS0{};

            while(not conv)
            {
                iter++;
                _integration->fillVectorWithScalar(0.0);
                _integration->fillSolutionWithScalar(0.0);
                _integration->fillMatrixWithScalar(0.0);
                _integration->InitialiseTissIntegralFields(0.0);
                _integration->InitialiseCellIntegralFields(0.0);
                _integration->computeSingleIntegral();
                _integration->computeDoubleIntegral();
                _integration->assemble();
                
                _linearSolver->solve();
                bool converged = _linearSolver->getConvergence();
                
                if(!converged)
                {
                    ierr = 1;
                    break;
                }

                _integration->addSolToDOFs(-1.0);

                double normR{}, normS{};
                _integration->getVector()->NormInf(&normR);
                _integration->getSolution()->NormInf(&normS);

                if(_convRel)
                {
                    if(iter==1)
                    {
                        normR0 = normR;
                        normS0 = normS;
                    }
                    
                    normR /= normR0;
                    normS /= normS0;
                }



                if(_verbose and _integration->getTissue()->getMyPart()==0)
                    cout << scientific << "Iteration " << iter << " of " <<  _maxIter << ":   ResErr=" << normR << "    " << "SolErr=" << normS << endl;

                if(normR < _resTol and normS < _solTol)
                    conv = true;
                else if (iter==_maxIter)
                {
                    ierr = 2;
                    break;
                }

                _integration->getTissue()->updateGhosts();
            }

            if(not conv)
            {
                if ( _integration->getTissue()->getMyPart()==0)
                {
                    if (ierr == 1)
                    {
                        cout << "NewtonRaphson::solve: exit with error: linear solved failed" << endl;

                    }
                    else
                        cout << "NewtonRaphson::solve: exit with error: maximum number of iterations reached" << endl;
                }
            }
            
            _converged = conv;
            _nIter = iter;
        }
    }
}
