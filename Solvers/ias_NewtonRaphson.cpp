#include <AztecOO.h>
#include <Amesos.h>
#include <Amesos2.hpp>
#include <Amesos2_Factory.hpp>
#include <iomanip>


#include "ias_NewtonRaphson.h"

namespace ias
{


    int NewtonRaphson(Teuchos::RCP<Integration> integration, double restol, double soltol, int maxiter, bool verbose, Teuchos::RCP<Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator>> solver, Teuchos::RCP<Ifpack_Preconditioner> precond)
    {
        using namespace std;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        int ierr{0};
        int  iter{0};
        bool conv{false};
        
        double normR0{};
        double normS0{};
        
        RCP<AztecOO> solverrr;

        
//        RCP<Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>> problem = Teuchos::rcp(new Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>(integration->_matrix, integration->_sol, integration->_vector));
//
//
//          RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList ();
//        solverParams->set ("Num Blocks", 100);
//        solverParams->set ("Maximum Restarts", 100);
//        solverParams->set ("Maximum Iterations", 1000);
//        solverParams->set ("Convergence Tolerance", 1.E-8);
//        Teuchos::RCP<Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator>> solverr = Teuchos::rcp( new Belos::PseudoBlockGmresSolMgr<double, Epetra_MultiVector, Epetra_Operator>(problem, solverParams));
//
//
//        Belos::ReturnType ret;
        while(not conv)
        {
            iter++;
            
            
//            integration->Update();
            integration->fillVectorWithScalar(0.0);
            integration->fillSolutionWithScalar(0.0);
            integration->fillMatrixWithScalar(0.0);
            integration->InitialiseGlobalIntegrals(0.0);
            integration->InitialiseCellIntegrals(0.0);
//            integration->_tissue->calculateInteractingElements(0.11);
            integration->computeSingleIntegral();
            integration->computeDoubleIntegral();
            integration->assemble();
            
//            Teuchos::ParameterList ifpackList;  // Parameters of Ifpack, very simple ones
//            ifpackList.set("fact: ict level-of-fill", 1.0);
//            ifpackList.set("schwarz: combine mode", "Add");
//            Ifpack Factory;
//            int OverlapLevel = 1; // must be >= 0. If Comm.NumProc() == 1 it is ignored.
//            auto _precond = Teuchos::rcp(Factory.Create(Ifpack::ILU, integration->_matrix.getRawPtr(), OverlapLevel));
//            assert(_precond != Teuchos::null);
//            _precond->SetParameters(ifpackList);
//            _precond->Initialize();
////            _precond->Compute();
            
//            RCP<Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>> problem = Teuchos::rcp(new Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>(integration->_matrix, integration->_sol, integration->_vector));
//
//            precond->Compute();
//            RCP<Belos::EpetraPrecOp> belosPrec = rcp( new Belos::EpetraPrecOp( precond ) );
//            problem->setLeftPrec(belosPrec);
//            bool set = problem->setProblem();
//            solverr->setProblem(problem);
//
//
//            solverr->reset (Belos::Problem);
//
//
//            try
//            {
//                ret = solverr->solve();
//            }
//            catch (Belos::StatusTestError)
//            {
//                cout << "HOLA" << endl;
//            }
//
//            bool converged = ret == Belos::Converged;

            solverrr = rcp(new AztecOO(*(integration->getLinearProblem())));

            solverrr->SetAztecOption(AZ_max_iter, 500);
            solverrr->SetAztecOption(AZ_tol, restol);
            solverrr->SetAztecOption(AZ_solver, AZ_bicgstab);
//            solverrr->SetAztecOption(AZ_solver, AZ_gmres);
            solverrr->SetAztecOption(AZ_precond, AZ_dom_decomp);
            solverrr->SetAztecOption(AZ_subdomain_solve, AZ_ilu);
//            solver->SetAztecOption(AZ_conv, AZ_noscaled);
//            solver->SetAztecOption(AZ_conv, AZ_rhs);
            solverrr->SetAztecOption(AZ_output, AZ_none);
//            solver->SetAztecOption(AZ_output, AZ_last);
//
            solverrr->Iterate(500, restol);
//            int numIter = solver->NumIters();
            auto* status = solverrr->GetAztecStatus();
            bool converged = status[AZ_why] == 0;
//            bool converged = true;
            
//            Amesos AmesFact;
//            RCP<Amesos_BaseSolver> solverrr = rcp(AmesFact.Create("Klu", *(integration->getLinearProblem())));
//            solverrr->SymbolicFactorization();
//            solverrr->NumericFactorization();
//            int err = solverrr->Solve();
//            bool converged = err == 0;

            
//            RCP<Amesos2::Solver<Epetra_CrsMatrix, Epetra_MultiVector>> solver = Amesos2::create<Epetra_CrsMatrix, Epetra_MultiVector>("MUMPS", integration->A, integration->sol, integration->rhs);
//            solver->symbolicFactorization();
//            solver->numericFactorization();
//            solver->solve();
//
//            double residual{};
//            bool converged{false};
//            auto A = integration->A;
//            auto B = integration->rhs;
//            auto X = integration->sol;
//            Epetra_MultiVector Ax(B->Map(), 1);
//            A->Multiply(false, *X, Ax);
//            Ax.Update(1.0, *B, -1.0);
//            Ax.Norm2(&residual);
//            if (residual < restol)
//                converged = true;
            
//            auto sol = integration->sol;
//            auto sol = integration.rhs / integration.A;
//            auto prod = integration.A * sol;
//            double linerr{};
//            for(size_t i=0; i < sol.size(); i++)
//            {
//                double dif = prod[i]-integration._b[i];
//                linerr += dif*dif;
//            }
//            if (linerr>restol)
//            {
//                ierr = 1;
//                break;
//            }
            
            if(!converged)
            {
                ierr = 1;
                break;
            }
            
            integration->addSolToDOFs(-1.0);
            
//            double **raw_ptr;
//            integration->sol->ExtractView(&raw_ptr);
            
//            integration->tissue->addVectorToDOFs(raw_ptr[0], -1.0);

//            //Solution and residual norms
//            double normR{},normS{};
//            for(size_t i=0; i < sol.size(); i++)
//            {
//                normR += integration._b[i]*integration._b[i];
//                normS += sol[i]*sol[i];
//            }
//            normS = sqrt(normS);
//            normR = sqrt(normR);
            
            double normR{}, normS{};
            integration->getVector()->NormInf(&normR);
            integration->getSolution()->NormInf(&normS);
            
//            if(iter==1)
//            {
//                normR0 = normR;
//                normS0 = normS;
//            }
//            normR /= normR0;
//            normS /= normS0;

            if(verbose and integration->getTissue()->getMyPart()==0)
                cout << scientific << "Iteration " << iter << " of " <<  maxiter << ":   ResErr=" << normR << "    " << "SolErr=" << normS << endl;

            if(normR < restol and normS < soltol)
                conv = true;
            else if (iter==maxiter)
            {
                ierr = 2;
                break;
            }
            
            integration->getTissue()->updateGhosts();
        }
        
        if(not conv)
        {
            if ( integration->getTissue()->getMyPart()==0)
            {
                if (ierr == 1)
                {
                    cout << "Newton Raphson exits with error: linear solved failed" << endl;
                    
                }
                else
                    cout << "Newton Raphson exits with error: maximum number of iterations reached" << endl;
            }
        }
        else
            ierr = -iter;
        
        return ierr;
    }
}
