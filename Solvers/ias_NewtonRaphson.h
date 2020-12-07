#ifndef _NewtonRaphson_h
#define _NewtonRaphson_h


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

#include "ias_Integration.h"

namespace ias
{
    inline Teuchos::RCP<Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator> >
    makeSolver (Teuchos::RCP<Epetra_MultiVector> X,
                Teuchos::RCP<Epetra_MultiVector> B,
                Teuchos::RCP<Epetra_CrsMatrix> A,
//                Teuchos::RCP<const OP> M
                Teuchos::RCP<Ifpack_Preconditioner> precond
                )
    {
      using Teuchos::ParameterList;
      using Teuchos::parameterList;
      using Teuchos::RCP;
      using Teuchos::rcp;


        RCP<Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>> problem = Teuchos::rcp(new Belos::LinearProblem<double, Epetra_MultiVector, Epetra_Operator>(A, X, B));
        

//        problem->setHermitian(true);
//        problem->setLeftPrec(precond);
        bool set = problem->setProblem();

        // Make an empty new parameter list.
        RCP<ParameterList> solverParams = parameterList ();

      solverParams->set ("Num Blocks", 40);
      solverParams->set ("Maximum Restarts", 10);
      solverParams->set ("Maximum Iterations", 1000);
      solverParams->set ("Convergence Tolerance", 1.E-8);
      
      Teuchos::RCP<Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator>> solver = Teuchos::rcp( new Belos::BlockGmresSolMgr<double, Epetra_MultiVector, Epetra_Operator>(problem, solverParams));
        
        
      // Tell the solver what problem you want to solve.
//      solver->setProblem(problem);
//        problem->setOperator(A);
//        problem->setProblem(X, B);
//
//        solver->reset (Belos::Problem);

//        solver->solve();
        
//        cout << "HOLA2" << endl;
        
      return solver;
    }

    inline Teuchos::RCP<Ifpack_Preconditioner> precond( Teuchos::RCP<Epetra_CrsMatrix> A )
    {
        Teuchos::ParameterList ifpackList;  // Parameters of Ifpack, very simple ones

        ifpackList.set("fact: ict level-of-fill", 1.0);
        ifpackList.set("schwarz: combine mode", "Add");

        Ifpack Factory;
        int OverlapLevel = 1; // must be >= 0. If Comm.NumProc() == 1 it is ignored.
        auto _precond = Teuchos::rcp(Factory.Create(Ifpack::ILU, A.getRawPtr(), OverlapLevel));
        assert(_precond != Teuchos::null);
        _precond->SetParameters(ifpackList);
        _precond->Initialize();
        _precond->Compute();
        
        return _precond;
    }


    int NewtonRaphson(Teuchos::RCP<Integration> integration, double restol=1.E-8, double soltol=1.E-8, int maxiter=5, bool verbose=true, Teuchos::RCP<Belos::SolverManager<double, Epetra_MultiVector, Epetra_Operator>> solver = Teuchos::null, Teuchos::RCP<Ifpack_Preconditioner> precond = Teuchos::null);
}

#endif //_NewtonRaphson_h
