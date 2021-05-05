//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <vtkPolyData.h>
#include <vtkCellLocator.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkPolyDataWriter.h>

#include <Tensor.h>

#include <ias_Tissue.h>
#include <ias_Integration.h>
#include <ias_AztecOO.h>

namespace ias
{
    void leastSquaresCPP(Teuchos::RCP<ias::SingleIntegralStr> fill)
    {
        using namespace Tensor;

        Cell* cell = static_cast<Cell*>(fill->userAuxiliaryObjects[0]);
        vtkPolyData* polydata = static_cast<vtkPolyData*>(fill->userAuxiliaryObjects[1]);
        vtkCellLocator* cellLocator = static_cast<vtkCellLocator*>(fill->userAuxiliaryObjects[2]);

        int eNN = fill->eNN;
        tensor<double,2>& nborFields = fill->nborFields;
        tensor<double,1>   bfs(fill->bfs[0].data(),eNN);
        tensor<double,2>  Dbfs(fill->bfs[1].data(),eNN, 2);

        tensor<double,1> x  = bfs * nborFields(all,range(0,2));
        tensor<double,2> Dx = Dbfs.T() * nborFields(all,range(0,2));
        tensor<double,2> metric = Dx*Dx.T();
        double jac = sqrt(metric.det());

        double closestPoint[3];
        double closestPointDist2; 
        vtkIdType cellId; 
        int subId;
        cellLocator->FindClosestPoint(x.data(), closestPoint, cellId, subId, closestPointDist2);
        
        std::vector<double> parametric(3), weights(3);
        double closestPoint_[3];
        double dist;
        polydata->GetCell(cellId)->EvaluatePosition(closestPoint, closestPoint_, subId, parametric.data(), dist, weights.data());

        // std::cout << parametric[0] << " " << parametric[1] << " " << parametric[2] << std::endl;
        tensor<double,1> fields = cell->getInterpolatedNodeFields({parametric[0], parametric[1]}, cellId);

        int nDOFs = fill->vec_n.shape()[1];
        fill->mat_nn += jac * outer(outer(bfs,bfs),Identity(nDOFs)).transpose({0,2,1,3});
        fill->vec_n += jac * outer(bfs, fields);
    }


    void mapFields(Teuchos::RCP<Cell> cell1, Teuchos::RCP<Cell> cell2)
    {
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        RCP<Tissue> cellTissue = rcp(new Tissue(MPI_COMM_SELF));
        cellTissue->addCellToTissue(cell1);
        cellTissue->Update();

        vtkSmartPointer<vtkPolyData> polydata1 = cell1->getPolyData();
        vtkSmartPointer<vtkPolyData> polydata2 = cell2->getPolyData();
        // if(cell1->getBasisFunctionType() == BasisFunctionType::LoopSubdivision)
        // {
        //     vtkSmartPointer<vtkPolyDataAlgorithm> subdivisionFilter = vtkSmartPointer<vtkLoopSubdivisionFilter>::New();
        //     subdivisionFilter->SetInputData(polydata2);
        //     dynamic_cast<vtkLoopSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(3);
        //     subdivisionFilter->Update();
        //     polydata2 = subdivisionFilter->GetOutput();
        // }

        int myRank;
        std::string filename="polydata"+std::to_string(myRank)+".vtk";
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
        writer->SetInputData(polydata2);
        writer->SetFileName(filename.c_str());
        writer->Update();

        vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
        cellLocator->SetDataSet(polydata2);
        cellLocator->BuildLocator();

        RCP<Integration> integration = rcp(new Integration);
        integration->setTissue(cellTissue);
        integration->setNodeDOFs(cell1->getNodeFieldNames());
        integration->setCellDOFs({});
        integration->setSingleIntegrand(leastSquaresCPP);
        integration->setNumberOfIntegrationPointsSingleIntegral(12);
        integration->setNumberOfIntegrationPointsDoubleIntegral(1);
        integration->userAuxiliaryObjects.push_back(&(*cell2));
        integration->userAuxiliaryObjects.push_back(&(*polydata2));
        integration->userAuxiliaryObjects.push_back(&(*cellLocator));
        integration->Update();
        

        RCP<solvers::TrilinosAztecOO> solver = rcp(new solvers::TrilinosAztecOO);
        solver->setIntegration(integration);
        solver->addAztecOOParameter("solver","gmres");
        solver->addAztecOOParameter("precond","dom_decomp");
        solver->addAztecOOParameter("subdomain_solve","ilu");
        solver->addAztecOOParameter("output","none");
        solver->setMaximumNumberOfIterations(500);
        solver->setResidueTolerance(1.E-8);
        solver->Update();

        for (int i = 0; i < 2; i++)
        {
            integration->computeSingleIntegral();
            integration->assemble();
            solver->solve();
            integration->setSolToDOFs();
        }

    }
}