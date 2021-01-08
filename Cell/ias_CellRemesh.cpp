//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkCleanPolyData.h>
#include <vtkCurvatures.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkCellLocator.h>

#include "SurfaceRemeshing.h"

#include "ias_Cell.h"

namespace ias
{
    void Cell::remesh(double tArea)
    {
        //FIXME: add more options anc clean it!
        
        using namespace std;
        using namespace Tensor;
        
        if (_connec.shape()[1] != 3)
            throw runtime_error("Cell::remesh: Only programmed for triangular meshes at the time");
               
        if(_bfType == BasisFunctionType::LoopSubdivision)
            tArea *= 4.0;
        
        vtkSmartPointer<vtkPolyData> polydata = getPolyData();
        
        vtkSmartPointer<vtkCurvatures> meanCurva = vtkSmartPointer<vtkCurvatures>::New();
        meanCurva->SetInputData(polydata);
        meanCurva->SetCurvatureTypeToMean();
        meanCurva->Update();
        
        vtkSmartPointer<vtkCurvatures> gaussCurva = vtkSmartPointer<vtkCurvatures>::New();
        gaussCurva->SetInputData(polydata);
        gaussCurva->SetCurvatureTypeToGaussian();
        gaussCurva->Update();
                
        vtkSmartPointer<vtkDoubleArray> targetAreaArray = vtkSmartPointer<vtkDoubleArray>::New();
        targetAreaArray->SetNumberOfComponents(1);
        targetAreaArray->SetNumberOfTuples(getNumberOfPoints());
        targetAreaArray->SetName("TargetArea");
        
        double bounds[6];
        polydata->GetBounds(bounds);
        
        double dx = bounds[1]-bounds[0];
        double dy = bounds[3]-bounds[2];
        double dz = bounds[5]-bounds[4];
        double r = 1./3.0*(dx+dy+dz);
        
        for(int i =0; i < getNumberOfPoints(); i++) //FIXME: this should be an option
        {

            double H = meanCurva->GetOutput()->GetPointData()->GetScalars()->GetTuple(i)[0];
            double K = gaussCurva->GetOutput()->GetPointData()->GetScalars()->GetTuple(i)[0];
            
            double targetArea{};
            if(H*H-K>0)
            {
                double c1 = H + sqrt(H*H-K);
                double c2 = H - sqrt(H*H-K);
                
                 targetArea = tArea/(r*max(c1,c2));
            }
            else
                targetArea = tArea/(r*sqrt(abs(K)));

            targetArea = min(tArea,targetArea);
            targetArea = max(0.5*tArea,targetArea);
            
            targetAreaArray->SetTuple(i, &targetArea);
//            x(1) = curvaturesFilter->GetPoint(i)[1];
//            x(2) = curvaturesFilter->GetPoint(i)[2];
        }
        
        polydata->GetPointData()->SetScalars(targetAreaArray);

        
        
        vtkSmartPointer<SurfaceRemeshing> vtkRemeshing = SurfaceRemeshing::New();
        vtkRemeshing->SetInputData( polydata );
        vtkRemeshing->SetElementSizeModeToTargetAreaArray();
        vtkRemeshing->SetTargetAreaArrayName("TargetArea");
        vtkRemeshing->SetNumberOfIterations(10);
        vtkRemeshing->SetAspectRatioThreshold(1.1);
//            vtkRemeshing->SetCollapseAngleThreshold(1.0);
        vtkRemeshing->SetTriangleSplitFactor(10.0);
        vtkRemeshing->SetNumberOfConnectivityOptimizationIterations(500);
//            vtkRemeshing->SetPreserveBoundaryEdges(true);
        vtkRemeshing->Update();
        
        vtkSmartPointer<vtkCleanPolyData> vtkClean = vtkCleanPolyData::New();
        vtkClean->SetInputData(  vtkRemeshing->GetOutput() );
//        vtkClean->SetPointMerging(true);
//        vtkClean->SetConvertPolysToLines(false);
//        vtkClean->SetConvertLinesToPoints(false);
//        vtkClean->SetTolerance(1.E-8);
        vtkClean->Update();

        vtkSmartPointer<vtkPolyData>  polydata_f = vtkClean->GetOutput();
        
        if(_bfType==BasisFunctionType::LoopSubdivision)
        {
            vtkSmartPointer<vtkPolyDataAlgorithm> subdivisionFilter = vtkSmartPointer<vtkLoopSubdivisionFilter>::New();
            subdivisionFilter->SetInputData(polydata_f);
            dynamic_cast<vtkLoopSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(1);
            subdivisionFilter->Update();
            polydata_f = subdivisionFilter->GetOutput();
            
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            for(int i = 0; i < polydata_f->GetNumberOfPoints(); i++)
            {
                double *x = polydata_f->GetPoint(i);
                double closestpoint[3];
                vtkIdType cellId;
                int subId;
                double dist;
                
                vtkSmartPointer<vtkCellLocator> loc = vtkSmartPointer<vtkCellLocator>::New();
                loc->SetDataSet(polydata);
                loc->BuildLocator();
                
                loc->FindClosestPoint(x, closestpoint, cellId, subId, dist);
                
                points->InsertNextPoint(closestpoint);
            }
            
            polydata_f->SetPoints(points);
        }
        
        auto points = polydata_f->GetPoints();
        auto cells  = polydata_f->GetPolys();
        
        //Dimentionalise eveything
        int nVert = 3;
        int nElem = cells->GetNumberOfCells();
        int nPts = points->GetNumberOfPoints();
        
        _connec.resize(nElem,nVert);
        
        for(int i =0; i < nElem; i++)
        {
            tensor<int,1> c = _connec(i,all);
            
            vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();
            cells->GetNextCell(element);
            
            if(nVert != element->GetNumberOfIds())
                throw runtime_error("Some element is not a triangle after remeshing!");
            
            for(int k = 0; k < element->GetNumberOfIds(); k++)
                c(k) = element->GetId(k);
        }
                
        _nodeFields.resize(nPts,int(_nodeFieldNames.size()));
        
        _nodeFields = 0.0;
        for(int i =0; i < nPts; i++)
        {
            tensor<double,1> x = _nodeFields(i,all);

            x(0) = points->GetPoint(i)[0];
            x(1) = points->GetPoint(i)[1];
            x(2) = points->GetPoint(i)[2];
        }
        
        Update();
        //FIXME: do a least squares here for the rest of fields!
    }
}
