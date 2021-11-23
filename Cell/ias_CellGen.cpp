//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <vtkSmartPointer.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkReverseSense.h>

#include "Tensor.h"

#include "ias_Cell.h"

namespace ias
{
    void Cell::generateSphereFromPlatonicSolid(int nSubdiv, double r, int type)
    {
        using namespace Tensor;
        using namespace std;
        
        if(type != VTK_SOLID_ICOSAHEDRON and type != VTK_SOLID_OCTAHEDRON and type != VTK_SOLID_TETRAHEDRON)
            throw runtime_error("Cell::generateSphereFromPlatonicSolid: type can only be VTK_SOLID_ICOSAHEDRON or VTK_SOLID_OCTAHEDRON or VTK_SOLID_TETRAHEDRON");
        
        vtkSmartPointer<vtkPlatonicSolidSource> source = vtkSmartPointer<vtkPlatonicSolidSource>::New();
        source->SetSolidType(type);
        source->Update();
        
        vtkSmartPointer<vtkPolyDataAlgorithm> subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
        subdivisionFilter->SetInputData(source->GetOutput());
        dynamic_cast<vtkLinearSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(nSubdiv);
        subdivisionFilter->Update();
        vtkSmartPointer<vtkPolyData> vtkoutput = subdivisionFilter->GetOutput();

        if(type == VTK_SOLID_OCTAHEDRON)
        {
            vtkSmartPointer<vtkReverseSense> reverseSense = vtkSmartPointer<vtkReverseSense>::New();
            reverseSense->SetInputData(vtkoutput);
            reverseSense->ReverseCellsOn();
            reverseSense->Update();
            vtkoutput = reverseSense->GetOutput();
        }
        
        
        int nPts  = vtkoutput->GetNumberOfPoints();
        int nElem = vtkoutput->GetNumberOfCells();
        
        _connec.resize(nElem,3);
        _nodeFields.resize(nPts,3);

        for( int i = 0; i < nPts; i++ )
        {
            array<double,3> x;
            vtkoutput->GetPoint(i, x.data());
            double norm = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
            double factor = r/norm;
            _nodeFields(i,0) = x[0] * factor;
            _nodeFields(i,1) = x[1] * factor;
            _nodeFields(i,2) = x[2] * factor;
        }

        for( int i = 0; i < nElem; i++ )
        {
            vtkSmartPointer<vtkIdList> ptsIds = vtkSmartPointer<vtkIdList>::New();
            vtkoutput->GetCellPoints(i, ptsIds);
            
            _connec(i,0) = ptsIds->GetId(0);
            _connec(i,1) = ptsIds->GetId(1);
            _connec(i,2) = ptsIds->GetId(2);
        }
    }
}

