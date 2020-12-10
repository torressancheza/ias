#include <vtkSmartPointer.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkCleanPolyData.h>

#include "Tensor.h"

#include "ias_Cell.h"

namespace ias
{
    void Cell::generateSphereFromOctahedron(int nSubdiv, double r)
    {
        using namespace Tensor;
        using namespace std;
        
        vtkSmartPointer<vtkPlatonicSolidSource> source = vtkSmartPointer<vtkPlatonicSolidSource>::New();
        source->SetSolidType(VTK_SOLID_OCTAHEDRON);
        source->Update();
        
        vtkSmartPointer<vtkPolyDataAlgorithm> subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
        subdivisionFilter->SetInputData(source->GetOutput());
        dynamic_cast<vtkLinearSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(nSubdiv);
        subdivisionFilter->Update();
        
        auto vtkoutput = subdivisionFilter->GetOutput();
        
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

