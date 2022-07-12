//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iostream>

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkCell.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDataSetSurfaceFilter.h>

#include "Tensor.h"
#include "ias_LinearElements.h"
#include "ias_LoopSubdivision.h"
#include "ias_Cell.h"

namespace ias
{
    void InsertNodeDataInVTK(vtkSmartPointer<vtkUnstructuredGrid> vtkmesh, Tensor::tensor<double,2>& dofs, const std::vector<std::string>& names)
    {
        auto shape = dofs.shape();
        
        assert(shape[1] == int(names.size()));

        for(int i = 0; i < dofs.shape()[1]; i++)
        {
            //Insert nodeFields
            vtkSmartPointer<vtkDoubleArray> vtknodeFields = vtkSmartPointer<vtkDoubleArray>::New();
            vtknodeFields->SetNumberOfComponents(1);
            vtknodeFields->SetNumberOfTuples(shape[0]);
            vtknodeFields->SetName(names[i].c_str());
            
            for(int j = 0; j < shape[0]; j++)
                vtknodeFields->SetValue(j,dofs(j,i));
            vtkmesh->GetPointData()->AddArray(vtknodeFields);
        }
    }

    void GetNodeDataInVTK(Tensor::tensor<double,2>& dofs, std::vector<std::string>& names, vtkSmartPointer<vtkUnstructuredGrid> polydata)
    {
        using namespace std;
        using namespace Tensor;

        int nNodeFields = polydata->GetPointData()->GetNumberOfArrays();
        int nPts = polydata->GetNumberOfPoints();
        dofs.resize(nPts,nNodeFields);

        names.clear();
        for(int i = 0; i < nNodeFields; i++)
        {
            string name = polydata->GetPointData()->GetArrayName(i);
            names.push_back(name);
            vtkSmartPointer<vtkDataArray> vtknodeFields = polydata->GetPointData()->GetArray(name.c_str());

            for(int k = 0; k < nPts; k++)
                dofs(k,i) = vtknodeFields->GetTuple(k)[0];
        }
    }

    void InsertGlobDataInVTK(vtkSmartPointer<vtkUnstructuredGrid> polydata, Tensor::tensor<double,1>& dofs, const std::vector<std::string>& names)
    {
        assert(dofs.shape()[0] == int(names.size()));

        for(int i = 0; i < dofs.shape()[0]; i++)
        {
            vtkSmartPointer<vtkDoubleArray> vtkglobFields = vtkSmartPointer<vtkDoubleArray>::New();
//            vtkglobFields->SetNumberOfTuples(1);
            vtkglobFields->SetNumberOfComponents(1);
            vtkglobFields->SetName(names[i].c_str());
            vtkglobFields->InsertNextValue(dofs(i));
            polydata->GetFieldData()->AddArray(vtkglobFields);
        }
    }

    void GetGlobDataInVTK(Tensor::tensor<double,1>& dofs, std::vector<std::string>& names, vtkSmartPointer<vtkUnstructuredGrid> polydata)
    {
        using namespace Tensor;
        using namespace std;
        
        int nGlobFields = polydata->GetFieldData()->GetNumberOfArrays();
        names.clear();
        dofs.resize(nGlobFields);

        for(int i = 0; i < nGlobFields; i++)
        {
            string name = polydata->GetFieldData()->GetArrayName(i);
            names.push_back(name);

            //Insert nodeFields
            vtkSmartPointer<vtkDataArray> vtkglobFields = polydata->GetFieldData()->GetArray(name.c_str());
            
            dofs(i) = vtkglobFields->GetTuple(0)[0];
        }
    }

    void Cell::saveVTK(std::string namefile)
    {
        using namespace std;
        using namespace Tensor;

        vtkSmartPointer<vtkUnstructuredGrid> polydata = vtkSmartPointer<vtkUnstructuredGrid>::New();
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->SetDataTypeToDouble();

        for(int i =0; i < getNumberOfPoints(); i++)
        {
            tensor<double,1> x = _nodeFields(i,all);
            points->InsertNextPoint ( x(0), x(1), x(2) );
        }
        polydata->SetPoints(points);

        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        for(int i =0; i < getNumberOfElements(); i++)
        {
            tensor<int,1> c = _connec(i,all);
            vtkSmartPointer<vtkIdList> element =vtkSmartPointer<vtkIdList>::New();
            
            for(int k = 0; k < 3; k++)
                element->InsertNextId(c(k));
            
            cells->InsertNextCell(element);
        }
        
        int type = _connec.shape()[1] == 3 ? VTK_TRIANGLE : VTK_LINE;
        polydata->SetCells(type,cells);
        
        InsertNodeDataInVTK(polydata, _nodeFields, _nodeFieldNames);
        InsertGlobDataInVTK(polydata, _cellFields, _cellFieldNames);

        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetInputData(polydata);
        writer->SetFileName(namefile.c_str());
        writer->Update();
    }

    void Cell::loadVTK(std::string namefile)
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
                
        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(namefile.c_str());
        reader->Update();
        vtkSmartPointer<vtkUnstructuredGrid> polydata = reader->GetOutput();
        
        
        vtkSmartPointer<vtkCellArray> cells = polydata->GetCells();
        //Dimentionalise eveything
        vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();;
        cells->GetCell(0,element);
        int nElem = cells->GetNumberOfCells();
        int nVert = element->GetNumberOfIds();
        
        _connec.resize(nElem,nVert);
        for(int i =0; i < nElem; i++)
        {
            auto c = _connec(i,all);
            vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();;
            cells->GetNextCell(element);
                        
            for(int k = 0; k < element->GetNumberOfIds(); k++)
                c(k) = element->GetId(k);
        }
        
        GetNodeDataInVTK(_nodeFields, _nodeFieldNames, polydata);
        GetGlobDataInVTK(_cellFields, _cellFieldNames, polydata);
    }


    vtkSmartPointer<vtkPolyData> Cell::getPolyData(bool withNodeFields, bool withGlobFields)
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        vtkSmartPointer<vtkUnstructuredGrid> polydata = vtkSmartPointer<vtkUnstructuredGrid>::New();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        for(int i =0; i < getNumberOfPoints(); i++)
        {
            tensor<double,1> x = _nodeFields(i,range(0,2));
            points->InsertNextPoint ( x(0), x(1), x(2) );
        }
        polydata->SetPoints(points);

        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        for(int i =0; i < getNumberOfElements(); i++)
        {
            vtkSmartPointer<vtkIdList> element =vtkSmartPointer<vtkIdList>::New();

            if(_bfType==BasisFunctionType::LoopSubdivision)
            {
                int eNN = _bfs->getNumberOfNeighbours(i);
                int*  adjEN = _bfs->getNeighbours(i);
                element->InsertNextId(adjEN[0]);
                element->InsertNextId(adjEN[1]);
                element->InsertNextId(adjEN[eNN-6]);
            }
            else
            {
                tensor<int,1> c = _connec(i,all);
                
                for(int k = 0; k < 3; k++)
                    element->InsertNextId(c(k));
            }
            
            cells->InsertNextCell(element);
        }
        int type = _connec.shape()[1] == 3 ? VTK_TRIANGLE : VTK_LINE;
        polydata->SetCells(type,cells);
        
        if(withNodeFields)
            InsertNodeDataInVTK(polydata, _nodeFields,  _nodeFieldNames);
        if(withGlobFields)
            InsertGlobDataInVTK(polydata, _cellFields, _cellFieldNames);
        
        vtkSmartPointer<vtkDataSetSurfaceFilter> surfaceFilter = vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
        surfaceFilter->SetInputData(polydata);
        surfaceFilter->Update();
        
        return surfaceFilter->GetOutput();
    }

}
