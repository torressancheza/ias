//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <random>
#include <numeric>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkIdList.h>
#include <vtkCleanPolyData.h>
#include <vtkCurvatures.h>
#include <vtkClipClosedSurface.h>
#include <vtkTriangle.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
//#include <vtkCenterOfMass.h>
#include <vtkIntegrateAttributes.h>
#include <vtkFillHolesFilter.h>

#include <vtkFeatureEdges.h>
#include <vtkClipPolyData.h>

#include <vtkPolyDataWriter.h>

#include <vtkPointLocator.h>

#include <vtkAppendPolyData.h>

#include <vtkDelaunay2D.h>

#include <vtkPolyDataNormals.h>

#include <vtkContourTriangulator.h>

#include <vtkSmartPointer.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkCleanPolyData.h>

#include "SurfaceRemeshing.h"
#include <vtkLoopSubdivisionFilter.h>

#include "ias_LinearElements.h"
#include "ias_LoopSubdivision.h"
#include "ias_Cell.h"


namespace ias
{

    void removePoints3Neighbours(vtkSmartPointer<vtkPolyData> polydata)
    {
        using namespace std;

        polydata->BuildLinks();
        vector<int> newPointLabels(polydata->GetNumberOfPoints());
        iota (newPointLabels.begin(), newPointLabels.end(), 0);

        vector<int> cellsToDelete;
        vector<array<int,3>> cellsToAdd;
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

        cout << "AQUI 0 " << endl;
        for(int i = 0; i < polydata->GetNumberOfPoints(); i++)
        {
            vtkSmartPointer<vtkIdList> cellIdList = vtkSmartPointer<vtkIdList>::New();
            polydata->GetPointCells(i, cellIdList);

            if(cellIdList->GetNumberOfIds() == 3)
            {
                transform(newPointLabels.begin()+i,newPointLabels.end(),newPointLabels.begin()+i,[](int x){return x-1;});

                cellsToDelete.push_back(cellIdList->GetId(0));
                cellsToDelete.push_back(cellIdList->GetId(1));
                cellsToDelete.push_back(cellIdList->GetId(2));

                set<int> ids;
                for(int j = 0; j < 3; j++)
                {
                    auto cell = polydata->GetCell(cellIdList->GetId(j));
                    auto pointIdList = cell->GetPointIds();
                    ids.insert(pointIdList->GetId(0));
                    ids.insert(pointIdList->GetId(1));
                    ids.insert(pointIdList->GetId(2));
                }
                if(ids.size() != 4)
                    throw runtime_error("removePoints3Neighbours: Point identified with 3 neighbouring elements but there are more than 4 vertices forming the elements, which does not make sense. This is probably caused by a nonmanifold problem during remeshing.");

                array<int,3> cell;
                int n{};
                for (auto j = ids.begin(); j != ids.end(); j++) 
                {
                    if(i != (*j))
                    {
                        cell[n] = *j;
                        n++;
                    }      
                }
                cellsToAdd.push_back(cell);
            }
            else
            {
                points->InsertNextPoint(polydata->GetPoint(i));
            }
        }

        cout << "AQUI 1 " << endl;
        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        for(int e = 0; e < polydata->GetNumberOfCells(); e++)
        {
            if(find(cellsToDelete.begin(), cellsToDelete.end(), e) == cellsToDelete.end())
            {
                auto cell = polydata->GetCell(e);
                if(cell->GetNumberOfPoints() != 3)
                    throw runtime_error("removePoints3Neighbours: Element "+to_string(e)+" is not a triangle.");
                int i1 = cell->GetPointId(0);
                int i2 = cell->GetPointId(1);
                int i3 = cell->GetPointId(2);
                cells->InsertNextCell({newPointLabels[i1], newPointLabels[i2], newPointLabels[i3]});
            }
        }
        cout << "AQUI 2 " << endl;

        for(auto& c: cellsToAdd)
        {
            cells->InsertNextCell({newPointLabels[c[0]], newPointLabels[c[1]], newPointLabels[c[2]]});
        }

        polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->SetPoints(points);
        polydata->SetPolys(cells);
        cout << "SALGO" << endl;
    }

    Teuchos::RCP<Cell> Cell::cellDivision(double sep, double el_area)
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        if(_bfType == BasisFunctionType::LoopSubdivision)
            el_area *= 4.0;

        //FIXME: make this an option
        random_device rd;  //Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        uniform_real_distribution<> dis(-1.0, 1.0);
        
        vtkSmartPointer<vtkPolyData> polydata = getPolyData();
        
        double separation = sep/2.0;
        
        vtkSmartPointer<vtkIntegrateAttributes> intatr = vtkSmartPointer<vtkIntegrateAttributes>::New();
        intatr->SetInputData(polydata);
        intatr->SetDivideAllCellDataByVolume(true);
        intatr->Update();
        

        auto integral = intatr->GetOutput();
        tensor<double,1> center = {integral->GetPoint(0)[0],integral->GetPoint(0)[1],integral->GetPoint(0)[2]};
        
        double theta = M_PI*(1.0+dis(gen))*0.5;
        double phi   = M_PI*dis(gen);
        
        tensor<double,1> normal = {cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)};
        normal /= sqrt(normal*normal);
        
        RCP<Cell> daughter = rcp(new Cell());
        RCP<Cell> mother = rcp(new Cell());

        for(int n = -1; n <= 1; n+=2)
        {
            tensor<double,1> center_plane = center + n * separation * normal;
            tensor<double,1> normal_plane = n * normal;
            tensor<double,1> e1_plane = {1.0,0.0,0.0};
            tensor<double,1> e2_plane = {0.0,1.0,0.0};
            
            e1_plane -= (e1_plane * normal_plane) * normal_plane;
            e1_plane /= sqrt(e1_plane*e1_plane);
            
            e2_plane -= (e2_plane * normal_plane) * normal_plane + (e2_plane * e1_plane) * e1_plane;
            e2_plane /= sqrt(e2_plane*e2_plane);

            
            vtkSmartPointer<vtkPlane> vtkplane = vtkSmartPointer<vtkPlane>::New();
            vtkplane->SetNormal(normal_plane.data());
            vtkplane->SetOrigin(center_plane.data());


            vtkSmartPointer<vtkPlaneCollection> capPlanes = vtkSmartPointer<vtkPlaneCollection>::New();
            capPlanes->AddItem(vtkplane);

            vtkSmartPointer<vtkClipPolyData> clipPolyData = vtkSmartPointer<vtkClipPolyData>::New();
            clipPolyData->SetInputData(polydata);
            clipPolyData->SetClipFunction(vtkplane);
            clipPolyData->Update();
            
            vtkSmartPointer<vtkPolyData> polydata_1 = vtkSmartPointer<vtkPolyData>(clipPolyData->GetOutput());
            

            vtkSmartPointer<SurfaceRemeshing> vtkRemeshing = SurfaceRemeshing::New();
            vtkRemeshing->SetInputData( polydata_1 );
            vtkRemeshing->SetElementSizeModeToTargetArea();
            vtkRemeshing->SetTargetArea(el_area);
            vtkRemeshing->SetNumberOfIterations(10);
            vtkRemeshing->SetTriangleSplitFactor(10.0);
            vtkRemeshing->SetNumberOfConnectivityOptimizationIterations(100);
            vtkRemeshing->Update();
            
            vtkSmartPointer<vtkCleanPolyData> vtkClean = vtkCleanPolyData::New();
            vtkClean->SetInputData(  vtkRemeshing->GetOutput() );

            vtkClean->Update();
            polydata_1 = vtkClean->GetOutput();

            if(polydata_1->GetNumberOfPoints() == 0 or polydata_1->GetNumberOfPolys() == 0)
                throw runtime_error("Cell::cellDivision: Got polydata without points/cells");

            vtkSmartPointer<vtkFeatureEdges> edges = vtkSmartPointer<vtkFeatureEdges>::New();
            edges->SetInputData(polydata_1);
            edges->BoundaryEdgesOn();
            edges->FeatureEdgesOff();
            edges->NonManifoldEdgesOff();
            edges->ManifoldEdgesOff();
            edges->Update();
            
            auto polydata_b = edges->GetOutput();
            
            vtkSmartPointer<vtkPolyData> polydata_2 = vtkSmartPointer<vtkPolyData>::New();
            vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

            for(int i = 0; i < polydata_b->GetPoints()->GetNumberOfPoints(); i++)
                points->InsertNextPoint(polydata_b->GetPoints()->GetPoint(i));
            points->InsertNextPoint(center_plane.data());

            for(int i = 0; i < polydata_b->GetNumberOfCells(); i++)
            {
                vtkCell* pts = polydata_b->GetCell(i);

                vtkSmartPointer<vtkIdList> pts_ = vtkSmartPointer<vtkIdList>::New();
                pts_->InsertNextId(pts->GetPointId(0));
                pts_->InsertNextId(pts->GetPointId(1));
                pts_->InsertNextId(polydata_b->GetPoints()->GetNumberOfPoints());

                cells->InsertNextCell(pts_);
            }
            polydata_2->SetPoints(points);
            polydata_2->SetPolys(cells);

            vtkRemeshing = SurfaceRemeshing::New();
            vtkRemeshing->SetInputData( polydata_2 );
            vtkRemeshing->SetElementSizeModeToTargetArea();
            vtkRemeshing->SetTargetArea(el_area);
            vtkRemeshing->SetNumberOfIterations(10);
            vtkRemeshing->SetTriangleSplitFactor(10.0);
            vtkRemeshing->SetNumberOfConnectivityOptimizationIterations(100);
            vtkRemeshing->PreserveBoundaryEdgesOn();
            vtkRemeshing->Update();

            polydata_2 = vtkRemeshing->GetOutput();
            
            vtkClean = vtkCleanPolyData::New();
            vtkClean->SetInputData( polydata_2 );
            vtkClean->Update();

            polydata_2 = vtkClean->GetOutput();


            if(polydata_2->GetNumberOfPoints() == 0 or polydata_2->GetNumberOfPolys() == 0)
                throw runtime_error("Cell::cellDivision: Got polydata without points/cells");
            
            vtkSmartPointer<vtkFeatureEdges> edges2 = vtkSmartPointer<vtkFeatureEdges>::New();
            edges2->SetInputData(polydata_2);
            edges2->BoundaryEdgesOn();
            edges2->FeatureEdgesOff();
            edges2->NonManifoldEdgesOff();
            edges2->ManifoldEdgesOff();
            edges2->Update();
            
            auto polydata_b2 = edges2->GetOutput();
            
            polydata_2->BuildLinks();

            cout << "LLEGO AQUI 1" << endl;

            //Check that boundaries match with each other
            if(polydata_b->GetPoints()->GetNumberOfPoints()!=polydata_b2->GetPoints()->GetNumberOfPoints())
            {
                //Let's fix this...
                int n{};
                vector<int> removedCells;
                vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
                vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
                for(int i = 0; i < polydata_2->GetNumberOfPoints(); i++)
                    points->InsertNextPoint(polydata_2->GetPoint(i));

                //Add the last points
                vtkSmartPointer<vtkCellLocator> cellLocator = vtkCellLocator::New();
                cellLocator->SetDataSet(polydata_2);
                cellLocator->BuildLocator();

                vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
                writer->SetInputData(polydata_2);
                writer->SetFileName("polydata_2.vtk");
                writer->Update();
                
                for(int i = 0; i < polydata_b->GetNumberOfPoints(); i++)
                {
                    double* x = polydata_b->GetPoints()->GetPoint(i);

                    double cpoint[3];
                    vtkIdType cellId;
                    int subId;
                    double dist;
                    cellLocator->FindClosestPoint(x, cpoint, cellId, subId, dist);

                    //If the distance is larger than the tolerance, this point has been erased by the remesher...
                    if(dist > 1.E-8)
                    {                        
                        vtkSmartPointer<vtkCell> cell = polydata_2->GetCell(cellId);
                        vtkSmartPointer<vtkIdList> cellPoints = cell->GetPointIds();

                        int i1 = cellPoints->GetId(0);
                        int i2 = cellPoints->GetId(1);
                        int i3 = cellPoints->GetId(2);

                        vector<pair<int,int>> edges = {{i1,i2},{i2,i3},{i3,i1}};

                        points->InsertNextPoint(x);

                        double area = 0.0;

                        pair<int,int> edgeIds = {-1, -1};
                        for(auto& e: edges)
                        {
                            vtkSmartPointer<vtkIdList> nborsIds = vtkIdList::New();
                            vtkSmartPointer<vtkIdList> edge = vtkIdList::New();
                            edge->InsertNextId(e.first);
                            edge->InsertNextId(e.second);
                            
                            polydata_2->GetCellNeighbors(cellId, edge, nborsIds);

                            if(nborsIds->GetNumberOfIds() == 0)
                            {
                                double p1[3]; 
                                polydata_2->GetPoints()->GetPoint(e.first, p1);
                                double p2[3];
                                polydata_2->GetPoints()->GetPoint(e.second, p2);
                                
                                double area_new = vtkTriangle::TriangleArea(x, p1, p2);
                                if(area_new > area)
                                {
                                    area = area_new;
                                    edgeIds.first = e.first;
                                    edgeIds.second = e.second;
                                }
                            }
                        }
                        if(edgeIds.first >= 0 and edgeIds.second >= 0)
                        {
                            double p1[3]; 
                            polydata_2->GetPoints()->GetPoint(edgeIds.first, p1);
                            double p2[3];
                            polydata_2->GetPoints()->GetPoint(edgeIds.second, p2);
                            int opposite = edgeIds.first == i1 ? (edgeIds.second == i2 ? i3 : i2) : (edgeIds.first == i2 ? (edgeIds.second == i1 ? i3: i1) : (edgeIds.second == i1 ? i2: i1));
                            

                            vtkSmartPointer<vtkIdList> triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(opposite);
                            triangle->InsertNextId(edgeIds.first);
                            triangle->InsertNextId(polydata_2->GetNumberOfPoints()+n);
                            cells->InsertNextCell(triangle);

                            triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(opposite);
                            triangle->InsertNextId(polydata_2->GetNumberOfPoints()+n);
                            triangle->InsertNextId(edgeIds.second);
                            cells->InsertNextCell(triangle);

                            removedCells.push_back(cellId);
                            n++;
                        }
                        else
                            throw runtime_error("Cell::cellDivision: edges are not well defined");
                    }
                }

                for(int i =0; i < polydata_2->GetNumberOfCells(); i++)
                {
                    if(find(removedCells.begin(), removedCells.end(), i) == removedCells.end())
                        cells->InsertNextCell(polydata_2->GetCell(i));
                }
                
                polydata_2 = vtkSmartPointer<vtkPolyData>::New();
                polydata_2->SetPoints(points);
                polydata_2->SetPolys(cells);

            }
            
            //Append the two meshes
            vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
            appendFilter->AddInputData(polydata_1);
            appendFilter->AddInputData(polydata_2);

            // Remove any duplicate points.
            vtkClean->SetInputConnection(appendFilter->GetOutputPort());
            vtkClean->Update();
            vtkSmartPointer<vtkPolyData> polydata_f = vtkClean->GetOutput();


            vtkSmartPointer<vtkFillHolesFilter> fillHoles = vtkSmartPointer<vtkFillHolesFilter>::New();
            fillHoles->SetInputData(polydata_f);
            fillHoles->Update();
            polydata_f = fillHoles->GetOutput();

            edges = vtkSmartPointer<vtkFeatureEdges>::New();
            edges->SetInputData(polydata_f);
            edges->BoundaryEdgesOn();
            edges->FeatureEdgesOff();
            edges->NonManifoldEdgesOn();
            edges->ManifoldEdgesOff();
            edges->Update();
            if(edges->GetOutput()->GetNumberOfLines()>0)
            {
                cout << "HERE --------------------------------------------------" << endl;
                throw runtime_error("Cell::cellDivision: Non-manifold edges found!");
            }

            cout << "LLEGO AQUI 2" << endl;

            removePoints3Neighbours(polydata_f);
            
            vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
            writer->SetInputData(polydata_f);
            writer->SetFileName("polydata_f.vtk");
            writer->Update();
            
            cout << "normalGenerator" << endl;
            vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
            normalGenerator->SetInputData(polydata_f);
            normalGenerator->ComputePointNormalsOff();
            normalGenerator->ComputeCellNormalsOn();
            normalGenerator->ConsistencyOn();
            normalGenerator->SplittingOff();
            normalGenerator->SetFeatureAngle(180.0);
            normalGenerator->Update();

            cout << "vtkClean" << endl;
            vtkClean->SetInputConnection(normalGenerator->GetOutputPort());
            vtkClean->Update();


            polydata_f = vtkSmartPointer<vtkPolyData>(vtkClean->GetOutput());
                    
            if(_bfType==BasisFunctionType::LoopSubdivision)
            {
                vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
                writer->SetInputData(polydata_f);
                writer->SetFileName("polydata_f.vtk");
                writer->Update();
                writer = vtkSmartPointer<vtkPolyDataWriter>::New();
                writer->SetInputData(polydata_1);
                writer->SetFileName("polydata_1.vtk");
                writer->Update();

                cout << "ENTRO?" << endl;
                vtkSmartPointer<vtkPolyDataAlgorithm> subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
                subdivisionFilter->SetInputData(polydata_f);
                dynamic_cast<vtkLinearSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(1);
                subdivisionFilter->Update();
                polydata_f = subdivisionFilter->GetOutput();
                
                subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
                subdivisionFilter->SetInputData(polydata_1);
                dynamic_cast<vtkLinearSubdivisionFilter *> (subdivisionFilter.GetPointer())->SetNumberOfSubdivisions(1);
                subdivisionFilter->Update();
                polydata_1 = subdivisionFilter->GetOutput();
                
                vtkSmartPointer<vtkCellLocator> loc = vtkSmartPointer<vtkCellLocator>::New();
                loc->SetDataSet(polydata);
                loc->BuildLocator();
                
                vtkSmartPointer<vtkPointLocator> ploc1 = vtkSmartPointer<vtkPointLocator>::New();
                ploc1->SetDataSet(polydata_1);
                ploc1->BuildLocator();

                points = vtkSmartPointer<vtkPoints>::New();
                for(int i = 0; i < polydata_f->GetNumberOfPoints(); i++)
                {
                    double *x = polydata_f->GetPoint(i);
                    double closestpoint[3];
                    vtkIdType cellId;
                    int subId;
                    double dist;
                    loc->FindClosestPoint(x, closestpoint, cellId, subId, dist);
                    
                    int id1 = ploc1->FindClosestPoint(x);
                    double *xcl1 = polydata_1->GetPoint(id1);
                    
                    double dist1 = sqrt((x[0]-xcl1[0]) * (x[0]-xcl1[0]) + (x[1]-xcl1[1]) * (x[1]-xcl1[1]) + (x[2]-xcl1[2]) * (x[2]-xcl1[2]));
                    
                    if(dist1<1.E-10)
                        points->InsertNextPoint(closestpoint);
                    else
                        points->InsertNextPoint(x);
                }

                polydata_f->SetPoints(points);
            }
            
            cout << "LLEGO AQUI 3" << polydata_f->GetPoints()->GetNumberOfPoints() << " " << polydata_f->GetPolys()->GetNumberOfCells() << endl;
            if(n==1)
            {
                int nPts = polydata_f->GetPoints()->GetNumberOfPoints();
                int nElem = polydata_f->GetPolys()->GetNumberOfCells();
                
                mother->_connec.resize(nElem,3);
                for(int i =0; i < nElem; i++)
                {
                    auto c = mother->_connec(i,all);
                    vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();;
                    polydata_f->GetPolys()->GetNextCell(element);
                    
                    if(element->GetNumberOfIds() != 3)
                        throw runtime_error("Cell::cellDivision: The output mesh has elements different from triangles!");
                                
                    for(int k = 0; k < element->GetNumberOfIds(); k++)
                        c(k) = element->GetId(k);
                }
                
                mother->_nodeFields.resize(nPts,3);
                for(int i = 0; i < nPts; i++)
                {
                    array<double,3> x;
                    polydata_f->GetPoints()->GetPoint(i, x.data() );
                    mother->_nodeFields(i,0) = x[0];
                    mother->_nodeFields(i,1) = x[1];
                    mother->_nodeFields(i,2) = x[2];
                }
                
                mother->_bfType = _bfType;
                mother->_nodeFieldNames = _nodeFieldNames;
                mother->_cellFieldNames = _cellFieldNames;

                cout << "UPDATE DAUGHTER" << endl;

                mother->Update();

                mother->_cellFields = _cellFields;

                // int nPts = polydata_f->GetPoints()->GetNumberOfPoints();
                // int nElem = polydata_f->GetPolys()->GetNumberOfCells();
                
                // _connec.resize(nElem,3);
                // for(int i =0; i < nElem; i++)
                // {
                //     auto c = _connec(i,all);
                //     vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();;
                //     polydata_f->GetPolys()->GetNextCell(element);
                    
                //     //Check number of Ids is 3
                //     if(element->GetNumberOfIds() != 3)
                //         throw runtime_error("Cell::cellDivision: The output mesh has elements different from triangles! (nIds="+to_string(element->GetNumberOfIds())+")");
                                
                //     for(int k = 0; k < element->GetNumberOfIds(); k++)
                //         c(k) = element->GetId(k);
                // }

                // _nodeFields.resize(nPts,3);
                // for(int i = 0; i < nPts; i++)
                // {
                //     array<double,3> x;
                //     polydata_f->GetPoints()->GetPoint(i, x.data() );
                //     _nodeFields(i,0) = x[0];
                //     _nodeFields(i,1) = x[1];
                //     _nodeFields(i,2) = x[2];
                // }
                
                // cout << "UPDATE MOTHER" << endl;
                // Update();
                // cout << "FINISHED UPDATE MOTHER" << endl;

            }
            else
            {
                int nPts = polydata_f->GetPoints()->GetNumberOfPoints();
                int nElem = polydata_f->GetPolys()->GetNumberOfCells();
                
                daughter->_connec.resize(nElem,3);
                for(int i =0; i < nElem; i++)
                {
                    auto c = daughter->_connec(i,all);
                    vtkSmartPointer<vtkIdList> element = vtkSmartPointer<vtkIdList>::New();;
                    polydata_f->GetPolys()->GetNextCell(element);
                    
                    if(element->GetNumberOfIds() != 3)
                        throw runtime_error("Cell::cellDivision: The output mesh has elements different from triangles!");
                                
                    for(int k = 0; k < element->GetNumberOfIds(); k++)
                        c(k) = element->GetId(k);
                }
                
                daughter->_nodeFields.resize(nPts,3);
                for(int i = 0; i < nPts; i++)
                {
                    array<double,3> x;
                    polydata_f->GetPoints()->GetPoint(i, x.data() );
                    daughter->_nodeFields(i,0) = x[0];
                    daughter->_nodeFields(i,1) = x[1];
                    daughter->_nodeFields(i,2) = x[2];
                }
                
                daughter->_bfType = _bfType;
                daughter->_nodeFieldNames = _nodeFieldNames;
                daughter->_cellFieldNames = _cellFieldNames;

                cout << "UPDATE DAUGHTER" << endl;

                daughter->Update();

                daughter->_cellFields = _cellFields;
            }
        }
        cout << "SALGO" << endl;

        _nodeFields.resize(mother->_nodeFields.shape()[0], mother->_nodeFields.shape()[1]);
        _nodeFields = mother->_nodeFields;
        _connec.resize(mother->_connec.shape()[0], mother->_connec.shape()[1]);
        _connec = mother->_connec;
        _bfs = mother->_bfs;

        return daughter;

    }
}

        
