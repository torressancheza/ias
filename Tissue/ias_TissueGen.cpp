//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <numeric>

#include <vtkSmartPointer.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkCleanPolyData.h>


#include "ias_LinearElements.h"
#include "ias_LoopSubdivision.h"
#include "ias_TissueGen.h"

namespace ias
{

    void TissueGen::_checkFieldNames()
    {
        using namespace std;


        if(std::find(_nodeFieldNames.begin(),_nodeFieldNames.end(),"x") != _nodeFieldNames.end())
            throw runtime_error("x is already a node field; you do not need to set it.");

        if(std::find(_nodeFieldNames.begin(),_nodeFieldNames.end(),"y") != _nodeFieldNames.end())
            throw runtime_error("y is already a node field; you do not need to set it.");
        
        if(std::find(_nodeFieldNames.begin(),_nodeFieldNames.end(),"z") != _nodeFieldNames.end())
            throw runtime_error("z is already a node field; you do not need to set it.");
        
        if(std::find(_cellFieldNames.begin(),_cellFieldNames.end(),"cellId") != _cellFieldNames.end())
            throw runtime_error("cellId is already a cell field; you do not need to set it.");

        //Check names (here we just check if names are repeated because Tissue::Update will check if names in different partitions are a match)
        for(auto& f: _nodeFieldNames)
        {
            if(std::count(_nodeFieldNames.begin(), _nodeFieldNames.end(), f) > 1)
                throw runtime_error("Node field " + f + " has been set twice.");
        }
        for(auto& f: _cellFieldNames)
        {
            if(std::count(_cellFieldNames.begin(), _cellFieldNames.end(), f) > 1)
                throw runtime_error("Cell field " + f + " has been set twice.");
        }
        for(auto& f: _tissFieldNames)
        {
            if(std::count(_tissFieldNames.begin(), _tissFieldNames.end(), f) > 1)
                throw runtime_error("Tissue field " + f + " has been set twice.");
        }
    }

    Teuchos::RCP<Tissue> TissueGen::genRegularGridSpheres(int nx, int ny, int nz, double deltax, double deltay, double deltaz, double r, int nSubdiv)
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;

        try
        {
            _checkFieldNames();
        }
        catch (const runtime_error& error)
        {
            string what = error.what();
            throw runtime_error("TissueGen::genRegularGridSphere: " + what);
        }
        
        RCP<Tissue> tissue = rcp(new Tissue);

        tissue->_nCells = nx * ny * nz;

        int loc_nCells{};
        int baseLocItem = tissue->_nCells / tissue->_nParts;
        int remain_nItem = tissue->_nCells - baseLocItem * tissue->_nParts;
        int currOffs = 0;
        for (int p = 0; p < tissue->_nParts; p++)
        {
            int p_locnitem = (p < remain_nItem) ? (baseLocItem + 1) : (baseLocItem);

            tissue->_nCellPart[p]  = p_locnitem;
            tissue->_offsetPart[p] = currOffs;

            currOffs += p_locnitem;
        }
        tissue->_offsetPart[tissue->_nParts] = tissue->_nCells;
        loc_nCells = tissue->_nCellPart[tissue->getMyPart()];

        RCP<Cell> cell = rcp(new Cell);
        cell->generateSphereFromPlatonicSolid(nSubdiv, r);
        cell->setBasisFunctionType(_bfType);
        for(auto f: _nodeFieldNames)
            cell->addNodeField(f);
        for(auto f: _cellFieldNames)
            cell->addCellField(f);
        cell->Update();
                                
        for(int n = 0; n < loc_nCells; n++)
        {
            int glo_n = tissue->getGlobalIdx(n);
            int k =  glo_n / (nx*ny);
            int j = (glo_n-k*nx*ny)/nx;
            int i =  glo_n-(k*ny+j)*nx;

            RCP<Cell> newcell = rcp(new Cell);
            newcell->_connec = cell->_connec;
            newcell->_nodeFields = cell->_nodeFields;
            newcell->_cellFields = cell->_cellFields;
            newcell->_bfType = _bfType;
            newcell->_nodeFieldNames = cell->_nodeFieldNames;
            newcell->_cellFieldNames = cell->_cellFieldNames;
            newcell->Update();
            
            newcell->getCellField("cellId") = glo_n;

            newcell->_nodeFields(all,0) += i * deltax;
            newcell->_nodeFields(all,1) += j * deltay;
            newcell->_nodeFields(all,2) += k * deltaz;

            tissue->_cells.emplace_back(newcell);
        }
        
        tissue->_tissFields.resize(_tissFieldNames.size());
        tissue->_tissFieldNames = _tissFieldNames;
        
        tissue->Update();
        
        return tissue;
    }

    Teuchos::RCP<Tissue> TissueGen::genTripletSpheres(double r, double delta,int nSubdiv)
    {
        //Generates triplet of cells of radius r, whose centers are the points of an equilateral triangle, and so that there
        //is a distance delta between the membranes of two neigbouring cells (distance 2r+delta beetween centers)
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;

        try
        {
            _checkFieldNames();
        }
        catch (const runtime_error& error)
        {
            string what = error.what();
            throw runtime_error("TissueGen::genRegularGridSphere: " + what);
        }
        
        RCP<Tissue> tissue = rcp(new Tissue);

        tissue->_nCells = 3;

        int loc_nCells{};
        int baseLocItem = tissue->_nCells / tissue->_nParts;
        int remain_nItem = tissue->_nCells - baseLocItem * tissue->_nParts;
        int currOffs = 0;
        for (int p = 0; p < tissue->_nParts; p++)
        {
            int p_locnitem = (p < remain_nItem) ? (baseLocItem + 1) : (baseLocItem);

            tissue->_nCellPart[p]  = p_locnitem;
            tissue->_offsetPart[p] = currOffs;

            currOffs += p_locnitem;
        }
        tissue->_offsetPart[tissue->_nParts] = tissue->_nCells;
        loc_nCells = tissue->_nCellPart[tissue->getMyPart()];

        RCP<Cell> cell = rcp(new Cell);
        cell->generateSphereFromPlatonicSolid(nSubdiv, r);
        cell->setBasisFunctionType(_bfType);
        for(auto f: _nodeFieldNames)
            cell->addNodeField(f);
        for(auto f: _cellFieldNames)
            cell->addCellField(f);
        cell->Update();
                                
        for(int n = 0; n < loc_nCells; n++)
        {
            int glo_n = tissue->getGlobalIdx(n);
            double a=2*r+delta; //length of the side of the equilateral triangle
            double z =  0; //z coordinate is zero for every cell
            double y =  (glo_n == 0) ? 0 : a*sqrt(3)/2.0;  //y coordinate: 0 [0], a*sqrt(3)/2 [1 and 2] 
            double x =  (glo_n == 0) ? 0 : (2*glo_n-3)*a*0.5; // x coordinate: 0 [0] -a/2 [1] a/2 [2]

            cout << "Coordinates generated for glo_n= "<< glo_n <<" : " << x << " " << y << " " << z << endl;

            RCP<Cell> newcell = rcp(new Cell);
            newcell->_connec = cell->_connec;
            newcell->_nodeFields = cell->_nodeFields;
            newcell->_cellFields = cell->_cellFields;  
            newcell->_bfType = _bfType;
            newcell->_nodeFieldNames = cell->_nodeFieldNames;
            newcell->_cellFieldNames = cell->_cellFieldNames;
            newcell->Update();
            
            newcell->getCellField("cellId") = glo_n;

            newcell->_nodeFields(all,0) += x;
            newcell->_nodeFields(all,1) += y;
            newcell->_nodeFields(all,2) += z;

            tissue->_cells.emplace_back(newcell);
        }
        
        tissue->_tissFields.resize(_tissFieldNames.size());
        tissue->_tissFieldNames = _tissFieldNames;
        
        tissue->Update();
        
        return tissue;
    }
}
    
