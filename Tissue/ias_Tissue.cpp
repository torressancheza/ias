//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <math.h>
#include <iostream>
#include <numeric>
#include <thread>

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>
#include <Epetra_FECrsGraph.h>

#include <Teuchos_ParameterList.hpp>
#include <Isorropia_Epetra.hpp>
#include <Epetra_IntVector.h>
#include <Isorropia_EpetraOrderer.hpp>
#include <Isorropia_EpetraPartitioner.hpp>
#include <Isorropia_EpetraRedistributor.hpp>

#include "ias_Cell.h"
#include "ias_Tissue.h"

namespace ias
{
    void Tissue::Update()
    {
        using namespace std;
                
        int loc_nCells = _cells.size();
        
        _nCellPart.resize(_nParts);
        _offsetPart.resize(_nParts+1);
        
        MPI_Allgather(&loc_nCells, 1, MPI_INT, _nCellPart.data(), 1, MPI_INT, _comm);
        
        int currOffs{};
        for (int p = 0; p < _nParts; p++)
        {
            _offsetPart[p] = currOffs;
            currOffs += _nCellPart[p];
        }
        
        _offsetPart[_nParts] = currOffs;
        _nCells = _offsetPart[_nParts];
        
        if(_nCells < _nParts)
            throw std::runtime_error("Tissue::Update: Number of cells (" + std::to_string(_nCells) + ") is smaller than number of partitions (" + std::to_string(_nParts) + ")");

        try
        {
            _checkCellIds();
        }
        catch (const runtime_error& error)
        {
            string what = error.what();
            throw runtime_error("Tissue::Update: " + what);
        }
        
        _checkFieldNames();
    }

    void Tissue::_checkFieldNames()
    {
        using namespace std;
        
        //[1] Check that all cells have the same fields. FIXME: this is mandatory right now but it could be easily changed
        int firstPart = _cells.size() > 0 ? _myPart : _nParts; //First partition with at least one cell (it could not be zero)
        MPI_Allreduce(MPI_IN_PLACE, &firstPart, 1, MPI_INT, MPI_MIN, _comm);
        
        string cellNodeFieldNames0{};
        string cellCellFieldNames0{};
        vector<char> v_cellNodeFieldNames0;
        vector<char> v_cellCellFieldNames0;
        if(_myPart == firstPart)
        {
            for(int i = 0; i < int(_cells[0]->getNodeFieldNames().size())-1; i++)
                cellNodeFieldNames0 += _cells[0]->getNodeFieldNames()[i] + ",";
            if(_cells[0]->getNodeFieldNames().size()>0)
                cellNodeFieldNames0 += _cells[0]->getNodeFieldNames()[_cells[0]->getNodeFieldNames().size()-1];
            
            for(int i = 0; i < int(_cells[0]->getCellFieldNames().size())-1; i++)
                cellCellFieldNames0 += _cells[0]->getCellFieldNames()[i] + ",";
            if(_cells[0]->getCellFieldNames().size()>0)
                cellCellFieldNames0 += _cells[0]->getCellFieldNames()[_cells[0]->getCellFieldNames().size()-1];
            
            v_cellNodeFieldNames0 = vector<char>(cellNodeFieldNames0.begin(),cellNodeFieldNames0.end());
            v_cellCellFieldNames0 = vector<char>(cellCellFieldNames0.begin(),cellCellFieldNames0.end());
        }
        
        int sizeCellNodeFieldNames0 = v_cellNodeFieldNames0.size();
        MPI_Bcast(&sizeCellNodeFieldNames0, 1, MPI_INT, firstPart, _comm);
        v_cellNodeFieldNames0.resize(sizeCellNodeFieldNames0);
        
        int sizeCellCellFieldNames0 = v_cellCellFieldNames0.size();
        MPI_Bcast(&sizeCellCellFieldNames0, 1, MPI_INT, firstPart, _comm);
        v_cellCellFieldNames0.resize(sizeCellCellFieldNames0);
        
        MPI_Bcast(v_cellNodeFieldNames0.data(), sizeCellNodeFieldNames0, MPI_CHAR, firstPart, _comm);
        MPI_Bcast(v_cellCellFieldNames0.data(), sizeCellCellFieldNames0, MPI_CHAR, firstPart, _comm);

        cellNodeFieldNames0 = string(v_cellNodeFieldNames0.begin(),v_cellNodeFieldNames0.end());
        cellCellFieldNames0 = string(v_cellCellFieldNames0.begin(),v_cellCellFieldNames0.end());

        for(size_t i = 0; i < _cells.size(); i++)
        {
            string cellNodeFieldNames{};
            for(int j = 0; j < int(_cells[i]->getNodeFieldNames().size())-1; j++)
                cellNodeFieldNames += _cells[i]->getNodeFieldNames()[j] + ",";
            if(_cells[i]->getNodeFieldNames().size()>0)
                cellNodeFieldNames += _cells[i]->getNodeFieldNames()[_cells[i]->getNodeFieldNames().size()-1];
            
            if(cellNodeFieldNames != cellNodeFieldNames0)
                throw runtime_error("The nodal fields defined in cell " + to_string(getGlobalIdx(i)) + " (" + cellNodeFieldNames + ") do not coincide with those in cell " + to_string(getGlobalIdx(0)) + " (" + cellNodeFieldNames0+ ").");
            
            string cellCellFieldNames{};
            for(int j = 0; j < int(_cells[i]->getCellFieldNames().size())-1; j++)
                cellCellFieldNames += _cells[i]->getCellFieldNames()[j] + ",";
            if(_cells[i]->getCellFieldNames().size()>0)
                cellCellFieldNames += _cells[i]->getCellFieldNames()[_cells[i]->getCellFieldNames().size()-1];
            
            if(cellCellFieldNames != cellCellFieldNames0)
                throw runtime_error("The global fields defined in cell " + to_string(getGlobalIdx(i)) + " (" + cellCellFieldNames + ") do not coincide with those in cell " + to_string(getGlobalIdx(0)) + " (" + cellCellFieldNames0 + ").");
        }
        
        std::istringstream ss_n(cellNodeFieldNames0);
        std::istringstream ss_g(cellCellFieldNames0);
        std::string token;

        _nodeFieldNames.clear();
        while(std::getline(ss_n, token, ','))
            _nodeFieldNames.push_back(token);
        _cellFieldNames.clear();
        while(std::getline(ss_g, token, ','))
            _cellFieldNames.push_back(token);
        
        _mapNodeFieldNames.clear();
        for(size_t i = 0; i < _nodeFieldNames.size(); i++)
            _mapNodeFieldNames[_nodeFieldNames[i]] = i;
        
        _mapCellFieldNames.clear();
        for(size_t i = 0; i < _cellFieldNames.size(); i++)
            _mapCellFieldNames[_cellFieldNames[i]] = i;
        
        _mapTissFieldNames.clear();
        for(size_t i = 0; i < _tissFieldNames.size(); i++)
            _mapTissFieldNames[_tissFieldNames[i]] = i;      

        //FIXME: is this ok here?
        if( _tissFields.size() != _tissFieldNames.size())
            _tissFields.resize(_tissFieldNames.size());
    }

    void Tissue::calculateCellCellAdjacency(double eps)
    {
        using namespace std;
        using Teuchos::RCP;
        using Teuchos::rcp;
                        
        int loc_nCells = _cells.size();
        
        _eps = eps;
        
        //[1] Calculate local boxes and communicate
        
        std::array<std::array<std::vector<double>,2>,3> loc_boxes;
        
        for(int m = 0; m < 3; m++)
        {
            loc_boxes[m][0].resize(loc_nCells);
            loc_boxes[m][1].resize(loc_nCells);
        }

        for(int n=0; n < loc_nCells; n++)
        {
            for(int m = 0; m < 3; m++)
            {
                loc_boxes[m][0][n] = _cells[n]->_nodeFields(0,m);
                loc_boxes[m][1][n] = _cells[n]->_nodeFields(0,m);
            }
            
            for(int i = 1; i < _cells[n]->_nodeFields.shape()[0]; i++)
            {
                for(int m = 0; m < 3; m++)
                {
                    if (loc_boxes[m][0][n] > _cells[n]->_nodeFields(i,m))
                        loc_boxes[m][0][n] = _cells[n]->_nodeFields(i,m);
                    if (loc_boxes[m][1][n] < _cells[n]->_nodeFields(i,m))
                        loc_boxes[m][1][n] = _cells[n]->_nodeFields(i,m);
                }
            }
        }
        
        std::array<std::array<std::vector<double>,2>,3> boxes;
        for(int m = 0; m < 3; m++)
        {
            boxes[m][0].resize(_nCells);
            MPI_Allgatherv(loc_boxes[m][0].data(), loc_nCells, MPI_DOUBLE, boxes[m][0].data(), _nCellPart.data(), _offsetPart.data(), MPI_DOUBLE, _comm);
            boxes[m][1].resize(_nCells);
            MPI_Allgatherv(loc_boxes[m][1].data(), loc_nCells, MPI_DOUBLE, boxes[m][1].data(), _nCellPart.data(), _offsetPart.data(), MPI_DOUBLE, _comm);
        }
        
        //[2] Check the graph, has it changed? (equal = 1)

        int equal{1};
        if(_adjyCC == Teuchos::null)
        {
            equal = 0;
        }
        else
        {
            for (int glo_n = 0; glo_n < _nCells; glo_n++)
            {
                for (int glo_m = glo_n+1; glo_m < _nCells; glo_m++)
                {

                    if( (min(boxes[0][1][glo_n],boxes[0][1][glo_m]) > max(boxes[0][0][glo_n],boxes[0][0][glo_m])-eps) and
                        (min(boxes[1][1][glo_n],boxes[1][1][glo_m]) > max(boxes[1][0][glo_n],boxes[1][0][glo_m])-eps) and
                        (min(boxes[2][1][glo_n],boxes[2][1][glo_m]) > max(boxes[2][0][glo_n],boxes[2][0][glo_m])-eps) )
                    {

                        int found = 0;

                        
                        int part_1 = whichPart(glo_n);
                        int part_2 = whichPart(glo_m);
                        

                        if(part_1==_myPart)
                        {
                            int  nNN{};
                            int* nbors;
                            _adjyCC->ExtractMyRowView(getLocalIdx(glo_n), nNN, nbors);

                            for(int i = 0; i < nNN; i++)
                            {
                                int g_m = _adjyCC->ColMap().GID(nbors[i]);
                                if(g_m==glo_m)
                                {
                                    found = 1;
                                    break;
                                }
                            }
                        }
                        else if (part_2==_myPart)
                        {
                            int  nNN{};
                            int* nbors;
                            _adjyCC->ExtractMyRowView(getLocalIdx(glo_m), nNN, nbors);

                            for(int i = 0; i < nNN; i++)
                            {
                                int g_n = _adjyCC->ColMap().GID(nbors[i]);
                                if(g_n==glo_n)
                                {
                                    found = 1;
                                    break;
                                }
                            }
                        }
                        
                        MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, _comm);
                        
                        if(found == 0)
                        {
                            equal = 0;
                            glo_n = _nCells;
                            glo_m = _nCells;
                        }

                    }
                }
            }
        
            if ( equal == 1)
            {
                for (size_t n = 0; n < _cells.size(); n++)
                {
                    int glo_n = getGlobalIdx(n);
                    
                    int  nNN{};
                    int* nbors;
                    _adjyCC->ExtractMyRowView(n, nNN, nbors);

                    for(int i = 0; i < nNN; i++)
                    {
                        int glo_m = _adjyCC->ColMap().GID(nbors[i]);
                        
                        if( not ( (min(boxes[0][1][glo_n],boxes[0][1][glo_m]) > max(boxes[0][0][glo_n],boxes[0][0][glo_m])-eps) and
                                (min(boxes[1][1][glo_n],boxes[1][1][glo_m]) > max(boxes[1][0][glo_n],boxes[1][0][glo_m])-eps) and
                                (min(boxes[2][1][glo_n],boxes[2][1][glo_m]) > max(boxes[2][0][glo_n],boxes[2][0][glo_m])-eps) ) )
                        {
                            equal = 0;
                            glo_n = _nCells;
                            i = nNN;
                        }
                    }
                }
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, &equal, 1, MPI_INT, MPI_MIN, _comm);
        
        //[3] If the graph has changed, create the new graph
        
        if (equal == 0)
        {
            Epetra_Map vec_map(-1, loc_nCells, 0, Epetra_MpiComm(_comm));
            RCP<Epetra_FECrsGraph> new_adjyCC = rcp(new Epetra_FECrsGraph(Copy, vec_map, 10));

            vector<pair<int,int>> remain_iterList;
            vector<pair<int,int>> remain_iterPartList;

            vector<int> remaining(_nParts);
            
            _inters.clear();
            _inters.resize(_nParts);
            int nInter{};
            
            for (int glo_n = 0; glo_n < _nCells; glo_n++)
            {
                for (int glo_m = glo_n+1; glo_m < _nCells; glo_m++)
                {

                    if( (min(boxes[0][1][glo_n],boxes[0][1][glo_m]) > max(boxes[0][0][glo_n],boxes[0][0][glo_m])-eps) and
                        (min(boxes[1][1][glo_n],boxes[1][1][glo_m]) > max(boxes[1][0][glo_n],boxes[1][0][glo_m])-eps) and
                        (min(boxes[2][1][glo_n],boxes[2][1][glo_m]) > max(boxes[2][0][glo_n],boxes[2][0][glo_m])-eps) )
                    {
                        
                        int part_1 = whichPart(glo_n);
                        int part_2 = whichPart(glo_m);
                        
                        if(part_1==part_2)
                        {
                            if(part_1==_myPart)
                            {
                                new_adjyCC->InsertGlobalIndices(1, &glo_n, 1, &glo_m);
                            }
                            
                            _inters[part_1].push_back({glo_n,glo_m});

                            nInter++;
                        }
                        else
                        {
                            remain_iterList.push_back({glo_n,glo_m});
                            remain_iterList.push_back({glo_m,glo_n});

                            remain_iterPartList.push_back({part_1,part_2});
                            remain_iterPartList.push_back({part_2,part_1});
                            
                            remaining[part_1]++;
                            remaining[part_2]++;
                            
                            nInter++;
                        }
                    }
                }
            }
            
            while(remain_iterPartList.size()>0)
            {
                
                vector<pair<double,int>> mark(_nParts);
                for(int i = 0; i < _nParts; i++)
                {
                    mark[i].first  = _inters[i].size()+remaining[i]/2.0;
                    mark[i].second = i;
                    
                }
                std::sort(mark.begin(), mark.end());

                for(int i = 0; i<_nParts; i++)
                {
                    bool b{false};

                    for(int j = _nParts-1; j>=0; j--)
                    {
                        auto f = std::find(remain_iterPartList.begin(),remain_iterPartList.end(),make_pair(mark[i].second,mark[j].second));
                        
                        if(f!=remain_iterPartList.end())
                        {
                            
                            int c1 = remain_iterList[f-remain_iterPartList.begin()].first;
                            int c2 = remain_iterList[f-remain_iterPartList.begin()].second;
                            
                            if(mark[i].second==_myPart)
                                new_adjyCC->InsertGlobalIndices(1, &c1, 1, &c2);

                            _inters[mark[i].second].push_back({c1,c2});

                            remain_iterList.erase(f-remain_iterPartList.begin()+remain_iterList.begin());

                            remaining[mark[i].second] --;
                            remaining[mark[j].second] --;
                            
                            
                            remain_iterPartList.erase(f);
                            f = std::find(remain_iterList.begin(),remain_iterList.end(),make_pair(c2,c1));
                            remain_iterList.erase(f);
                            remain_iterPartList.erase(f-remain_iterList.begin()+remain_iterPartList.begin());

                            b=true;
                            
                            break;
                        }
                    }
                    
                    if(b)  break;
                }
            }
            
            MPI_Barrier(_comm);
            
            new_adjyCC->FillComplete();
            _adjyCC = new_adjyCC;
            
            
            //[4] We have distributed the interactions in a (presumably) intelligent way
            //    but there may be some inherent imbalances. Here we move interactions
            //    to avoid these imbalances. We put interactions in partitions that may not
            //    contain either of the cells involved in them; this increases the amount
            //    of comunication but decreases the waiting time of each core (which has
            //    a higher computational impact in our problems)

            int base_nInter  = nInter / _nParts;
            int remain_nIter = nInter - base_nInter * _nParts;
            
            //Optimal number of partitions with base_nInter and base_nInter+1 interactions
            int nOpBase_nInter{_nParts-remain_nIter};
            int nOpBase_nInter1{remain_nIter};
            
            int nBase_nInter{};
            int nBase_nInter1{};
            for(int i = 0; i < _nParts; i++)
            {
                if(int(_inters[i].size()) == base_nInter)
                    nBase_nInter ++;
                else if (int(_inters[i].size()) == base_nInter+1)
                    nBase_nInter1++;
            }
            bool  fin = (nBase_nInter==nOpBase_nInter) and (nBase_nInter1==nOpBase_nInter1);

            while(not fin)
            {

                auto maxInterPart = max_element(_inters.begin(),_inters.end(),[](std::vector<std::pair<int,int>> a, std::vector<std::pair<int,int>> b) {return a.size() < b.size();});
                auto minInterPart = min_element(_inters.begin(),_inters.end(),[](std::vector<std::pair<int,int>> a, std::vector<std::pair<int,int>> b) {return a.size() < b.size();});

                _inters[minInterPart-_inters.begin()].push_back(_inters[maxInterPart-_inters.begin()][_inters[maxInterPart-_inters.begin()].size()-1]);
                _inters[maxInterPart-_inters.begin()].erase(_inters[maxInterPart-_inters.begin()].end()-1);
                
                if(int(_inters[minInterPart-_inters.begin()].size()) == base_nInter)
                {
                    nBase_nInter++;
                }
                else if (int(_inters[minInterPart-_inters.begin()].size()) == base_nInter+1)
                {
                    nBase_nInter1++;
                    nBase_nInter--;
                }
                if(int(_inters[maxInterPart-_inters.begin()].size()) == base_nInter)
                {
                    nBase_nInter++;
                    nBase_nInter1--;
                }
                else if(int(_inters[maxInterPart-_inters.begin()].size()) == base_nInter+1)
                {
                    nBase_nInter1++;
                }
                
                fin = (nBase_nInter==nOpBase_nInter) and (nBase_nInter1==nOpBase_nInter1);
            }
            
            _genGhostImportersExporters();
            updateGhosts(true,true,true);
        }
    }

    void Tissue::balanceDistribution()
    {
        using namespace std;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        //------------------------------------------------------
        //BALANCE CELL DISTRIBUTION BASED ON CELL-CELL CONTACTS
        double imbalaceTolerance = 0.05; //FIXME: user parameter
        // Get imbalance tolerance
        string ImbTol = to_string(1 + imbalaceTolerance);

        // Set parameters for the partitioning method
        Teuchos::ParameterList params;
        params.set("PARTITIONING METHOD", "GRAPH");
        params.set("IMBALANCE TOL", ImbTol);
        Teuchos::ParameterList &sublist = params.sublist("Zoltan");
        sublist.set("LB_APPROACH", "PARTITION");
        sublist.set("DEBUG_LEVEL", "0");

        RCP<const Epetra_CrsGraph> c_ptr = _adjyCC;
        RCP<Isorropia::Epetra::Partitioner> part = rcp(new Isorropia::Epetra::Partitioner(c_ptr, params));
        auto epetraMap = *part->createNewMap();
        int loc_newNCells = epetraMap.NumMyElements();
        
        //Use the ghost structure to send and receive information
        Epetra_Map vec_map(-1, loc_newNCells, 0, Epetra_MpiComm(_comm));
        RCP<Epetra_FECrsGraph> adjyCC = rcp(new Epetra_FECrsGraph(Copy, vec_map, 10));
        
        for(int i = 0; i < loc_newNCells; i++)
        {
            int oldi = epetraMap.GID(i);
            adjyCC->InsertGlobalIndices(1, &i, 1, &oldi);
        }
        _adjyCC = adjyCC;
        
        _genGhostImportersExporters();
        updateGhosts(true,true,true);
        calculateCellCellAdjacency(_eps);
    }

    void Tissue::_genGhostImportersExporters()
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;
         
        //Calculate first the partition and the label of the ghosts (and put them in a vector of pairs structure for sorting)
        vector<pair<int,int>> partAndLabel;
        for(auto inter: _inters[_myPart])
        {
            if(!isItemInPart(inter.first))
                partAndLabel.push_back({whichPart(inter.first),inter.first});
            if(!isItemInPart(inter.second))
                partAndLabel.push_back({whichPart(inter.second),inter.second});
        }

        //Sort it (first cells of first partition and so on)
        sort(partAndLabel.begin(),partAndLabel.end());
        partAndLabel.erase( unique( partAndLabel.begin(), partAndLabel.end() ), partAndLabel.end() ); //FIXME: I don't knoe if this is very efficient

        //Count how many come from the different partitions and create the _IpartInGhosts structure
        std::vector<int> InNGhosts(_nParts);
        _inGhostLabels.clear();
        for(auto p: partAndLabel)
        {
            _inGhostLabels.emplace_back(p.second);
            InNGhosts[p.first]++;
        }
        _inGhostOffsetPart.clear();
        _inGhostOffsetPart.push_back(0);
        for(int n=0; n < _nParts; n++)
            _inGhostOffsetPart.push_back(_inGhostOffsetPart[n] + InNGhosts[n]);
        
        vector<int> OutNGhosts(_nParts);
        MPI_Alltoall(InNGhosts.data(), 1, MPI_INT, OutNGhosts.data(), 1, MPI_INT, _comm);
        
        _outGhostOffsetPart.clear();
        _outGhostOffsetPart.push_back(0);
        for(int n=0; n < _nParts; n++)
            _outGhostOffsetPart.push_back(_outGhostOffsetPart[n] + OutNGhosts[n]);
        
        //Pass labels
        _outGhostLabels.resize(_outGhostOffsetPart[_nParts]);
        MPI_Alltoallv(_inGhostLabels.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _outGhostLabels.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, _comm);
   
        //Pass info with number of DOFs etc
        std::vector<int>    OutGhostnDOFs;
        std::vector<int>    OutGhostnglobFields;
        std::vector<int>    OutGhostnPts;
        std::vector<int>    OutGhostnElem;
        std::vector<int>    OutGhostnVert;
        std::vector<int>    OutGhostBFType;
        for(auto l: _outGhostLabels)
        {
            int locl = getLocalIdx(l);
            OutGhostnDOFs.push_back(_cells[locl]->getNumberOfNodeFields());
            OutGhostnglobFields.push_back(_cells[locl]->getNumberOfCellFields());
            OutGhostnPts.push_back(_cells[locl]->getNumberOfPoints());
            OutGhostnElem.push_back(_cells[locl]->getNumberOfElements());
            OutGhostnVert.push_back(_cells[locl]->getNumberOfVerticesPerElement());
            OutGhostBFType.push_back(int(_cells[locl]->getBasisFunctionType()));
        }
        std::vector<int>    InGhostnDOFs(_inGhostOffsetPart[_nParts]);
        std::vector<int>    InGhostnAuxF(_inGhostOffsetPart[_nParts]);
        std::vector<int>    InGhostnglobFields(_inGhostOffsetPart[_nParts]);
        std::vector<int>    InGhostnPts(_inGhostOffsetPart[_nParts]);
        std::vector<int>    InGhostnElem(_inGhostOffsetPart[_nParts]);
        std::vector<int>    InGhostnVert(_inGhostOffsetPart[_nParts]);;
        std::vector<int>    InGhostBFType(_inGhostOffsetPart[_nParts]);;

        MPI_Alltoallv(OutGhostnDOFs.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostnDOFs.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);
        MPI_Alltoallv(OutGhostnglobFields.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostnglobFields.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);
        MPI_Alltoallv(OutGhostnPts.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostnPts.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);
        MPI_Alltoallv(OutGhostnElem.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostnElem.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);
        MPI_Alltoallv(OutGhostnVert.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostnVert.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);
        MPI_Alltoallv(OutGhostBFType.data(), OutNGhosts.data(), _outGhostOffsetPart.data(), MPI_INT, InGhostBFType.data(), InNGhosts.data(), _inGhostOffsetPart.data(), MPI_INT, _comm);


        //Create exporters
        _outGhost_nodeFields_offsetCells.resize(_outGhostLabels.size()+1);
        _outGhost_cellFields_offsetCells.resize(_outGhostLabels.size()+1);
        _outGhost_connec_offsetCells.resize(_outGhostLabels.size()+1);
        _outGhost_nodeFields_offsetCells[0] = _outGhost_cellFields_offsetCells[0] = _outGhost_connec_offsetCells[0] = 0;
        for(int i = 0; i < _outGhostOffsetPart[_nParts]; i++)
        {
            _outGhost_nodeFields_offsetCells[i+1] = _outGhost_nodeFields_offsetCells[i] + OutGhostnPts[i] * OutGhostnDOFs[i];
            _outGhost_cellFields_offsetCells[i+1] = _outGhost_cellFields_offsetCells[i] + OutGhostnglobFields[i];
            _outGhost_connec_offsetCells[i+1] = _outGhost_connec_offsetCells[i] + OutGhostnElem[i] * OutGhostnVert[i];
        }
        
        _outGhost_nodeFields_offsetParts.resize(_nParts+1);
        _outGhost_cellFields_offsetParts.resize(_nParts+1);
        _outGhost_connec_offsetParts.resize(_nParts+1);
        _outGhost_nodeFields_offsetParts[0] = _outGhost_cellFields_offsetParts[0] = _outGhost_connec_offsetParts[0] = 0;
        for(int i = 0; i < _nParts; i++ )
        {
            _outGhost_nodeFields_offsetParts[i+1] = _outGhost_nodeFields_offsetCells[_outGhostOffsetPart[i+1]];
            _outGhost_cellFields_offsetParts[i+1] = _outGhost_cellFields_offsetCells[_outGhostOffsetPart[i+1]];
            _outGhost_connec_offsetParts[i+1]     = _outGhost_connec_offsetCells[_outGhostOffsetPart[i+1]];
        }
        
        _outGhost_nodeFields_countParts.resize(_nParts);
        _outGhost_cellFields_countParts.resize(_nParts);
        _outGhost_connec_countParts.resize(_nParts);
        for(int i = 0; i < _nParts; i++ )
        {
            _outGhost_nodeFields_countParts[i] = _outGhost_nodeFields_offsetParts[i+1] - _outGhost_nodeFields_offsetParts[i];
            _outGhost_cellFields_countParts[i] = _outGhost_cellFields_offsetParts[i+1] - _outGhost_cellFields_offsetParts[i];
            _outGhost_connec_countParts[i]    =  _outGhost_connec_offsetParts[i+1] - _outGhost_connec_offsetParts[i];
        }

        
        _outGhost_nodeFields.resize(_outGhost_nodeFields_offsetCells[_outGhostLabels.size()]);
        _outGhost_nodeFields0.resize(_outGhost_nodeFields_offsetCells[_outGhostLabels.size()]);
        _outGhost_cellFields.resize(_outGhost_cellFields_offsetCells[_outGhostLabels.size()]);
        _outGhost_cellFields0.resize(_outGhost_cellFields_offsetCells[_outGhostLabels.size()]);
        _outGhost_connec.resize(_outGhost_connec_offsetCells[_outGhostLabels.size()]);

        //Create importers
        //Dimensionalise tensors for incoming and outgoing data
        _inGhost_nodeFields_offsetCells.resize(_inGhostLabels.size()+1);
        _inGhost_cellFields_offsetCells.resize(_inGhostLabels.size()+1);
        _inGhost_connec_offsetCells.resize(_inGhostLabels.size()+1);
        _inGhost_nodeFields_offsetCells[0] = _inGhost_cellFields_offsetCells[0] = _inGhost_connec_offsetCells[0] = 0;
        for(int i = 0; i < _inGhostOffsetPart[_nParts]; i++)
        {
            _inGhost_nodeFields_offsetCells[i+1] = _inGhost_nodeFields_offsetCells[i] + InGhostnPts[i] * InGhostnDOFs[i];
            _inGhost_cellFields_offsetCells[i+1] = _inGhost_cellFields_offsetCells[i] + InGhostnglobFields[i];
            _inGhost_connec_offsetCells[i+1] = _inGhost_connec_offsetCells[i] + InGhostnElem[i] * InGhostnVert[i];
        }
        
        _inGhost_nodeFields_offsetParts.resize(_nParts+1);
        _inGhost_cellFields_offsetParts.resize(_nParts+1);
        _inGhost_connec_offsetParts.resize(_nParts+1);
        _inGhost_nodeFields_offsetParts[0] = _inGhost_cellFields_offsetParts[0] = _inGhost_connec_offsetParts[0] = 0;
        for(int i = 0; i < _nParts; i++ )
        {
            _inGhost_nodeFields_offsetParts[i+1] = _inGhost_nodeFields_offsetCells[_inGhostOffsetPart[i+1]];
            _inGhost_cellFields_offsetParts[i+1] = _inGhost_cellFields_offsetCells[_inGhostOffsetPart[i+1]];
            _inGhost_connec_offsetParts[i+1] = _inGhost_connec_offsetCells[_inGhostOffsetPart[i+1]];
        }
        
        _inGhost_nodeFields_countParts.resize(_nParts);
        _inGhost_cellFields_countParts.resize(_nParts);
        _inGhost_connec_countParts.resize(_nParts);
        for(int i = 0; i < _nParts; i++ )
        {
            _inGhost_nodeFields_countParts[i] = _inGhost_nodeFields_offsetParts[i+1] - _inGhost_nodeFields_offsetParts[i];
            _inGhost_cellFields_countParts[i] = _inGhost_cellFields_offsetParts[i+1] - _inGhost_cellFields_offsetParts[i];
            _inGhost_connec_countParts[i] = _inGhost_connec_offsetParts[i+1] - _inGhost_connec_offsetParts[i];
        }
        
        
        _inGhost_nodeFields.resize(_inGhost_nodeFields_offsetCells[_inGhostLabels.size()]);
        _inGhost_nodeFields0.resize(_inGhost_nodeFields_offsetCells[_inGhostLabels.size()]);
        _inGhost_cellFields.resize(_inGhost_cellFields_offsetCells[_inGhostLabels.size()]);
        _inGhost_cellFields0.resize(_inGhost_cellFields_offsetCells[_inGhostLabels.size()]);
        _inGhost_connec.resize(_inGhost_connec_offsetCells[_inGhostLabels.size()]);

        
        //Create ghosts pointing to the right data (so that they can be used as normal cells)
        _inGhostCells.resize(_inGhostLabels.size());
        for(int i = 0; i < int(_inGhostLabels.size()); i++)
        {
            _inGhostCells[i] = rcp(new Cell);
            
            _inGhostCells[i]->_nodeFields.set_pointer(&_inGhost_nodeFields[_inGhost_nodeFields_offsetCells[i]]);
            _inGhostCells[i]->_nodeFields.resize(InGhostnPts[i], InGhostnDOFs[i]);

            _inGhostCells[i]->_cellFields.set_pointer(&_inGhost_cellFields[_inGhost_cellFields_offsetCells[i]]);
            _inGhostCells[i]->_cellFields.resize(InGhostnglobFields[i]);
            
            _inGhostCells[i]->_connec.set_pointer(&_inGhost_connec[_inGhost_connec_offsetCells[i]]);
            _inGhostCells[i]->_connec.resize(InGhostnElem[i],InGhostnVert[i]);

            _inGhostCells[i]->_cellFieldNames = _nodeFieldNames;
            _inGhostCells[i]->_cellFieldNames = _cellFieldNames;
            
            _inGhostCells[i]->_bfType = static_cast<BasisFunctionType>(InGhostBFType[i]);
        }
    }

    void Tissue::updateGhosts(bool updateNodeFields, bool updateGlobalFields, bool updateConnectivity)
    {
        for(int i = 0; i < int(_outGhostLabels.size()); i++)
        {
            int loc  = getLocalIdx(_outGhostLabels[i]);
            int nPts = _cells[loc]->getNumberOfPoints();
            int nElem = _cells[loc]->getNumberOfElements();
            int nVert = 3;

            //Copy the information from to export
            if(updateNodeFields)
            {
            std::memcpy(&_outGhost_nodeFields[_outGhost_nodeFields_offsetCells[i]],_cells[loc]->_nodeFields.data(),(nPts*_cells[loc]->getNumberOfNodeFields())*sizeof(double));
            }
            if(updateGlobalFields)
            {
                std::memcpy(&_outGhost_cellFields[_outGhost_cellFields_offsetCells[i]],_cells[loc]->_cellFields.data(),(_cells[loc]->getNumberOfCellFields())*sizeof(double));
            }
            
            if(updateConnectivity)
            {
                std::memcpy(&_outGhost_connec[_outGhost_connec_offsetCells[i]],_cells[loc]->_connec.data(),(nElem*nVert)*sizeof(int));
                _cells[loc]->Update();
            }

        }
        
        //Import it
        if(updateNodeFields)
        {
            MPI_Alltoallv(_outGhost_nodeFields.data(), _outGhost_nodeFields_countParts.data(), _outGhost_nodeFields_offsetParts.data(), MPI_DOUBLE, _inGhost_nodeFields.data(), _inGhost_nodeFields_countParts.data(), _inGhost_nodeFields_offsetParts.data(), MPI_DOUBLE, _comm);
            MPI_Alltoallv(_outGhost_nodeFields0.data(), _outGhost_nodeFields_countParts.data(), _outGhost_nodeFields_offsetParts.data(), MPI_DOUBLE, _inGhost_nodeFields0.data(), _inGhost_nodeFields_countParts.data(), _inGhost_nodeFields_offsetParts.data(), MPI_DOUBLE, _comm);
        }
        if(updateGlobalFields)
        {
            MPI_Alltoallv(_outGhost_cellFields.data(), _outGhost_cellFields_countParts.data(), _outGhost_cellFields_offsetParts.data(), MPI_DOUBLE, _inGhost_cellFields.data(), _inGhost_cellFields_countParts.data(), _inGhost_cellFields_offsetParts.data(), MPI_DOUBLE, _comm);
            MPI_Alltoallv(_outGhost_cellFields0.data(), _outGhost_cellFields_countParts.data(), _outGhost_cellFields_offsetParts.data(), MPI_DOUBLE, _inGhost_cellFields0.data(), _inGhost_cellFields_countParts.data(), _inGhost_cellFields_offsetParts.data(), MPI_DOUBLE, _comm);
        }
        if(updateConnectivity)
        {
            MPI_Alltoallv(_outGhost_connec.data(), _outGhost_connec_countParts.data(), _outGhost_connec_offsetParts.data(), MPI_INT, _inGhost_connec.data(), _inGhost_connec_countParts.data(), _inGhost_connec_offsetParts.data(), MPI_INT, _comm);
            
            for(auto c: _inGhostCells)
                c->Update();
        }
    }

    size_t Tissue::_checkCellIds()
    {
        using namespace std;

        vector<int> loc_cellId;
        vector<int> glo_cellId(_nCells);
        
        for(auto c: _cells)
            loc_cellId.push_back(c->getCellField("cellId")+0.5);
        
        MPI_Allgatherv(loc_cellId.data(), loc_cellId.size(), MPI_INT, glo_cellId.data(), _nCellPart.data(), _offsetPart.data(), MPI_INT, _comm);
        
        size_t maxCellId{};
        for(auto id: glo_cellId)
        {
            if(count(glo_cellId.begin(),glo_cellId.end(),id) > 1)
                throw runtime_error("The cell id " + to_string(id) + " is repeated several times.");
            if(int(maxCellId) < id)
                maxCellId = id;
        }
        
        return maxCellId;
    }
}
