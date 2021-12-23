//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iostream>
#include <numeric>
#include <thread>

#include <Epetra_MpiComm.h>

#include <vtkDistancePolyDataFilter.h>
#include <vtkPolyDataWriter.h>
#include <vtkNew.h>
#include <vtkPointLocator.h>
#include <vtkIdList.h>

#include "ias_Integration.h"
#include "ias_BasicStructures.h"
namespace ias
{

    void Integration::Update()
    {
        using namespace std;
        using Teuchos::rcp;
        
        _checkIntegralNames();
        
        _nodeDOFIdx.clear();
        int n{};
        for(auto s: _nodeDOFNames)
        {
            _nodeDOFIdx.push_back(_tissue->getNodeFieldIdx(s));
            _mapNodeDOFNames[s] = n;
        }
        
        _cellDOFIdx.clear();
        n = 0;
        for(auto s: _cellDOFNames)
        {
            _cellDOFIdx.push_back(_tissue->getCellFieldIdx(s));
            _mapCellDOFNames[s] = n;
        }
        
        int loc_nCells = _tissue->_cells.size();
        
        if(_nodeDOFIdx.size() == 0 and _cellDOFIdx.size() == 0)
            throw runtime_error("No degrees of freedom have been selected for the problem!");
        
        int nNodeDOFs = _nodeDOFIdx.size();
        int nGlobDOFs = _cellDOFIdx.size();
        
        vector<int> loc_cellDOFSize;
        vector<int> glo_cellDOFSize;
        int loc_totSize{};
        for(int n=0; n < loc_nCells;n++)
        {
            int nPts  = _tissue->_cells[n]->getNumberOfPoints();
            loc_cellDOFSize.emplace_back(nNodeDOFs*nPts+nGlobDOFs);
            loc_totSize += loc_cellDOFSize[n];
        }
        
        glo_cellDOFSize.resize(_tissue->_nCells);
        MPI_Allgatherv(loc_cellDOFSize.data(), loc_nCells, MPI_INT, glo_cellDOFSize.data(), _tissue->_nCellPart.data(), _tissue->_offsetPart.data(), MPI_INT, _tissue->_comm);

        _cellDOFOffset.clear();
        _cellDOFOffset.emplace_back(0);
        for(int n = 0; n < _tissue->_nCells; n++)
            _cellDOFOffset.emplace_back(_cellDOFOffset[n]+glo_cellDOFSize[n]);

        // Linear System:
        int numEntriesPerRow = _tissue->_cells[0]->_bfs->getMaxNumberOfNeighbours()*_tissue->_cells[0]->getNumberOfNodeFields();
        numEntriesPerRow *= numEntriesPerRow;
        Epetra_Map vec_map(-1, loc_totSize, 0, Epetra_MpiComm(_tissue->_comm));
        _matrix    = rcp(new Epetra_FECrsMatrix(Copy, vec_map, -1));
        _vector = rcp(new Epetra_FEVector(vec_map));
        _sol = rcp(new Epetra_FEVector(vec_map));
        _linProbl = rcp(new Epetra_LinearProblem(_matrix.getRawPtr(), _sol.getRawPtr(), _vector.getRawPtr()));
        
        if(_iPts_single == 0)
            throw runtime_error("Integration::Update: The number of integration points has not been defined!");
        
        CubatureGauss c1(_iPts_single,2);
        _wSamples_single = c1.getWsamples();
        _savedBFs_single.clear();
        for(auto t: _tissue->_cells[0]->_bfs->getTypes())
        {
            std::vector<std::vector<std::vector<double>>> bftype;
            for(auto k: c1.getXsamples())
            {
                bftype.emplace_back(_tissue->_cells[0]->_bfs->computeBasisFunctionsType(k, t));
            }
            _savedBFs_single.emplace_back(bftype);
        }
        
        if(_iPts_intera == 0)
            throw runtime_error("Integration::Update: The number of integration points has not been defined!");
        
        CubatureGauss c2(_iPts_intera,2);
        _wSamples_intera = c2.getWsamples();
        _savedBFs_intera.clear();
        for(auto t: _tissue->_cells[0]->_bfs->getTypes())
        {
            std::vector<std::vector<std::vector<double>>> bftype;
            for(auto k: c2.getXsamples())
            {
                bftype.emplace_back(_tissue->_cells[0]->_bfs->computeBasisFunctionsType(k, t));
            }
            _savedBFs_intera.emplace_back(bftype);
        }
        
        Epetra_Map vec_map_int(-1, _tissue->_cells.size() * _cellIntegralIdx.size(), 0, Epetra_MpiComm(_tissue->_comm));
        _cellIntegrals = rcp(new Epetra_FEVector(vec_map_int));
        _tissIntegrals.resize(_tissIntegralNames.size());

        //First use AssembleElementalMatrix with create to true
        _assembleElementalMatrix = AssembleElementalMatrix<true>;
    }

    void Integration::computeSingleIntegral()
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;

        if(_singleIntegrand==nullptr)
            return;
        
        Tensor::tensor<double,1> tissIntegrals(_tissIntegralIdx.size());
        tissIntegrals = 0.0;
                
        for(int n=0; n < int(_tissue->_cells.size());n++)
        {
            int glo_n = _tissue->getGlobalIdx(n);
            
            auto& nodeFields   = _tissue->_cells[n]->_nodeFields;
            auto& cellFields   = _tissue->_cells[n]->_cellFields;

            int nElem  = _tissue->_cells[n]->getNumberOfElements();
            int nPts   = _tissue->_cells[n]->getNumberOfPoints();
            int nNodeFields = nodeFields.shape()[1];
            int nCellFields = cellFields.size();
            int nNodeDOFs   = _nodeDOFIdx.size();
            int nCellDOFs   = _cellDOFIdx.size();
            int eNNMax      = _tissue->_cells[n]->_bfs->getMaxNumberOfNeighbours();

            #pragma omp parallel
            {
                std::vector<double> v_inputFields(eNNMax*nNodeFields+nCellFields,0.0);
                std::vector<double> v_outputVector(eNNMax*nNodeDOFs+nCellDOFs,0.0);
                std::vector<double> v_outputMatrix((eNNMax*nNodeDOFs+nCellDOFs)*(eNNMax*nNodeDOFs+nCellDOFs),0.0);
                std::vector<double> v_outputIntegrals(_tissIntegralIdx.size()+_cellIntegralIdx.size(),0.0);
                double* v_nborFields = &v_inputFields[0];
                double* v_cellFields = &v_inputFields[eNNMax*nNodeFields];
                double* v_vec_n  = &v_outputVector[0];
                double* v_vec_g  = &v_outputVector[eNNMax*nNodeDOFs];
                double* v_mat_nn = &v_outputMatrix[0];
                double* v_mat_ng = &v_outputMatrix[eNNMax*nNodeDOFs*eNNMax*nNodeDOFs];
                double* v_mat_gn = &v_outputMatrix[eNNMax*nNodeDOFs*(eNNMax*nNodeDOFs+nCellDOFs)];
                double* v_mat_gg = &v_outputMatrix[eNNMax*nNodeDOFs*(eNNMax*nNodeDOFs+2*nCellDOFs)];

                RCP<SingleIntegralStr> singIntStr = rcp(new SingleIntegralStr);
                singIntStr->cellID     = glo_n;
                singIntStr->nborFields.set_pointer(v_nborFields);
                singIntStr->cellFields.set_pointer(v_cellFields);
                singIntStr->cellFields.resize(nCellFields);
                singIntStr->cellFields = cellFields;
                singIntStr->pDim  = _tissue->_cells[n]->_bfs->getParametricDimension();
                singIntStr->tissFields = _tissue->_tissFields;

                singIntStr->vec_n.set_pointer(v_vec_n);
                singIntStr->vec_c.set_pointer(v_vec_g);
                singIntStr->mat_nn.set_pointer(v_mat_nn);
                singIntStr->mat_nc.set_pointer(v_mat_ng);
                singIntStr->mat_cn.set_pointer(v_mat_gn);
                singIntStr->mat_cc.set_pointer(v_mat_gg);
                singIntStr->mat_cc.resize(nCellDOFs,nCellDOFs);
                
                singIntStr->tissIntegrals.set_pointer(&v_outputIntegrals[0]);
                singIntStr->cellIntegrals.set_pointer(&v_outputIntegrals[_tissIntegralIdx.size()]);
                singIntStr->tissIntegrals.resize(_tissIntegralIdx.size());
                singIntStr->cellIntegrals.resize(_cellIntegralIdx.size());
                
                singIntStr->_nodeDOFNames = _nodeDOFNames;
                singIntStr->_cellDOFNames = _cellDOFNames;
                singIntStr->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr->_mapCellDOFNames = _mapCellDOFNames;
                
                singIntStr->_nodeFieldNames  = _tissue->_nodeFieldNames;
                singIntStr->_cellFieldNames  = _tissue->_cellFieldNames;
                singIntStr->_tissFieldNames  = _tissue->_tissFieldNames;
                singIntStr->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr->_mapCellFieldNames = _tissue->_mapCellFieldNames;
                singIntStr->_mapTissFieldNames = _tissue->_mapTissFieldNames;
                
                singIntStr->_cellIntegralNames = _cellIntegralNames;
                singIntStr->_tissIntegralNames = _tissIntegralNames;
                singIntStr->_mapCellIntegralNames = _mapCellIntegralNames;
                singIntStr->_mapTissIntegralNames = _mapTissIntegralNames;

                singIntStr->userAuxiliaryObjects = userAuxiliaryObjects;

                tensor<double,1> loc_tissIntegrals(_tissIntegralIdx.size());
                tensor<double,1> cellIntegrals(_cellIntegralIdx.size());
                loc_tissIntegrals = 0.0;
                cellIntegrals = 0.0;

                #pragma omp for
                for(int e=0; e < nElem; e++)
                {
                    int   eNN   = _tissue->_cells[n]->_bfs->getNumberOfNeighbours(e);
                    int*  adjEN = _tissue->_cells[n]->_bfs->getNeighbours(e);
                    
                    //Prepare fillStructure for integration
                                    
                    singIntStr->eNN    = int(eNN);
                    singIntStr->elemID = e;
                    singIntStr->nborFields.resize(eNN,nNodeFields);
                    
                    for(int i=0; i < eNN; i++)
                        singIntStr->nborFields(i,all) = nodeFields(adjEN[i],all);

                    singIntStr->vec_n.resize(eNN,nNodeDOFs);
                    singIntStr->vec_c.resize(nCellDOFs);
                    
                    singIntStr->mat_nn.resize(eNN,nNodeDOFs,eNN,nNodeDOFs);
                    singIntStr->mat_nc.resize(eNN,nNodeDOFs,nCellDOFs);
                    singIntStr->mat_cn.resize(nCellDOFs,eNN,nNodeDOFs);
                    
                    std::fill(v_outputVector.begin(),v_outputVector.end(),0.0);
                    std::fill(v_outputMatrix.begin(),v_outputMatrix.end(),0.0);
                    std::fill(v_outputIntegrals.begin(),v_outputIntegrals.end(),0.0);

                    auto& savedBFs_type = _savedBFs_single[_tissue->_cells[n]->_bfs->getElementType(e)];
                    for(size_t k=0; k < _wSamples_single.size(); k++)
                    {
                        singIntStr->bfs = savedBFs_type[k];
                        singIntStr->w_sample = _wSamples_single[k];
                        singIntStr->sampID = k;
                        _singleIntegrand(singIntStr);
                    }

                    loc_tissIntegrals += singIntStr->tissIntegrals;
                    cellIntegrals += singIntStr->cellIntegrals;
                    
                    int dummy{0};
                    int offsetDOFs       = _cellDOFOffset[glo_n];
                    int offsetglobFields = _cellDOFOffset[glo_n]+nPts*nNodeDOFs;

                    #pragma omp critical
                    AssembleElementalVector( offsetDOFs, eNN,  nNodeDOFs,    adjEN, singIntStr->vec_n.data(), _vector);
                    #pragma omp critical
                    AssembleElementalVector(offsetglobFields,   1, nCellDOFs,   &dummy, singIntStr->vec_c.data(), _vector);
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs, offsetDOFs, eNN, eNN,  nNodeDOFs,  nNodeDOFs, adjEN, adjEN, singIntStr->mat_nn.data(), _matrix );
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs,offsetglobFields, eNN,   1,  nNodeDOFs, nCellDOFs, adjEN, &dummy, singIntStr->mat_nc.data(), _matrix);
                    #pragma omp critical
                    _assembleElementalMatrix(offsetglobFields, offsetDOFs,   1, eNN, nCellDOFs,  nNodeDOFs, &dummy, adjEN, singIntStr->mat_cn.data(), _matrix);
                    #pragma omp critical
                    _assembleElementalMatrix(offsetglobFields,offsetglobFields,   1,   1, nCellDOFs, nCellDOFs, &dummy, &dummy, singIntStr->mat_cc.data(), _matrix);
                }
                
                #pragma omp critical
                tissIntegrals += loc_tissIntegrals;

                for(int i = 0; i < int(_cellIntegralIdx.size()); i++)
                {
                    int i1 = glo_n*_cellIntegralIdx.size()+i;
                    #pragma omp critical
                    _cellIntegrals->SumIntoGlobalValues(1, &i1, &cellIntegrals(i));
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, tissIntegrals.data(), tissIntegrals.size(), MPI_DOUBLE, MPI_SUM, _tissue->_comm);
        
        _tissIntegrals = tissIntegrals;
    }

    void Integration::computeDoubleIntegral()
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;

        int dummy{0};

        if(_doubleIntegrand==nullptr)
            return;
        
        Tensor::tensor<double,1> tissIntegrals(_tissIntegralIdx.size());
        tissIntegrals = 0.0;
        
        
        // Epetra_Map vec_map(-1, _tissue->_cells.size() * _cellIntegralIdx.size(), 0, Epetra_MpiComm(_tissue->_comm));
        // Teuchos::RCP<Epetra_FEVector> aux_cellIntegrals = rcp(new Epetra_FEVector(vec_map));
        
        for(size_t inte = 0; inte < _tissue->_inters[_tissue->getMyPart()].size(); inte++)
        {
            if(_elems_inte.size() == 0)
                throw runtime_error("Integration::computeDoubleIntegral: there are no interacting elements. Did you forget to call calculateInteractingElements?");
            
            int glo_n = _tissue->_inters[_tissue->getMyPart()][inte].first;
            int glo_m = _tissue->_inters[_tissue->getMyPart()][inte].second;
            
            if (glo_n == glo_m)
                continue;
            if(_elems_inte[inte].size() == 0)
                continue;
            
            RCP<Cell> cell_1 = _tissue->GetCell(glo_n);
            RCP<Cell> cell_2 = _tissue->GetCell(glo_m);
            
            auto& nodeFields_1 = cell_1->_nodeFields;
            auto& cellFields_1 = cell_1->_cellFields;
            int nPts_1         = cell_1->getNumberOfPoints();
            int nNodeFields_1  = nodeFields_1.shape()[1];
            int nCellFields_1  = cellFields_1.size();
            int eNNMax_1       = cell_1->_bfs->getMaxNumberOfNeighbours();

            auto& nodeFields_2 = cell_2->_nodeFields;
            auto& cellFields_2 = cell_2->_cellFields;
            int nPts_2         = cell_2->getNumberOfPoints();
            int nNodeFields_2  = nodeFields_2.shape()[1];
            int nCellFields_2  = cellFields_2.size();
            int eNNMax_2       = cell_2->_bfs->getMaxNumberOfNeighbours();

            int nNodeDOFs    = _nodeDOFIdx.size();
            int nCellDOFs    = _cellDOFIdx.size();
            
            #pragma omp parallel
            {
                std::vector<double> v_inputFields_1(eNNMax_1*nNodeFields_1+nCellFields_1,0.0);
                std::vector<double> v_outputVector_1(eNNMax_1*nNodeDOFs+nCellDOFs,0.0);
                std::vector<double> v_outputMatrix_1((eNNMax_1*nNodeDOFs+nCellDOFs)*(eNNMax_1*nNodeDOFs+nCellDOFs),0.0);
                std::vector<double> v_outputIntegrals_1(_tissIntegralIdx.size()+_cellIntegralIdx.size(),0.0);

                double* v_nborFields_1 = &v_inputFields_1[0];
                double* v_cellFields_1 = &v_inputFields_1[eNNMax_1*nNodeFields_1];
                double* v_vec_n_1= &v_outputVector_1[0];
                double* v_vec_c_1 = &v_outputVector_1[eNNMax_1*nNodeDOFs];
                double* v_mat_nn_1 = &v_outputMatrix_1[0];
                double* v_mat_nc_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*eNNMax_1*nNodeDOFs];
                double* v_mat_cn_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*(eNNMax_1*nNodeDOFs+nCellDOFs)];
                double* v_mat_cc_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*(eNNMax_1*nNodeDOFs+2*nCellDOFs)];
                
                RCP<SingleIntegralStr> singIntStr_1 = rcp(new SingleIntegralStr);
                singIntStr_1->cellID = glo_n;
                singIntStr_1->pDim   = cell_1->_bfs->getParametricDimension();
                singIntStr_1->nborFields.set_pointer(v_nborFields_1);
                singIntStr_1->cellFields.set_pointer(v_cellFields_1);
                singIntStr_1->vec_n.set_pointer(v_vec_n_1);
                singIntStr_1->vec_c.set_pointer(v_vec_c_1);
                singIntStr_1->mat_nn.set_pointer(v_mat_nn_1);
                singIntStr_1->mat_nc.set_pointer(v_mat_nc_1);
                singIntStr_1->mat_cn.set_pointer(v_mat_cn_1);
                singIntStr_1->mat_cc.set_pointer(v_mat_cc_1);
                singIntStr_1->mat_cc.resize(nCellDOFs,nCellDOFs);
                singIntStr_1->cellFields.resize(nCellFields_1);
                singIntStr_1->cellFields = cellFields_1;
                singIntStr_1->tissFields = _tissue->_tissFields;
                
                singIntStr_1->tissIntegrals.set_pointer(&v_outputIntegrals_1[0]);
                singIntStr_1->cellIntegrals.set_pointer(&v_outputIntegrals_1[_tissIntegralIdx.size()]);
                singIntStr_1->tissIntegrals.resize(_tissIntegralIdx.size());
                singIntStr_1->cellIntegrals.resize(_cellIntegralIdx.size());

                singIntStr_1->_nodeDOFNames = _nodeDOFNames;
                singIntStr_1->_cellDOFNames = _cellDOFNames;
                singIntStr_1->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr_1->_mapCellDOFNames = _mapCellDOFNames;
                
                singIntStr_1->_nodeFieldNames = _tissue->_nodeFieldNames;
                singIntStr_1->_cellFieldNames = _tissue->_cellFieldNames;
                singIntStr_1->_tissFieldNames = _tissue->_tissFieldNames;
                singIntStr_1->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr_1->_mapCellFieldNames = _tissue->_mapCellFieldNames;
                singIntStr_1->_mapTissFieldNames = _tissue->_mapTissFieldNames;
                
                singIntStr_1->_cellIntegralNames = _cellIntegralNames;
                singIntStr_1->_tissIntegralNames = _tissIntegralNames;
                singIntStr_1->_mapCellIntegralNames = _mapCellIntegralNames;
                singIntStr_1->_mapTissIntegralNames = _mapTissIntegralNames;

                std::vector<double> v_inputFields_2(eNNMax_2*nNodeFields_2+nCellFields_2,0.0);
                std::vector<double> v_outputVector_2(eNNMax_2*nNodeDOFs+nCellDOFs,0.0);
                std::vector<double> v_outputMatrix_2((eNNMax_2*nNodeDOFs+nCellDOFs)*(eNNMax_2*nNodeDOFs+nCellDOFs),0.0);
                std::vector<double> v_outputIntegrals_2(_tissIntegralIdx.size()+_cellIntegralIdx.size(),0.0);

                double* v_nborFields_2 = &v_inputFields_2[0];
                double* v_cellFields_2 = &v_inputFields_2[eNNMax_2*nNodeFields_2];
                double* v_vec_n_2= &v_outputVector_2[0];
                double* v_vec_c_2 = &v_outputVector_2[eNNMax_2*nNodeDOFs];
                double* v_mat_nn_2 = &v_outputMatrix_2[0];
                double* v_mat_nc_2= &v_outputMatrix_2[eNNMax_2*nNodeDOFs*eNNMax_2*nNodeDOFs];
                double* v_mat_cn_2 = &v_outputMatrix_2[eNNMax_2*nNodeDOFs*(eNNMax_2*nNodeDOFs+nCellDOFs)];
                double* v_mat_cc_2 = &v_outputMatrix_2[eNNMax_2*nNodeDOFs*(eNNMax_2*nNodeDOFs+2*nCellDOFs)];
                
                RCP<SingleIntegralStr> singIntStr_2 = rcp(new SingleIntegralStr);
                singIntStr_2->cellID     = glo_m;
                singIntStr_2->pDim  = cell_2->_bfs->getParametricDimension();
                singIntStr_2->nborFields.set_pointer(v_nborFields_2);
                singIntStr_2->cellFields.set_pointer(v_cellFields_2);
                singIntStr_2->vec_n.set_pointer(v_vec_n_2);
                singIntStr_2->vec_c.set_pointer(v_vec_c_2);
                singIntStr_2->mat_nn.set_pointer(v_mat_nn_2);
                singIntStr_2->mat_nc.set_pointer(v_mat_nc_2);
                singIntStr_2->mat_cn.set_pointer(v_mat_cn_2);
                singIntStr_2->mat_cc.set_pointer(v_mat_cc_2);
                singIntStr_2->mat_cc.resize(nCellDOFs,nCellDOFs);
                singIntStr_2->cellFields.resize(nCellFields_2);
                singIntStr_2->cellFields = cellFields_2;
                singIntStr_2->tissFields = _tissue->_tissFields;
                
                singIntStr_2->tissIntegrals.set_pointer(&v_outputIntegrals_2[0]);
                singIntStr_2->cellIntegrals.set_pointer(&v_outputIntegrals_2[_tissIntegralIdx.size()]);
                singIntStr_2->tissIntegrals.resize(_tissIntegralIdx.size());
                singIntStr_2->cellIntegrals.resize(_cellIntegralIdx.size());

                singIntStr_2->_nodeDOFNames = _nodeDOFNames;
                singIntStr_2->_cellDOFNames = _cellDOFNames;
                singIntStr_2->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr_2->_mapCellDOFNames = _mapCellDOFNames;
                
                singIntStr_2->_nodeFieldNames = _tissue->_nodeFieldNames;
                singIntStr_2->_cellFieldNames = _tissue->_cellFieldNames;
                singIntStr_2->_tissFieldNames = _tissue->_tissFieldNames;
                singIntStr_2->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr_2->_mapCellFieldNames = _tissue->_mapCellFieldNames;
                singIntStr_2->_mapTissFieldNames = _tissue->_mapTissFieldNames;
                
                singIntStr_2->_cellIntegralNames = _cellIntegralNames;
                singIntStr_2->_tissIntegralNames = _tissIntegralNames;
                singIntStr_2->_mapCellIntegralNames = _mapCellIntegralNames;
                singIntStr_2->_mapTissIntegralNames = _mapTissIntegralNames;
                
                RCP<DoubleIntegralStr> doubIntStr   = rcp(new DoubleIntegralStr);
                doubIntStr->fillStr1 = singIntStr_1;
                doubIntStr->fillStr2 = singIntStr_2;
                std::vector<double> v_outputMatrix_i(2*(eNNMax_1*nNodeDOFs+nCellDOFs)*(eNNMax_2*nNodeDOFs+nCellDOFs),0.0);
                int cum_p = 0;
                double* v_mat_n1n2 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_1*nNodeDOFs*eNNMax_2*nNodeDOFs;
                double* v_mat_n1c2 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_1*nNodeDOFs*nCellDOFs;
                double* v_mat_c1n2 = &v_outputMatrix_i[cum_p];
                cum_p += nCellDOFs*eNNMax_2*nNodeDOFs;
                double* v_mat_c1c2 = &v_outputMatrix_i[cum_p];
                cum_p += nCellDOFs*nCellDOFs;
                double* v_mat_n2n1 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_2*nNodeDOFs*eNNMax_1*nNodeDOFs;
                double* v_mat_n2c1 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_2*nNodeDOFs*nCellDOFs;
                double* v_mat_c2n1 = &v_outputMatrix_i[cum_p];
                cum_p += nCellDOFs*eNNMax_1*nNodeDOFs;
                double* v_mat_c2c1 = &v_outputMatrix_i[cum_p];
                
                doubIntStr->mat_n1n2.set_pointer(v_mat_n1n2);
                doubIntStr->mat_n1c2.set_pointer(v_mat_n1c2);
                doubIntStr->mat_c1n2.set_pointer(v_mat_c1n2);
                doubIntStr->mat_c1c2.set_pointer(v_mat_c1c2);
                doubIntStr->mat_n2n1.set_pointer(v_mat_n2n1);
                doubIntStr->mat_n2c1.set_pointer(v_mat_n2c1);
                doubIntStr->mat_c2n1.set_pointer(v_mat_c2n1);
                doubIntStr->mat_c2c1.set_pointer(v_mat_c2c1);
                
                int offsetDOFs_1       = _cellDOFOffset[glo_n];
                int offsetglobFields_1 = _cellDOFOffset[glo_n]+nPts_1*nNodeDOFs;
                int offsetDOFs_2       = _cellDOFOffset[glo_m];
                int offsetglobFields_2 = _cellDOFOffset[glo_m]+nPts_2*nNodeDOFs;

                int e0 = -1;
                int   eNN_1{};
                int*  adjEN_1 = nullptr;
                int g0 = -1;
                int   eNN_2{};
                int*  adjEN_2 = nullptr;
                std::vector<std::vector<std::vector<double>>> *savedBFs_type_1;
                std::vector<std::vector<std::vector<double>>> *savedBFs_type_2;

                tensor<double,1> loc_tissIntegrals(_tissIntegralIdx.size());
                tensor<double,1> cellIntegrals_1(_cellIntegralIdx.size());
                tensor<double,1> cellIntegrals_2(_cellIntegralIdx.size());
                loc_tissIntegrals = 0.0;
                cellIntegrals_1 = 0.0;
                cellIntegrals_2 = 0.0;

                #pragma omp for
                for(size_t h = 0; h <  _elems_inte[inte].size(); h++)
                { 
                    int e =  _elems_inte[inte][h][0];
                    int g =  _elems_inte[inte][h][1];
                    if(e != e0 or g != g0)
                    {
                        if(e != e0 and e0 != -1)
                        {
                            loc_tissIntegrals += singIntStr_1->tissIntegrals;
                            cellIntegrals_1 += singIntStr_1->cellIntegrals;
                            
                            #pragma omp critical
                            AssembleElementalVector( offsetDOFs_1,  eNN_1,  nNodeDOFs, adjEN_1, singIntStr_1->vec_n.data(), _vector);
                            #pragma omp critical
                            _assembleElementalMatrix( offsetDOFs_1,    offsetDOFs_1, eNN_1, eNN_1,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_1, singIntStr_1->mat_nn.data(), _matrix );
                            
                            if(_cellDOFsInt)
                            {
                                #pragma omp critical
                                AssembleElementalVector( offsetglobFields_1,     1, nCellDOFs,  &dummy, singIntStr_1->vec_c.data(), _vector);

                                #pragma omp critical
                                _assembleElementalMatrix( offsetDOFs_1,   offsetglobFields_1, eNN_1,     1,  nNodeDOFs, nCellDOFs, adjEN_1,  &dummy, singIntStr_1->mat_nc.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_1,   offsetDOFs_1,     1, eNN_1, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_1, singIntStr_1->mat_cn.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_1,  offsetglobFields_1,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, singIntStr_1->mat_cc.data(), _matrix );
                            }
                        }
                        if(g != g0  and g0 != -1)
                        {
                            loc_tissIntegrals += singIntStr_2->tissIntegrals;
                            cellIntegrals_2 += singIntStr_2->cellIntegrals;
                            
                            //Assemble
                            #pragma omp critical
                            AssembleElementalVector(   offsetDOFs_2, eNN_2,  nNodeDOFs, adjEN_2, singIntStr_2->vec_n.data(), _vector);
                            
                            #pragma omp critical
                            _assembleElementalMatrix( offsetDOFs_2,    offsetDOFs_2, eNN_2, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_2, singIntStr_2->mat_nn.data(), _matrix );
                            
                            if(_cellDOFsInt)
                            {
                                #pragma omp critical
                                AssembleElementalVector(  offsetglobFields_2,     1, nCellDOFs,  &dummy, singIntStr_2->vec_c.data(), _vector);

                                #pragma omp critical
                                _assembleElementalMatrix( offsetDOFs_2,   offsetglobFields_2, eNN_2,     1,  nNodeDOFs, nCellDOFs, adjEN_2,  &dummy, singIntStr_2->mat_nc.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_2,   offsetDOFs_2,     1, eNN_2, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_2, singIntStr_2->mat_cn.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_2,  offsetglobFields_2,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, singIntStr_2->mat_cc.data(), _matrix );
                            }
                        }

                        if(e0!=-1 and g0!=-1)
                        {
                            #pragma omp critical
                            _assembleElementalMatrix( offsetDOFs_1, offsetDOFs_2, eNN_1, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_2, doubIntStr->mat_n1n2.data(), _matrix );
                            
                            #pragma omp critical
                            _assembleElementalMatrix( offsetDOFs_2, offsetDOFs_1, eNN_2, eNN_1,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_1, doubIntStr->mat_n2n1.data(), _matrix );

                            if(_cellDOFsInt)
                            {
                                #pragma omp critical
                                _assembleElementalMatrix(offsetDOFs_1,  offsetglobFields_2, eNN_1,     1,  nNodeDOFs, nCellDOFs, adjEN_1,  &dummy, doubIntStr->mat_n1c2.data(), _matrix);
                                #pragma omp critical
                                _assembleElementalMatrix(offsetglobFields_2,  offsetDOFs_1,     1, eNN_1, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_1, doubIntStr->mat_c2n1.data(), _matrix);

                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_1, offsetDOFs_2,    1, eNN_2, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_2,  doubIntStr->mat_c1n2.data(), _matrix);
                                #pragma omp critical
                                _assembleElementalMatrix(offsetDOFs_2,  offsetglobFields_1, eNN_2,     1,  nNodeDOFs, nCellDOFs, adjEN_2,  &dummy, doubIntStr->mat_n2c1.data(), _matrix);
                                #pragma omp critical
                                _assembleElementalMatrix(offsetglobFields_1,   offsetglobFields_2,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, doubIntStr->mat_c1c2.data(), _matrix);
                                #pragma omp critical
                                _assembleElementalMatrix(offsetglobFields_2,   offsetglobFields_1,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, doubIntStr->mat_c2c1.data(), _matrix);
                            }
                        }

                        if(e!=e0)
                        {
                            eNN_1 = cell_1->_bfs->getNumberOfNeighbours(e);
                            adjEN_1 = cell_1->_bfs->getNeighbours(e);
                            
                            savedBFs_type_1 = &(_savedBFs_intera[cell_1->_bfs->getElementType(e)]);

                            singIntStr_1->elemID = e;
                            singIntStr_1->eNN    = eNN_1;
                            singIntStr_1->nborFields.resize(eNN_1,nNodeFields_1);
                            for(int i=0; i < eNN_1; i++)
                                singIntStr_1->nborFields(i,all)   = nodeFields_1(adjEN_1[i],all);

                            singIntStr_1->vec_n.resize(eNN_1,nNodeDOFs);
                            singIntStr_1->vec_c.resize(nCellDOFs);
                            singIntStr_1->mat_nn.resize(eNN_1,nNodeDOFs,eNN_1,nNodeDOFs);
                            singIntStr_1->mat_nc.resize(eNN_1,nNodeDOFs,nCellDOFs);
                            singIntStr_1->mat_cn.resize(nCellDOFs,eNN_1,nNodeDOFs);
                            singIntStr_1->mat_cc.resize(nCellDOFs,nCellDOFs);
                            
                            std::fill(v_outputVector_1.begin(),v_outputVector_1.end(),0.0);
                            std::fill(v_outputMatrix_1.begin(),v_outputMatrix_1.end(),0.0);
                            std::fill(v_outputIntegrals_1.begin(),v_outputIntegrals_1.end(),0.0);
                        }

                        if(g != g0)
                        {
                            eNN_2   = cell_2->_bfs->getNumberOfNeighbours(g);
                            adjEN_2 = cell_2->_bfs->getNeighbours(g);
                            savedBFs_type_2 = &(_savedBFs_intera[cell_2->_bfs->getElementType(g)]);

                            singIntStr_2->elemID = g;
                            singIntStr_2->eNN    = eNN_2;
                            singIntStr_2->nborFields.resize(eNN_2,nNodeFields_2);
                            for(int i=0; i < eNN_2; i++)
                                singIntStr_2->nborFields(i,all)   = nodeFields_2(adjEN_2[i],all);
                        
                            singIntStr_2->vec_n.resize(eNN_2,nNodeDOFs);
                            singIntStr_2->vec_c.resize(nCellDOFs);
                            singIntStr_2->mat_nn.resize(eNN_2,nNodeDOFs,eNN_2,nNodeDOFs);
                            singIntStr_2->mat_nc.resize(eNN_2,nNodeDOFs,nCellDOFs);
                            singIntStr_2->mat_cn.resize(nCellDOFs,eNN_2,nNodeDOFs);
                            singIntStr_2->mat_cc.resize(nCellDOFs,nCellDOFs);
                            
                            std::fill(v_outputVector_2.begin(),v_outputVector_2.end(),0.0);
                            std::fill(v_outputMatrix_2.begin(),v_outputMatrix_2.end(),0.0);
                            std::fill(v_outputIntegrals_2.begin(),v_outputIntegrals_2.end(),0.0);
                        }


                        doubIntStr->mat_n1n2.resize(eNN_1,nNodeDOFs,eNN_2,nNodeDOFs);
                        doubIntStr->mat_n1c2.resize(eNN_1,nNodeDOFs,nCellDOFs);
                        doubIntStr->mat_c1n2.resize(nCellDOFs,eNN_2,nNodeDOFs);
                        doubIntStr->mat_c1c2.resize(nCellDOFs,nCellDOFs);
                        doubIntStr->mat_n2n1.resize(eNN_2,nNodeDOFs,eNN_1,nNodeDOFs);
                        doubIntStr->mat_n2c1.resize(eNN_2,nNodeDOFs,nCellDOFs);
                        doubIntStr->mat_c2n1.resize(nCellDOFs,eNN_1,nNodeDOFs);
                        doubIntStr->mat_c2c1.resize(nCellDOFs,nCellDOFs);
                        
                        std::fill(v_outputMatrix_i.begin(),v_outputMatrix_i.end(),0.0);
                    }

                    int k = _elems_inte[inte][h][2];
                    int l = _elems_inte[inte][h][3];
                    
                    singIntStr_1->bfs = savedBFs_type_1->operator[](k);
                    singIntStr_1->w_sample = _wSamples_intera[k];
                    singIntStr_2->bfs = savedBFs_type_2->operator[](l);
                    singIntStr_2->w_sample = _wSamples_intera[l];
                     _doubleIntegrand(doubIntStr);
                                        
                    e0 = e;
                    g0 = g;
                }

                if(e0!=-1 and g0!=-1)
                {
                    loc_tissIntegrals += singIntStr_1->tissIntegrals;
                    cellIntegrals_1 += singIntStr_1->cellIntegrals;

                    #pragma omp critical
                    AssembleElementalVector( offsetDOFs_1,  singIntStr_1->eNN,  nNodeDOFs, adjEN_1, singIntStr_1->vec_n.data(), _vector);
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_1,    offsetDOFs_1, singIntStr_1->eNN, singIntStr_1->eNN,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_1, singIntStr_1->mat_nn.data(), _matrix );
                    
                    if(_cellDOFsInt)
                    {
                        #pragma omp critical
                        AssembleElementalVector( offsetglobFields_1,     1, nCellDOFs,  &dummy, singIntStr_1->vec_c.data(), _vector);

                        #pragma omp critical
                        _assembleElementalMatrix( offsetDOFs_1,   offsetglobFields_1, singIntStr_1->eNN,     1,  nNodeDOFs, nCellDOFs, adjEN_1,  &dummy, singIntStr_1->mat_nc.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_1,   offsetDOFs_1,     1, singIntStr_1->eNN, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_1, singIntStr_1->mat_cn.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_1,  offsetglobFields_1,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, singIntStr_1->mat_cc.data(), _matrix );
                    }
                    
                    loc_tissIntegrals += singIntStr_2->tissIntegrals;
                    cellIntegrals_2 += singIntStr_2->cellIntegrals;
                                
                    //Assemble
                    #pragma omp critical
                    AssembleElementalVector(   offsetDOFs_2, eNN_2,  nNodeDOFs, adjEN_2, singIntStr_2->vec_n.data(), _vector);
                    
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_2,    offsetDOFs_2, eNN_2, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_2, singIntStr_2->mat_nn.data(), _matrix );
                    
                    if(_cellDOFsInt)
                    {
                        #pragma omp critical
                        AssembleElementalVector(  offsetglobFields_2,     1, nCellDOFs,  &dummy, singIntStr_2->vec_c.data(), _vector);

                        #pragma omp critical
                        _assembleElementalMatrix( offsetDOFs_2,   offsetglobFields_2, eNN_2,     1,  nNodeDOFs, nCellDOFs, adjEN_2,  &dummy, singIntStr_2->mat_nc.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_2,   offsetDOFs_2,     1, eNN_2, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_2, singIntStr_2->mat_cn.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_2,  offsetglobFields_2,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, singIntStr_2->mat_cc.data(), _matrix );
                    }

                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_1, offsetDOFs_2, eNN_1, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_2, doubIntStr->mat_n1n2.data(), _matrix );
                    
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_2, offsetDOFs_1, eNN_2, eNN_1,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_1, doubIntStr->mat_n2n1.data(), _matrix );

                    if(_cellDOFsInt)
                    {
                        #pragma omp critical
                        _assembleElementalMatrix(offsetDOFs_1,  offsetglobFields_2, eNN_1,     1,  nNodeDOFs, nCellDOFs, adjEN_1,  &dummy, doubIntStr->mat_n1c2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_2,  offsetDOFs_1,     1, eNN_1, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_1, doubIntStr->mat_c2n1.data(), _matrix);

                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_1, offsetDOFs_2,    1, eNN_2, nCellDOFs,  nNodeDOFs,  &dummy, adjEN_2,  doubIntStr->mat_c1n2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetDOFs_2,  offsetglobFields_1, eNN_2,     1,  nNodeDOFs, nCellDOFs, adjEN_2,  &dummy, doubIntStr->mat_n2c1.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_1,   offsetglobFields_2,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, doubIntStr->mat_c1c2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_2,   offsetglobFields_1,     1,     1, nCellDOFs, nCellDOFs,  &dummy,  &dummy, doubIntStr->mat_c2c1.data(), _matrix);
                    }

                    #pragma omp critical
                    tissIntegrals += loc_tissIntegrals;
                    
                    for(int i = 0; i < int(_cellIntegralIdx.size()); i++)
                    {
                        int i1 = glo_n*_cellIntegralIdx.size()+i;
                        #pragma omp critical
                        _cellIntegrals->SumIntoGlobalValues(1, &i1, &cellIntegrals_1(i));
                        int i2 = glo_m*_cellIntegralIdx.size()+i;
                        #pragma omp critical
                        _cellIntegrals->SumIntoGlobalValues(1, &i2, &cellIntegrals_2(i));
                    }
                }
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, tissIntegrals.data(), tissIntegrals.size(), MPI_DOUBLE, MPI_SUM, _tissue->_comm);
        
        _tissIntegrals = tissIntegrals;
    }


    void Integration::_addArrayToDOFs(Teuchos::RCP<Epetra_FEVector> vec, double scale)
    {
        using Teuchos::RCP;
        
        double **sol_raw_ptr;
        vec->ExtractView(&sol_raw_ptr);
        
        int t{};
        for(size_t n = 0; n < _tissue->_cells.size(); n++)
        {
            RCP<Cell> cell = _tissue->_cells[n];
            
            //We assume that vector is ordered in such a way that it first contains nPts * nDOFs nodal DOFs and then nglobFields
            int nPts = cell->getNumberOfPoints();
            int nFields = cell->_nodeFields.shape()[1];
            
            double* dof_raw_ptr = cell->_nodeFields.data();
            
            for(int i = 0; i < nPts; i++)
            {
                for(size_t m=0; m < _nodeDOFIdx.size(); m++)
                {
                    dof_raw_ptr[i*nFields+_nodeDOFIdx[m]] += scale * sol_raw_ptr[0][t];
                    t++;
                }
            }
            
            double* gdof_raw_ptr = cell->_cellFields.data();
            for(size_t g = 0; g < _cellDOFIdx.size(); g++)
            {
                gdof_raw_ptr[_cellDOFIdx[g]] += scale * sol_raw_ptr[0][t];
                t++;
            }
        }
    }
    void Integration::_setArrayToDOFs(Teuchos::RCP<Epetra_FEVector> vec, double scale)
    {
        using Teuchos::RCP;
        
        double **sol_raw_ptr;
        vec->ExtractView(&sol_raw_ptr);
        
        int t{};
        for(size_t n = 0; n < _tissue->_cells.size(); n++)
        {
            RCP<Cell> cell = _tissue->_cells[n];
            
            //We assume that vector is ordered in such a way that it first contains nPts * nDOFs nodal DOFs and then nglobFields
            int nPts = cell->getNumberOfPoints();
            int nDOFs = cell->_nodeFields.shape()[1];
            
            double* dof_raw_ptr = cell->_nodeFields.data();
            
            for(int i = 0; i < nPts; i++)
            {
                for(size_t m=0; m < _nodeDOFIdx.size(); m++)
                {
                    dof_raw_ptr[i*nDOFs+_nodeDOFIdx[m]] = scale * sol_raw_ptr[0][t];
                    t++;
                }
            }
            
            double* gdof_raw_ptr = cell->_cellFields.data();
            for(size_t g = 0; g < _cellDOFIdx.size(); g++)
            {
                gdof_raw_ptr[_cellDOFIdx[g]] = scale * sol_raw_ptr[0][t];
                t++;
            }
        }
    }

    void Integration::_checkIntegralNames()
    {
        using namespace std;


        _cellIntegralIdx.clear();
        _mapCellIntegralNames.clear();
        int n{};
        for(auto& f: _cellIntegralNames)
        {
            if(std::count(_cellIntegralNames.begin(), _cellIntegralNames.end(), f) > 1)
                throw runtime_error("Cell Integral " + f + " has been set twice.");
            else
            {
                _cellIntegralIdx.push_back(_tissue->getCellFieldIdx(f));
                _mapCellIntegralNames[f] = n;
                n++;
            }
        }
        
        _tissIntegralIdx.clear();
        _mapTissIntegralNames.clear();
        n = 0;
        for(auto& f: _tissIntegralNames)
        {
            if(std::count(_tissIntegralNames.begin(), _tissIntegralNames.end(), f) > 1)
                throw runtime_error("Tissue integral " + f + " has been set twice.");
            else
            {
                _tissIntegralIdx.push_back(_tissue->getTissFieldIdx(f));
                _mapTissIntegralNames[f] = n;
                n++;
            }
        }
    }

    void Integration::calculateInteractingGaussPoints()
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;

        _elems_inte.resize(_tissue->_inters[_tissue->getMyPart()].size());
        
        int change{};
        
        for(size_t inte = 0; inte < _tissue->_inters[_tissue->_myPart].size(); inte ++)
        {
            int glo_1 = _tissue->_inters[_tissue->_myPart][inte].first;
            int glo_2 = _tissue->_inters[_tissue->_myPart][inte].second;

            RCP<Cell> cell_1 = _tissue->GetCell(glo_1);
            RCP<Cell> cell_2 = _tissue->GetCell(glo_2);

            vector<array<int,4>> elems_inte;

            vtkSmartPointer<vtkPoints> points_2 = vtkSmartPointer<vtkPoints>::New();
            for(int e = 0; e < cell_2->getNumberOfElements(); e++)
            {
                int eNN_2 = cell_2->_bfs->getNumberOfNeighbours(e);
                tensor<double,2> nborCoords(eNN_2,3);
                for(int i=0; i < eNN_2; i++)
                {
                    int ii = cell_2->_bfs->getNeighbours(e)[i];
                    nborCoords(i,all) = cell_2->_nodeFields(ii,range(0,2));
                    if(_displNames.size() == 3)
                    {
                        nborCoords(i,0) += cell_2->getNodeField(_displNames[0])(ii);
                        nborCoords(i,1) += cell_2->getNodeField(_displNames[1])(ii);
                        nborCoords(i,2) += cell_2->getNodeField(_displNames[2])(ii);
                    }
                }
                for(int k = 0; k < _iPts_intera; k++)
                {
                    tensor<double,1> bfs(_savedBFs_intera[cell_2->_bfs->getElementType(e)][k][0].data(),eNN_2);
                    tensor<double,1> x = bfs * nborCoords;
                    points_2->InsertNextPoint(x(0),x(1),x(2));
                }
            }
            vtkSmartPointer<vtkPolyData> polydata_2 = vtkSmartPointer<vtkPolyData>::New();
            polydata_2->SetPoints(points_2);

            vtkNew<vtkPointLocator> pointTree;
            pointTree->SetDataSet(polydata_2);
            pointTree->BuildLocator();

            for(int e = 0; e < cell_1->getNumberOfElements(); e++)
            {
                int eNN_1 = cell_1->_bfs->getNumberOfNeighbours(e);
                tensor<double,2> nborCoords(eNN_1,3);
                for(int i=0; i < eNN_1; i++)
                {
                    int ii = cell_1->_bfs->getNeighbours(e)[i];
                    nborCoords(i,all) = cell_1->_nodeFields(ii,range(0,2));

                    if(_displNames.size() == 3)
                    {
                        nborCoords(i,0) += cell_1->getNodeField(_displNames[0])(ii);
                        nborCoords(i,1) += cell_1->getNodeField(_displNames[1])(ii);
                        nborCoords(i,2) += cell_1->getNodeField(_displNames[2])(ii);
                    }
                }
                for(int k = 0; k < _iPts_intera; k++)
                {
                    tensor<double,1> bfs(_savedBFs_intera[cell_1->_bfs->getElementType(e)][k][0].data(),eNN_1);
                    tensor<double,1> x = bfs * nborCoords;

                    vtkNew<vtkIdList> result;
                    pointTree->FindPointsWithinRadius(_deltaInter, x.data(), result);

                    vtkIdType n = result->GetNumberOfIds();
                    for(int g = 0; g < n; g++)
                    {
                        int elemId = result->GetId(g)/_iPts_intera;
                        int kPtId = result->GetId(g)%_iPts_intera;
                        elems_inte.push_back({e,elemId,k,kPtId});   
                    }        
                }
            }

            sort(elems_inte.begin(),elems_inte.end());

            
            if(_elems_inte[inte] != elems_inte)
            {
                _elems_inte[inte] = elems_inte;
                change = 1;
            }
        }
        
        
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_INT, MPI_MAX, _tissue->_comm);

        _rec_str = _rec_str ? _rec_str : static_cast<bool>(change);       
    }
}
