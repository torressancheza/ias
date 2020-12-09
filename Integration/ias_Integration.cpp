#include <iostream>
#include <numeric>
#include <omp.h>

#include <Epetra_MpiComm.h>

#include "ias_Integration.h"
#include "ias_BasicStructures.h"

namespace ias
{

    void Integration::Update()
    {
        using namespace std;
        using Teuchos::rcp;
        
        
        _nodeDOFIdx.clear();
        int n{};
        for(auto s: _nodeDOFNames)
        {
            _nodeDOFIdx.push_back(_tissue->getNodeFieldIdx(s));
            _mapNodeDOFNames[s] = n;
        }
        
        _globDOFIdx.clear();
        n = 0;
        for(auto s: _globDOFNames)
        {
            _globDOFIdx.push_back(_tissue->getGlobFieldIdx(s));
            _mapGlobDOFNames[s] = n;
        }
        
        int loc_nCells = _tissue->_cells.size();
        
        if(_nodeDOFIdx.size() == 0 and _globDOFIdx.size() == 0)
            throw runtime_error("No degrees of freedom have been selected for the problem!");
        
        int nNodeDOFs = _nodeDOFIdx.size();
        int nGlobDOFs = _globDOFIdx.size();
        
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
        
        //First use AssembleElementalMatrix with create to true
        _assembleElementalMatrix = AssembleElementalMatrix<true>;
        
        //Global integrals
        _globIntegrals.resize(_nGlobIntegrals);

        //integrals per cell
        _cellIntegrals.resize(loc_nCells);
        for(int i=0; i <loc_nCells; i++)
            _cellIntegrals[i].resize(_nCellIntegrals);

//        //fields per cell
//        _cellIntFields.resize(loc_nCells);
//        for(int i=0; i <loc_nCells; i++)
//        {
//            Epetra_Map vec_map(-1, loc_totSize, 0, Epetra_MpiComm(tissue->_comm));
//            _cellIntFields[i] = rcp(new Epetra_FEVector(vec_map));
//        }

    }

    void Integration::computeSingleIntegral()
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;

        if(_singleIntegrand==nullptr)
            return;
        
        fill(_globIntegrals.begin(), _globIntegrals.end(),0.0);
        
        for(int n=0; n < int(_tissue->_cells.size());n++)
        {
            int glo_n = _tissue->getGlobalIdx(n);
            
            auto& nodeFields   = _tissue->_cells[n]->_nodeFields;
            auto& globFields   = _tissue->_cells[n]->_globFields;

            int nElem  = _tissue->_cells[n]->getNumberOfElements();
            int nPts   = _tissue->_cells[n]->getNumberOfPoints();
            int nNodeFields = nodeFields.shape()[1];
            int nGlobFields = globFields.size();
            int nNodeDOFs   = _nodeDOFIdx.size();
            int nGlobDOFs   = _globDOFIdx.size();
            int eNNMax      = _tissue->_cells[n]->_bfs->getMaxNumberOfNeighbours();
            
            fill(_cellIntegrals[n].begin(), _cellIntegrals[n].end(),0.0);
            #pragma omp parallel
            {
                std::vector<double> v_inputFields(eNNMax*nNodeFields+nGlobFields,0.0);
                std::vector<double> v_outputVector(eNNMax*nNodeDOFs+nGlobDOFs,0.0);
                std::vector<double> v_outputMatrix((eNNMax*nNodeDOFs+nGlobDOFs)*(eNNMax*nNodeDOFs+nGlobDOFs),0.0);
                double* v_nborFields = &v_inputFields[0];
                double* v_globFields = &v_inputFields[eNNMax*nNodeFields];
                double* v_vec_n  = &v_outputVector[0];
                double* v_vec_g  = &v_outputVector[eNNMax*nNodeDOFs];
                double* v_mat_nn = &v_outputMatrix[0];
                double* v_mat_ng = &v_outputMatrix[eNNMax*nNodeDOFs*eNNMax*nNodeDOFs];
                double* v_mat_gn = &v_outputMatrix[eNNMax*nNodeDOFs*(eNNMax*nNodeDOFs+nGlobDOFs)];
                double* v_mat_gg = &v_outputMatrix[eNNMax*nNodeDOFs*(eNNMax*nNodeDOFs+2*nGlobDOFs)];

                RCP<SingleIntegralStr> singIntStr = rcp(new SingleIntegralStr);
                singIntStr->cellID     = glo_n;
                singIntStr->nborFields.set_pointer(v_nborFields);
                singIntStr->globFields.set_pointer(v_globFields);
                singIntStr->globFields.resize(nGlobFields);
                singIntStr->globFields = globFields;
                singIntStr->pDim  = _tissue->_cells[n]->_bfs->getParametricDimension();
                singIntStr->tissFields = _tissue->_tissFields;

                singIntStr->vec_n.set_pointer(v_vec_n);
                singIntStr->vec_g.set_pointer(v_vec_g);
                singIntStr->mat_nn.set_pointer(v_mat_nn);
                singIntStr->mat_ng.set_pointer(v_mat_ng);
                singIntStr->mat_gn.set_pointer(v_mat_gn);
                singIntStr->mat_gg.set_pointer(v_mat_gg);
                singIntStr->mat_gg.resize(nGlobDOFs,nGlobDOFs);

                singIntStr->globIntegrals.resize(_nGlobIntegrals,0.0);
                singIntStr->cellIntegrals.resize(_nCellIntegrals,0.0);
                
                singIntStr->_nodeDOFNames = _nodeDOFNames;
                singIntStr->_globDOFNames = _globDOFNames;
                singIntStr->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr->_mapGlobDOFNames = _mapGlobDOFNames;
                
                singIntStr->_nodeFieldNames  = _tissue->_nodeFieldNames;
                singIntStr->_globFieldNames  = _tissue->_globFieldNames;
                singIntStr->_tissFieldNames  = _tissue->_tissFieldNames;
                singIntStr->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr->_mapGlobFieldNames = _tissue->_mapGlobFieldNames;
                singIntStr->_mapTissFieldNames = _tissue->_mapTissFieldNames;

                vector<double> globIntegrals(_nGlobIntegrals,0.0);
                vector<double> cellIntegrals(_nCellIntegrals,0.0);

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
                    singIntStr->vec_g.resize(nGlobDOFs);
                    
                    singIntStr->mat_nn.resize(eNN,nNodeDOFs,eNN,nNodeDOFs);
                    singIntStr->mat_ng.resize(eNN,nNodeDOFs,nGlobDOFs);
                    singIntStr->mat_gn.resize(nGlobDOFs,eNN,nNodeDOFs);
                    
                    std::fill(v_outputVector.begin(),v_outputVector.end(),0.0);
                    std::fill(v_outputMatrix.begin(),v_outputMatrix.end(),0.0);
                    std::fill(singIntStr->globIntegrals.begin(),singIntStr->globIntegrals.end(),0.0);
                    std::fill(singIntStr->cellIntegrals.begin(),singIntStr->cellIntegrals.end(),0.0);

                    auto& savedBFs_type = _savedBFs_single[_tissue->_cells[n]->_bfs->getElementType(e)];
                    for(size_t k=0; k < _wSamples_single.size(); k++)
                    {
                        singIntStr->bfs = savedBFs_type[k];
                        singIntStr->w_sample = _wSamples_single[k];
                        singIntStr->sampID = k;
                        _singleIntegrand(singIntStr);
                    }

                    for(int t=0; t < _nGlobIntegrals; t++)
                        globIntegrals[t] += singIntStr->globIntegrals[t];
                    
                    for(int t=0; t < _nCellIntegrals; t++)
                        cellIntegrals[t] += singIntStr->cellIntegrals[t];
                    
                    int dummy{0};
                    int offsetDOFs       = _cellDOFOffset[glo_n];
                    int offsetglobFields = _cellDOFOffset[glo_n]+nPts*nNodeDOFs;

                    #pragma omp critical
                    AssembleElementalVector( offsetDOFs, eNN,  nNodeDOFs,    adjEN, singIntStr->vec_n.data(), _vector);
                    #pragma omp critical
                    AssembleElementalVector(offsetglobFields,   1, nGlobDOFs,   &dummy, singIntStr->vec_g.data(), _vector);
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs, offsetDOFs, eNN, eNN,  nNodeDOFs,  nNodeDOFs, adjEN, adjEN, singIntStr->mat_nn.data(), _matrix );
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs,offsetglobFields, eNN,   1,  nNodeDOFs, nGlobDOFs, adjEN, &dummy, singIntStr->mat_ng.data(), _matrix);
                    #pragma omp critical
                    _assembleElementalMatrix(offsetglobFields, offsetDOFs,   1, eNN, nGlobDOFs,  nNodeDOFs, &dummy, adjEN, singIntStr->mat_gn.data(), _matrix);
                    #pragma omp critical
                    _assembleElementalMatrix(offsetglobFields,offsetglobFields,   1,   1, nGlobDOFs, nGlobDOFs, &dummy, &dummy, singIntStr->mat_gg.data(), _matrix);
                }
                
                #pragma omp critical
                std::transform (_globIntegrals.begin(), _globIntegrals.end(), globIntegrals.begin(), _globIntegrals.begin(), std::plus<double>());
                #pragma omp critical
                std::transform (_cellIntegrals[n].begin(), _cellIntegrals[n].end(), cellIntegrals.begin(), _cellIntegrals[n].begin(), std::plus<double>());
            }
        }
        
//        double **sol_raw_ptr;
//        _vector->ExtractView(&sol_raw_ptr);
//        
//        for(int i = 0; i < 198; i++)
//        {
//            if(abs(sol_raw_ptr[0][i] - sol_raw_ptr[0][199+i])>1.E-16)
//                cout << i << " " << abs(sol_raw_ptr[0][i] - sol_raw_ptr[0][199+i]) << endl;
//        }
//        getchar();

//        cout << *_vector << endl;
        MPI_Allreduce(MPI_IN_PLACE, _globIntegrals.data(), _nGlobIntegrals, MPI_DOUBLE, MPI_SUM, _tissue->_comm);
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
        
        for(size_t inte = 0; inte < _tissue->_inters[_tissue->getMyPart()].size(); inte++)
        {
            if(_tissue->_elems_inte.size() == 0)
                throw runtime_error("Integration::computeDoubleIntegral: there are no interacting elements. Did you forget to call calculateInteractingElements?");
            
            int glo_n = _tissue->_inters[_tissue->getMyPart()][inte].first;
            int glo_m = _tissue->_inters[_tissue->getMyPart()][inte].second;
            
            if (glo_n == glo_m)
                continue;
            
            if(_tissue->_elems_inte[inte].size() == 0)
                continue;
            
            RCP<Cell> cell_1 = _tissue->GetCell(glo_n);
            RCP<Cell> cell_2 = _tissue->GetCell(glo_m);
            
            auto& nodeFields_1 = cell_1->_nodeFields;
            auto& globFields_1 = cell_1->_globFields;
            int nPts_1         = cell_1->getNumberOfPoints();
            int nNodeFields_1  = nodeFields_1.shape()[1];
            int nGlobFields_1  = globFields_1.size();
            int eNNMax_1       = cell_1->_bfs->getMaxNumberOfNeighbours();

            auto& nodeFields_2 = cell_2->_nodeFields;
            auto& globFields_2 = cell_2->_globFields;
            int nPts_2         = cell_2->getNumberOfPoints();
            int nNodeFields_2  = nodeFields_2.shape()[1];
            int nGlobFields_2  = globFields_2.size();
            int eNNMax_2       = cell_2->_bfs->getMaxNumberOfNeighbours();

            int nNodeDOFs    = _nodeDOFIdx.size();
            int nGlobDOFs    = _globDOFIdx.size();
            
            #pragma omp parallel
            {
                std::vector<double> v_inputFields_1(eNNMax_1*nNodeFields_1+nGlobFields_1,0.0);
                std::vector<double> v_outputVector_1(eNNMax_1*nNodeDOFs+nGlobDOFs,0.0);
                std::vector<double> v_outputMatrix_1((eNNMax_1*nNodeDOFs+nGlobDOFs)*(eNNMax_1*nNodeDOFs+nGlobDOFs),0.0);
                double* v_nborFields_1 = &v_inputFields_1[0];
                double* v_globFields_1 = &v_inputFields_1[eNNMax_1*nNodeFields_1];
                double* v_vec_n_1= &v_outputVector_1[0];
                double* v_vec_g_1 = &v_outputVector_1[eNNMax_1*nNodeDOFs];
                double* v_mat_nn_1 = &v_outputMatrix_1[0];
                double* v_mat_ng_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*eNNMax_1*nNodeDOFs];
                double* v_mat_gn_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*(eNNMax_1*nNodeDOFs+nGlobDOFs)];
                double* v_mat_gg_1 = &v_outputMatrix_1[eNNMax_1*nNodeDOFs*(eNNMax_1*nNodeDOFs+2*nGlobDOFs)];
                
                RCP<SingleIntegralStr> singIntStr_1 = rcp(new SingleIntegralStr);
                singIntStr_1->cellID = glo_n;
                singIntStr_1->pDim   = cell_1->_bfs->getParametricDimension();
                singIntStr_1->globIntegrals.resize(_nGlobIntegrals,0.0);
                singIntStr_1->cellIntegrals.resize(_nCellIntegrals,0.0);
                singIntStr_1->nborFields.set_pointer(v_nborFields_1);
                singIntStr_1->globFields.set_pointer(v_globFields_1);
                singIntStr_1->vec_n.set_pointer(v_vec_n_1);
                singIntStr_1->vec_g.set_pointer(v_vec_g_1);
                singIntStr_1->mat_nn.set_pointer(v_mat_nn_1);
                singIntStr_1->mat_ng.set_pointer(v_mat_ng_1);
                singIntStr_1->mat_gn.set_pointer(v_mat_gn_1);
                singIntStr_1->mat_gg.set_pointer(v_mat_gg_1);
                singIntStr_1->mat_gg.resize(nGlobDOFs,nGlobDOFs);
                singIntStr_1->globFields.resize(nGlobFields_1);
                singIntStr_1->globFields = globFields_1;
                singIntStr_1->tissFields = _tissue->_tissFields;

                singIntStr_1->_nodeDOFNames = _nodeDOFNames;
                singIntStr_1->_globDOFNames = _globDOFNames;
                singIntStr_1->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr_1->_mapGlobDOFNames = _mapGlobDOFNames;
                
                singIntStr_1->_nodeFieldNames = _tissue->_nodeFieldNames;
                singIntStr_1->_globFieldNames = _tissue->_globFieldNames;
                singIntStr_1->_tissFieldNames = _tissue->_tissFieldNames;
                singIntStr_1->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr_1->_mapGlobFieldNames = _tissue->_mapGlobFieldNames;
                singIntStr_1->_mapTissFieldNames = _tissue->_mapTissFieldNames;

                std::vector<double> v_inputFields_2(eNNMax_2*nNodeFields_2+nGlobFields_2,0.0);
                std::vector<double> v_outputVector_2(eNNMax_2*nNodeDOFs+nGlobDOFs,0.0);
                std::vector<double> v_outputMatrix_2((eNNMax_2*nNodeDOFs+nGlobDOFs)*(eNNMax_2*nNodeDOFs+nGlobDOFs),0.0);
                double* v_nborFields_2 = &v_inputFields_2[0];
                double* v_globFields_2 = &v_inputFields_2[eNNMax_2*nNodeFields_2];
                double* v_vec_n_2= &v_outputVector_2[0];
                double* v_vec_g_2 = &v_outputVector_2[eNNMax_2*nNodeDOFs];
                double* v_mat_nn_2 = &v_outputMatrix_2[0];
                double* v_mat_ng_2= &v_outputMatrix_2[eNNMax_2*nNodeDOFs*eNNMax_2*nNodeDOFs];
                double* v_mat_gn_2 = &v_outputMatrix_2[eNNMax_2*nNodeDOFs*(eNNMax_2*nNodeDOFs+nGlobDOFs)];
                double* v_mat_gg_2 = &v_outputMatrix_2[eNNMax_2*nNodeDOFs*(eNNMax_2*nNodeDOFs+2*nGlobDOFs)];
                
                RCP<SingleIntegralStr> singIntStr_2 = rcp(new SingleIntegralStr);
                singIntStr_2->cellID     = glo_m;
                singIntStr_2->pDim  = cell_2->_bfs->getParametricDimension();
                singIntStr_2->globIntegrals.resize(_nGlobIntegrals,0.0);
                singIntStr_2->cellIntegrals.resize(_nCellIntegrals,0.0);
                singIntStr_2->nborFields.set_pointer(v_nborFields_2);
                singIntStr_2->globFields.set_pointer(v_globFields_2);
                singIntStr_2->vec_n.set_pointer(v_vec_n_2);
                singIntStr_2->vec_g.set_pointer(v_vec_g_2);
                singIntStr_2->mat_nn.set_pointer(v_mat_nn_2);
                singIntStr_2->mat_ng.set_pointer(v_mat_ng_2);
                singIntStr_2->mat_gn.set_pointer(v_mat_gn_2);
                singIntStr_2->mat_gg.set_pointer(v_mat_gg_2);
                singIntStr_2->mat_gg.resize(nGlobDOFs,nGlobDOFs);
                singIntStr_2->globFields.resize(nGlobFields_2);
                singIntStr_2->globFields = globFields_2;
                singIntStr_2->tissFields = _tissue->_tissFields;

                singIntStr_2->_nodeDOFNames = _nodeDOFNames;
                singIntStr_2->_globDOFNames = _globDOFNames;
                singIntStr_2->_mapNodeDOFNames = _mapNodeDOFNames;
                singIntStr_2->_mapGlobDOFNames = _mapGlobDOFNames;
                
                singIntStr_2->_nodeFieldNames = _tissue->_nodeFieldNames;
                singIntStr_2->_globFieldNames = _tissue->_globFieldNames;
                singIntStr_2->_tissFieldNames = _tissue->_tissFieldNames;
                singIntStr_2->_mapNodeFieldNames = _tissue->_mapNodeFieldNames;
                singIntStr_2->_mapGlobFieldNames = _tissue->_mapGlobFieldNames;
                singIntStr_2->_mapTissFieldNames = _tissue->_mapTissFieldNames;
                
                RCP<DoubleIntegralStr> doubIntStr   = rcp(new DoubleIntegralStr);
                doubIntStr->fillStr1 = singIntStr_1;
                doubIntStr->fillStr2 = singIntStr_2;
                std::vector<double> v_outputMatrix_i(2*(eNNMax_1*nNodeDOFs+nGlobDOFs)*(eNNMax_2*nNodeDOFs+nGlobDOFs),0.0);
                int cum_p = 0;
                double* v_mat_n1n2 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_1*nNodeDOFs*eNNMax_2*nNodeDOFs;
                double* v_mat_n1g2 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_1*nNodeDOFs*nGlobDOFs;
                double* v_mat_g1n2 = &v_outputMatrix_i[cum_p];
                cum_p += nGlobDOFs*eNNMax_2*nNodeDOFs;
                double* v_mat_g1g2 = &v_outputMatrix_i[cum_p];
                cum_p += nGlobDOFs*nGlobDOFs;
                double* v_mat_n2n1 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_2*nNodeDOFs*eNNMax_1*nNodeDOFs;
                double* v_mat_n2g1 = &v_outputMatrix_i[cum_p];
                cum_p += eNNMax_2*nNodeDOFs*nGlobDOFs;
                double* v_mat_g2n1 = &v_outputMatrix_i[cum_p];
                cum_p += nGlobDOFs*eNNMax_1*nNodeDOFs;
                double* v_mat_g2g1 = &v_outputMatrix_i[cum_p];
                
                doubIntStr->mat_n1n2.set_pointer(v_mat_n1n2);
                doubIntStr->mat_n1g2.set_pointer(v_mat_n1g2);
                doubIntStr->mat_g1n2.set_pointer(v_mat_g1n2);
                doubIntStr->mat_g1g2.set_pointer(v_mat_g1g2);
                doubIntStr->mat_n2n1.set_pointer(v_mat_n2n1);
                doubIntStr->mat_n2g1.set_pointer(v_mat_n2g1);
                doubIntStr->mat_g2n1.set_pointer(v_mat_g2n1);
                doubIntStr->mat_g2g1.set_pointer(v_mat_g2g1);

                vector<double> globIntegrals(_nGlobIntegrals);
                vector<double> cellIntegrals_1(_nCellIntegrals);
                vector<double> cellIntegrals_2(_nCellIntegrals);
                
                int offsetDOFs_1       = _cellDOFOffset[glo_n];
                int offsetglobFields_1 = _cellDOFOffset[glo_n]+nPts_1*nNodeDOFs;
                int offsetDOFs_2       = _cellDOFOffset[glo_m];
                int offsetglobFields_2 = _cellDOFOffset[glo_m]+nPts_2*nNodeDOFs;

                int e0 = -1;
                int   eNN_1{};
                int*  adjEN_1 = nullptr;
                
                #pragma omp for
                for(size_t h = 0; h <  _tissue->_elems_inte[inte].size(); h++)
                {
                    auto elems =  _tissue->_elems_inte[inte][h];
                    int e = elems.first;

                    if(e != e0)
                    {
                        
                        if(e0 != -1)
                        {
                            for(int t=0; t < _nGlobIntegrals; t++)
                                globIntegrals[t] += singIntStr_1->globIntegrals[t];
                            
                            for(int t=0; t < _nCellIntegrals; t++)
                                cellIntegrals_1[t] += singIntStr_1->cellIntegrals[t];
        
                            #pragma omp critical
                            AssembleElementalVector( offsetDOFs_1,  eNN_1,  nNodeDOFs, adjEN_1, singIntStr_1->vec_n.data(), _vector);
                            #pragma omp critical
                            _assembleElementalMatrix( offsetDOFs_1,    offsetDOFs_1, eNN_1, eNN_1,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_1, singIntStr_1->mat_nn.data(), _matrix );
                            
                            if(_globalVarInt)
                            {
                                #pragma omp critical
                                AssembleElementalVector( offsetglobFields_1,     1, nGlobDOFs,  &dummy, singIntStr_1->vec_g.data(), _vector);

                                #pragma omp critical
                                _assembleElementalMatrix( offsetDOFs_1,   offsetglobFields_1, eNN_1,     1,  nNodeDOFs, nGlobDOFs, adjEN_1,  &dummy, singIntStr_1->mat_ng.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_1,   offsetDOFs_1,     1, eNN_1, nGlobDOFs,  nNodeDOFs,  &dummy, adjEN_1, singIntStr_1->mat_gn.data(), _matrix );
                                #pragma omp critical
                                _assembleElementalMatrix( offsetglobFields_1,  offsetglobFields_1,     1,     1, nGlobDOFs, nGlobDOFs,  &dummy,  &dummy, singIntStr_1->mat_gg.data(), _matrix );
                            }
                        }
                        
                        eNN_1 = cell_1->_bfs->getNumberOfNeighbours(e);
                        adjEN_1 = cell_1->_bfs->getNeighbours(e);
                        
                        
                        singIntStr_1->elemID = e;
                        singIntStr_1->eNN    = eNN_1;
                        singIntStr_1->nborFields.resize(eNN_1,nNodeFields_1);
                        for(int i=0; i < eNN_1; i++)
                            singIntStr_1->nborFields(i,all)   = nodeFields_1(adjEN_1[i],all);

                        singIntStr_1->vec_n.resize(eNN_1,nNodeDOFs);
                        singIntStr_1->vec_g.resize(nGlobDOFs);
                        singIntStr_1->mat_nn.resize(eNN_1,nNodeDOFs,eNN_1,nNodeDOFs);
                        singIntStr_1->mat_ng.resize(eNN_1,nNodeDOFs,nGlobDOFs);
                        singIntStr_1->mat_gn.resize(nGlobDOFs,eNN_1,nNodeDOFs);
                        singIntStr_1->mat_gg.resize(nGlobDOFs,nGlobDOFs);
                        
                        std::fill(v_outputVector_1.begin(),v_outputVector_1.end(),0.0);
                        std::fill(v_outputMatrix_1.begin(),v_outputMatrix_1.end(),0.0);
//                        std::fill(singIntStr_1->globIntegrals.begin(),singIntStr_1->globIntegrals.end(),0.0); //FIXME: I haven't programmed this part
                    }
                    
                    int g = elems.second;
                    singIntStr_2->elemID = g;

                    int   eNN_2   = cell_2->_bfs->getNumberOfNeighbours(g);
                    int*  adjEN_2 = cell_2->_bfs->getNeighbours(g);

                    singIntStr_2->eNN    = eNN_2;
                    
                    singIntStr_2->nborFields.resize(eNN_2,nNodeFields_2);
                    for(int i=0; i < eNN_2; i++)
                        singIntStr_2->nborFields(i,all)   = nodeFields_2(adjEN_2[i],all);
                
                    singIntStr_2->vec_n.resize(eNN_2,nNodeDOFs);
                    singIntStr_2->vec_g.resize(nGlobDOFs);
                    singIntStr_2->mat_nn.resize(eNN_2,nNodeDOFs,eNN_2,nNodeDOFs);
                    singIntStr_2->mat_ng.resize(eNN_2,nNodeDOFs,nGlobDOFs);
                    singIntStr_2->mat_gn.resize(nGlobDOFs,eNN_2,nNodeDOFs);
                    singIntStr_2->mat_gg.resize(nGlobDOFs,nGlobDOFs);
                    
                    std::fill(v_outputVector_2.begin(),v_outputVector_2.end(),0.0);
                    std::fill(v_outputMatrix_2.begin(),v_outputMatrix_2.end(),0.0);
//                    std::fill(singIntStr_2->globIntegrals.begin(),singIntStr_2->globIntegrals.end(),0.0); //FIXME: I haven't programmed this part

                    doubIntStr->mat_n1n2.resize(eNN_1,nNodeDOFs,eNN_2,nNodeDOFs);
                    doubIntStr->mat_n1g2.resize(eNN_1,nNodeDOFs,nGlobDOFs);
                    doubIntStr->mat_g1n2.resize(nGlobDOFs,eNN_2,nNodeDOFs);
                    doubIntStr->mat_g1g2.resize(nGlobDOFs,nGlobDOFs);
                    doubIntStr->mat_n2n1.resize(eNN_2,nNodeDOFs,eNN_1,nNodeDOFs);
                    doubIntStr->mat_n2g1.resize(eNN_2,nNodeDOFs,nGlobDOFs);
                    doubIntStr->mat_g2n1.resize(nGlobDOFs,eNN_1,nNodeDOFs);
                    doubIntStr->mat_g2g1.resize(nGlobDOFs,nGlobDOFs);
                    
                    std::fill(v_outputMatrix_i.begin(),v_outputMatrix_i.end(),0.0);
                    
                    auto& savedBFs_type_1 = _savedBFs_intera[cell_1->_bfs->getElementType(e)];
                    auto& savedBFs_type_2 = _savedBFs_intera[cell_2->_bfs->getElementType(g)];

                    for(size_t k=0; k < _wSamples_intera.size(); k++)
                    {
                        singIntStr_1->bfs = savedBFs_type_1[k];
                        singIntStr_1->w_sample = _wSamples_intera[k];

                        for(size_t l=0; l < _wSamples_intera.size(); l++)
                        {
                            singIntStr_2->bfs = savedBFs_type_2[l];
                            singIntStr_2->w_sample = _wSamples_intera[l];
                            _doubleIntegrand(doubIntStr);
                        }
                    }
                    
                    //Assemble
                    #pragma omp critical
                    AssembleElementalVector(   offsetDOFs_2, eNN_2,  nNodeDOFs, adjEN_2, singIntStr_2->vec_n.data(), _vector);
                    
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_2,    offsetDOFs_2, eNN_2, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_2, singIntStr_2->mat_nn.data(), _matrix );
                    
                    if(_globalVarInt)
                    {
                        #pragma omp critical
                        AssembleElementalVector(  offsetglobFields_2,     1, nGlobDOFs,  &dummy, singIntStr_2->vec_g.data(), _vector);

                        #pragma omp critical
                        _assembleElementalMatrix( offsetDOFs_2,   offsetglobFields_2, eNN_2,     1,  nNodeDOFs, nGlobDOFs, adjEN_2,  &dummy, singIntStr_2->mat_ng.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_2,   offsetDOFs_2,     1, eNN_2, nGlobDOFs,  nNodeDOFs,  &dummy, adjEN_2, singIntStr_2->mat_gn.data(), _matrix );
                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_2,  offsetglobFields_2,     1,     1, nGlobDOFs, nGlobDOFs,  &dummy,  &dummy, singIntStr_2->mat_gg.data(), _matrix );
                    }

                    //Interaction between cells
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_1, offsetDOFs_2, eNN_1, eNN_2,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_2, doubIntStr->mat_n1n2.data(), _matrix );
                    
                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_2, offsetDOFs_1, eNN_2, eNN_1,  nNodeDOFs,  nNodeDOFs, adjEN_2, adjEN_1, doubIntStr->mat_n2n1.data(), _matrix );

                    if(_globalVarInt)
                    {
                        #pragma omp critical
                        _assembleElementalMatrix(offsetDOFs_1,  offsetglobFields_2, eNN_1,     1,  nNodeDOFs, nGlobDOFs, adjEN_1,  &dummy, doubIntStr->mat_n1g2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_2,  offsetDOFs_1,     1, eNN_1, nGlobDOFs,  nNodeDOFs,  &dummy, adjEN_1, doubIntStr->mat_g2n1.data(), _matrix);

                        #pragma omp critical
                        _assembleElementalMatrix( offsetglobFields_1, offsetDOFs_2,    1, eNN_2, nGlobDOFs,  nNodeDOFs,  &dummy, adjEN_2,  doubIntStr->mat_g1n2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetDOFs_2,  offsetglobFields_1, eNN_2,     1,  nNodeDOFs, nGlobDOFs, adjEN_2,  &dummy, doubIntStr->mat_n2g1.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_1,   offsetglobFields_2,     1,     1, nGlobDOFs, nGlobDOFs,  &dummy,  &dummy, doubIntStr->mat_g1g2.data(), _matrix);
                        #pragma omp critical
                        _assembleElementalMatrix(offsetglobFields_2,   offsetglobFields_1,     1,     1, nGlobDOFs, nGlobDOFs,  &dummy,  &dummy, doubIntStr->mat_g2g1.data(), _matrix);
                    }

                    for(int t=0; t < _nGlobIntegrals; t++)
                        globIntegrals[t] += singIntStr_2->globIntegrals[t];
                    
                    for(int t=0; t < _nCellIntegrals; t++)
                        cellIntegrals_2[t] += singIntStr_2->cellIntegrals[t];
                    
                    e0 = e;
                }

                for(int t=0; t < _nGlobIntegrals; t++)
                    globIntegrals[t] += singIntStr_1->globIntegrals[t];
                
                for(int t=0; t < _nCellIntegrals; t++)
                    cellIntegrals_1[t] += singIntStr_1->cellIntegrals[t];

                #pragma omp critical
                AssembleElementalVector( offsetDOFs_1,  singIntStr_1->eNN,  nNodeDOFs, adjEN_1, singIntStr_1->vec_n.data(), _vector);
                #pragma omp critical
                _assembleElementalMatrix( offsetDOFs_1,    offsetDOFs_1, singIntStr_1->eNN, singIntStr_1->eNN,  nNodeDOFs,  nNodeDOFs, adjEN_1, adjEN_1, singIntStr_1->mat_nn.data(), _matrix );
                
                if(_globalVarInt)
                {
                    #pragma omp critical
                    AssembleElementalVector( offsetglobFields_1,     1, nGlobDOFs,  &dummy, singIntStr_1->vec_g.data(), _vector);

                    #pragma omp critical
                    _assembleElementalMatrix( offsetDOFs_1,   offsetglobFields_1, singIntStr_1->eNN,     1,  nNodeDOFs, nGlobDOFs, adjEN_1,  &dummy, singIntStr_1->mat_ng.data(), _matrix );
                    #pragma omp critical
                    _assembleElementalMatrix( offsetglobFields_1,   offsetDOFs_1,     1, singIntStr_1->eNN, nGlobDOFs,  nNodeDOFs,  &dummy, adjEN_1, singIntStr_1->mat_gn.data(), _matrix );
                    #pragma omp critical
                    _assembleElementalMatrix( offsetglobFields_1,  offsetglobFields_1,     1,     1, nGlobDOFs, nGlobDOFs,  &dummy,  &dummy, singIntStr_1->mat_gg.data(), _matrix );
                }
                
                #pragma omp critical
                std::transform (_globIntegrals.begin(), _globIntegrals.end(), globIntegrals.begin(), _globIntegrals.begin(), std::plus<double>());
//                #pragma omp critical
//                _cellIntegrals[n] = cellIntegrals; //FIXME: fix this!!!
            }
        }
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
            
//            #pragma omp parallel for
            for(int i = 0; i < nPts; i++)
            {
                for(size_t m=0; m < _nodeDOFIdx.size(); m++)
                {
                    dof_raw_ptr[i*nFields+_nodeDOFIdx[m]] += scale * sol_raw_ptr[0][t];
                    t++;
                }
            }
            
            double* gdof_raw_ptr = cell->_globFields.data();
            for(size_t g = 0; g < _globDOFIdx.size(); g++)
            {
                gdof_raw_ptr[_globDOFIdx[g]] += scale * sol_raw_ptr[0][t];
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
            
//            #pragma omp parallel for
            for(int i = 0; i < nPts; i++)
            {
                for(size_t m=0; m < _nodeDOFIdx.size(); m++)
                {
                    dof_raw_ptr[i*nDOFs+_nodeDOFIdx[m]] = scale * sol_raw_ptr[0][t];
                    t++;
                }
            }
            
            double* gdof_raw_ptr = cell->_globFields.data();
            for(size_t g = 0; g < _globDOFIdx.size(); g++)
            {
                gdof_raw_ptr[_globDOFIdx[g]] = scale * sol_raw_ptr[0][t];
                t++;
            }
        }
    }
}
