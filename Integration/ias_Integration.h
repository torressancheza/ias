#ifndef _Integration_h
#define _Integration_h

#include <Teuchos_RCP.hpp>
#include <Epetra_MpiComm.h>
#include <Epetra_FEVector.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_LinearProblem.h>

#include "ias_Tissue.h"
#include "ias_CubatureGauss.h"
#include "ias_BasicStructures.h"
#include "ias_Assembly.h"

namespace ias
{
    
    class Integration
    {
        public:
                          
            /** @name Constructor/destructor
            *  @{ */
            /*! @brief Constructor
             *  @param comm   MPI_Communicator (default value MPI_COMM_WORLD) */
            Integration(){}
            /*! @brief Destructor             */
            ~Integration(){};
            /** @} */

            /** @name Copy/move constructors/assignments
            *  @{ */
            /*! @brief Copy contructor deleted */
            Integration(const Integration&)             = delete;
            /*! @brief Copy assignment deleted */
            Integration& operator=(const Integration&)  = delete;
            /*! @brief Move constructor defaulted */
            Integration(Integration&&)                  = default;
            /*! @brief Copy constructor defaulted */
            Integration& operator=(Integration&&)       = default;
            /** @} */
                    
            /** @name Important setters/getters
             *  @{ */
             /*! @brief Set tissue */
            void setTissue( Teuchos::RCP<Tissue> tissue )
            {    _tissue = tissue;    }
            /*! @brief Get tissue */
            const Teuchos::RCP<Tissue> getTissue ( ) const
            {    return _tissue;    }
            /*! @brief Set number of integration points  */
            void setNumberOfIntegrationPointsSingleIntegral(int iPts_single)
            {    _iPts_single = iPts_single;    }
            void setNumberOfIntegrationPointsDoubleIntegral(int iPts_intera)
            {    _iPts_intera = iPts_intera;    }
            /*! @brief Set degrees of freedom at nodes */
            void setNodeDOFs(std::vector<std::string> setNodeDOFs)
            {
                for(auto s: setNodeDOFs)
                    _nodeDOFNames.push_back(s);
            }
            /*! @brief Set global degrees of freedom (per cell) */
            void setGlobalDOFs(std::vector<std::string> setGlobDOFs)
            {
                for(auto s: setGlobDOFs)
                    _globDOFNames.push_back(s);
            }
            /*! @brief Set function to be integrated at single-cell level  */
            void setSingleIntegrand( void (*singleIntegrand)(Teuchos::RCP<SingleIntegralStr>) )
            {    _singleIntegrand = singleIntegrand;    }
            /*! @brief Set function to be integrated as a double integral (over two cells)  */
            void setDoubleIntegrand(void (*doubleIntegrand)(Teuchos::RCP<DoubleIntegralStr>))
            {    _doubleIntegrand = doubleIntegrand;    }
            /*! @brief Set number of global integrals */
            void setNumberOfGlobalIntegrals(int nGlobIntegrals)
            {    _nGlobIntegrals=nGlobIntegrals;    }
            /*! @brief Set number of integrals to be stored at each cell */
            void setNumberOfCellIntegrals(int nCellIntegrals)
            {    _nCellIntegrals=nCellIntegrals;    }
//            void setNumberOfCellInteractionFields(int nCellIntFields)
//            {    _nCellIntFields=nCellIntFields;    }
            /** @} */

            void Update();

            /** @name Integrate
            *  @{ */
            /*! @brief Compute integrals at a single cell level */
            void computeSingleIntegral();
            /*! @brief Compute idouble integrals */
            void computeDoubleIntegral();
            /** @} */

            /** @name Assembly
            *  @{ */
            /*! @brief Assemble vector and matrix*/
            void assemble()
            {
                _vector->GlobalAssemble();
                _matrix->GlobalAssemble();
                if ( reinterpret_cast<void*>(_assembleElementalMatrix) == reinterpret_cast<void*>(AssembleElementalMatrix<true>))
                    _assembleElementalMatrix = AssembleElementalMatrix<false>;
            }
            void recalculateMatrixStructure()
            {
                _matrix = Teuchos::rcp(new Epetra_FECrsMatrix(Copy, Epetra_Map(_matrix->RowMap()), -1));
                _linProbl = Teuchos::rcp(new Epetra_LinearProblem(&(*_matrix ), &(*_sol), &(*_vector)));
                _assembleElementalMatrix = AssembleElementalMatrix<true>;
            }
            /** @} */
        
            /** @name Initialise vector and matrix with a value
            *  @{ */
            /*! @brief Fill vector with a value*/
            void fillVectorWithScalar(double scalar)
            {    _vector->PutScalar(scalar);    }
            /*! @brief Fill matrix with a value*/
            void fillMatrixWithScalar(double scalar)
            {    _matrix->PutScalar(scalar);    }
            void fillSolutionWithScalar(double scalar)
            {    _sol->PutScalar(scalar);    }
            void InitialiseGlobalIntegrals(double scalar = 0.0)
            {    std::fill(_globIntegrals.begin(),_globIntegrals.end(),scalar);    }

            void InitialiseCellIntegrals(double scalar=0.0)
            {
                for(int n = 0; n < _tissue->getLocalNumberOfCells(); n++)
                    std::fill(_cellIntegrals[n].begin(),_cellIntegrals[n].end(),scalar);
            }
            /** @} */

            /** @name Update fields in tissue according to solution (or vector) computed in this integration
            *  @{ */
            /*! @brief Add solution (times scaling factor) to degrees of freedom */
            void addSolToDOFs(double scale=1.0)
            {    _addArrayToDOFs(_sol,scale);    }
            /*! @brief Add vector (times scaling factor) to degrees of freedom */
            void addVecToDOFs(double scale=1.0)
            {    _addArrayToDOFs(_vector,scale);    }
            /*! @brief Set solution (times scaling factor) to degrees of freedom */
            void setSolToDOFs(double scale=1.0)
            {    _setArrayToDOFs(_sol,scale);    }
            /*! @brief Set vector (times scaling factor) to degrees of freedom */
            void setVecToDOFs(double scale=1.0)
            {    _setArrayToDOFs(_vector,scale);    }
            void setCellIntegralToCellField(int i, int j)
            {
                for(int n = 0; n < _tissue->getLocalNumberOfCells(); n++)
                    _tissue->_cells[n]->_globFields(i) = _cellIntegrals[n][j];
            }
            void addCellIntegralToCellField(int i, int j)
            {
                for(int n = 0; n < _tissue->getLocalNumberOfCells(); n++)
                    _tissue->_cells[n]->_globFields(i) += _cellIntegrals[n][j];
            }
            /** @} */
        
            /** @name Get computed objects
            *  @{ */
            const Teuchos::RCP<Epetra_LinearProblem> getLinearProblem() const
            {    return _linProbl;    }
            const Teuchos::RCP<Epetra_FEVector> getVector() const
            {    return _vector;    }
            const Teuchos::RCP<Epetra_FEVector> getSolution() const
            {    return _sol;    }
            /** @} */

            double getGlobalIntegral(int i ) const
            {    return _globIntegrals[i];  }
        
            double getCellIntegral(int i, int j ) const
            {    return _cellIntegrals[i][j];  }
        
            void setGlobalVariablesInInteractions(bool globalVarInt)
            {   _globalVarInt = globalVarInt;   }
        
//        private:
        
            void (*_singleIntegrand)(Teuchos::RCP<SingleIntegralStr>) = nullptr;
            void (*_doubleIntegrand)(Teuchos::RCP<DoubleIntegralStr>) = nullptr;
        
            Teuchos::RCP<Tissue>         _tissue = Teuchos::null;

            std::vector<int> _cellDOFOffset;
            
            Teuchos::RCP<Epetra_FECrsMatrix>     _matrix = Teuchos::null;
            Teuchos::RCP<Epetra_FEVector>        _vector = Teuchos::null;
            Teuchos::RCP<Epetra_FEVector>           _sol = Teuchos::null;
            Teuchos::RCP<Epetra_LinearProblem> _linProbl = Teuchos::null;

            int _nGlobIntegrals{};
            std::vector<double> _globIntegrals;     ///< Global integrals
            int _nCellIntegrals{};
            std::vector<std::vector<double>> _cellIntegrals; ///< Integrals to be computed per cell

//            int _nCellIntFields;
//            std::vector<Teuchos::RCP<Epetra_FEVector>> _cellIntFields; ///< Fields at Gauss points of the cell which depend on interations with other cells

            int _iPts_single{};
            std::vector<double> _wSamples_single;
            std::vector<std::vector<std::vector<std::vector<double>>>> _savedBFs_single; ///< Basis functions at gauss points (iPts*(order+1)*eNN)
                    
            int _iPts_intera{};
            std::vector<double> _wSamples_intera;
            std::vector<std::vector<std::vector<std::vector<double>>>> _savedBFs_intera; ///< Basis functions at gauss points (iPts*(order+1)*eNN)
                    
            
            std::vector<int> _nodeDOFIdx;
            std::vector<int> _globDOFIdx;
            std::vector<std::string> _nodeDOFNames;
            std::vector<std::string> _globDOFNames;
            std::map<std::string,int> _mapNodeDOFNames;
            std::map<std::string,int> _mapGlobDOFNames;
        
            bool _globalVarInt{true};
            
            void (*_assembleElementalMatrix)(size_t offset_1, size_t offset_2, size_t eNN_1, size_t eNN_2, size_t nDOFs_1, size_t nDOFs_2, int* loc2glo_1, int* loc2glo_2, double* Ael, Teuchos::RCP<Epetra_FECrsMatrix> A) = nullptr;
            
            void _addArrayToDOFs(Teuchos::RCP<Epetra_FEVector> vec, double scale);
            void _setArrayToDOFs(Teuchos::RCP<Epetra_FEVector> vec, double scale);

    };
}

#endif //_Integration_h
