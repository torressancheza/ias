//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _Tissue_h
#define _Tissue_h

#include <string>
#include <iostream>

#include <Teuchos_RCP.hpp>
#include <Epetra_FECrsGraph.h>

#include <iostream>
#include <stdexcept>
#include <numeric>

#include "mpi.h"
#include "ias_Cell.h"

namespace ias
{
    enum class idxType
    {
        local,
        global
    };

    class Tissue
    {
        public:
            /** @name Constructor/destructor
            *  @{ */
            /*! @brief Constructor
             *  @param comm   MPI_Communicator (default value MPI_COMM_WORLD) */
            Tissue(MPI_Comm comm = MPI_COMM_WORLD)
            {
                _comm = comm;
                MPI_Comm_rank(_comm, &_myPart);
                MPI_Comm_size(_comm, &_nParts);

                _nCellPart.resize(_nParts);
                _offsetPart.resize(_nParts+1);
            }
            /*! @brief Destructor             */
            ~Tissue(){};
            /** @} */

            /** @name Copy/move constructors/assignments
            *  @{ */
            /*! @brief Copy contructor deleted */
            Tissue(const Tissue&)             = delete;
            /*! @brief Copy assignment deleted */
            Tissue& operator=(const Tissue&)  = delete;
            /*! @brief Move constructor defaulted */
            Tissue(Tissue&&)                  = default;
            /*! @brief Copy constructor defaulted */
            Tissue& operator=(Tissue&&)       = default;
            /** @} */
        
            /** @name Add cells and compute total numbers and offsets
            *  @{ */
            /*! @brief Add cell to the tissue (locally in this partition)  */
            void addCellToTissue(Teuchos::RCP<Cell> cell)
            {    _cells.emplace_back(cell);    }
            /*! @brief Compute total number of cells, offsets, etc  */
            void Update();
            /** @} */
        
            /** @name IO
            *  @{ */
            void saveVTK(std::string prefix,std::string suffix);
            void loadVTK(std::string prefix,std::string suffix, BasisFunctionType bfType);
            /** @} */

            /** @name Cell-cell adjancency and distribution
            *  @{ */
            /*! @brief For each cell, compute its bounding box. This is used to calculate cell-cell adjacency
             *  @param delta two cells will be adjacent if their bounding boxes (expanded by delta) overlap */
            void calculateCellCellAdjacency(double delta);
            /*! @brief Balance the distribution of cells amongst partitions */
            void balanceDistribution();
            /*! @brief send and receive all ghost cell info (fields) */
            void updateGhosts(bool updateNodeFields = true, bool updateGlobalFields = true, bool UpdateConnectivity = false);
            /*! @brief send and receive all ghost cell info (fields) */
            bool calculateInteractingElements(double eps);
            /** @} */

            
            /** @name Basic parallel info
             *  @{ */
            /*! @brief Get number of partitions */
            int getNumParts()
            {    return _nParts;    }
            /*! @brief Get which partition you are */
            int getMyPart()
            {    return _myPart;    }
            /*! @brief Get number of cells in this partition */
            int getNumberOfCells()
            {   return _nCells;    }
        
            int getLocalNumberOfCells()
            {   return _cells.size();    }
            /*! @brief Check whether global index i is in this partition */
            bool isItemInPart(int i)
            {
                if(_offsetPart[_myPart] <= i and _offsetPart[_myPart+1] > i)
                    return true;
                else
                    return false;
            }
            /*! @brief Return in which partition the global index i is */
            int whichPart(int i)
            {
                if(i >= _nCells)
                    throw std::runtime_error("The index("+std::to_string(i)+") is larger than the number of cells ("+std::to_string(_nCells)+").");
                for(int n = _nParts-1; n >= 0 ; n--)
                {
                    if (_offsetPart[n] <= i)
                        return n;
                }
                throw std::runtime_error("The index("+std::to_string(i)+") was not found.");
            }
            /*! @brief Get global index of cell from local index loc_i*/
            int getGlobalIdx(int loc_i)
            {    return _offsetPart[getMyPart()] + loc_i;    }
            /*! @brief Get local index of cell from global index (it throws runtime_error if index is not present in the partition) */
            int getLocalIdx(int i)
            {
                if (isItemInPart(i))
                    return i - _offsetPart[_myPart];
                else
                    throw std::runtime_error("Item "+std::to_string(i)+" is not in partition " + std::to_string(getMyPart()));
            }
            Teuchos::RCP<Cell> GetCell(int i, idxType idType = idxType::global)
            {
                Teuchos::RCP<Cell> cell{};
                if(idType ==  idxType::local)
                    cell = _cells[i];
                else if(isItemInPart(i))
                    cell = _cells[getLocalIdx(i)];
                else
                {
                    auto itr = std::find(_inGhostLabels.begin(),_inGhostLabels.end(),i);
                    auto idx = std::distance(_inGhostLabels.begin(), itr);

                    if(size_t(idx) == _inGhostLabels.size())
                        throw std::runtime_error("Tissue::GetCell: Cell " + std::to_string(i) + " does not belong to partition " + std::to_string(_myPart) + " and is not in the list of ghosts.");

                    cell = _inGhostCells[idx];
                }
                return cell;
            }
        
            /*! @brief Get index of the node field*/
            int getNodeFieldIdx(std::string fieldName)
            {    return _mapNodeFieldNames[fieldName];    }
            /*! @brief Get index of the cell field*/
            int getCellFieldIdx(std::string fieldName)
            {    return _mapCellFieldNames[fieldName];    }
            /*! @brief Get index of the tissue field*/
            int getTissFieldIdx(std::string fieldName)
            {    return _mapTissFieldNames[fieldName];    }
        
            std::vector<Teuchos::RCP<Cell>> getLocalCells()
            {    return _cells;    }
        
            void openPVD(std::string suffix);
            void closePVD();
        
            /*! @brief Get the ith global field */
            double& getTissField(int i)
            {
                if(i<_tissFields.size())
                    return _tissFields(i);
                else
                    throw std::runtime_error("Tissue::getTissField: Field " + std::to_string(i) + " is beyond the size of tissue fields (" + std::to_string(_tissFields.size())+"). Did you call Update()?");
            }
            /*! @brief Get the global field with the given label*/
            double& getTissField(std::string label)
            {
                try
                {
                    return getTissField(_mapTissFieldNames.at(label));
                }
                catch (const std::out_of_range& error)
                {
                    throw std::runtime_error("Tissue::getTissField: Field \"" + label + "\" is not in the list of tissue fields. Did you call Update()?");
                }
            }
        
            void cellDivision(std::vector<int> cellIds, double sep, double elArea);

            /** @} */

        private:
        
            /** @name Basic parallel variables
            *  @{ */
            MPI_Comm _comm = MPI_COMM_WORLD;
            int _myPart{};                    ///<(local) Label of this partition
            int _nParts{};                    ///<(shared) Number of partitions
            std::vector<int>  _offsetPart;    ///<(shared) Label of the first cell in each partition
            std::vector<int>  _nCellPart;     ///<(shared) Label of the first cell in each partition
            /** @} */

            /** @name Main variables
            *  @{ */
            std::vector<Teuchos::RCP<Cell>> _cells;                  ///<(local) Cells in this partition
            std::vector<int>                _cellFieldOffset;        ///<(local) Jumps in the fields from one cell to the next (used mostly for addVectorToFields)
            Teuchos::RCP<Epetra_FECrsGraph> _adjyCC = Teuchos::null; ///<(distrib) Cell-cell adjacency
            std::vector<std::vector<std::pair<int,int>>> _inters;
            std::vector<std::vector<std::pair<int,int>>> _elems_inte;
            int _nCells{};                                           ///<(shared) Total number of cells
        
            double _eps{};
        
            Tensor::tensor<double,1> _tissFields;
        
            std::vector<std::string> _nodeFieldNames;     ///<List of names for the nodal fields (x,y,z always included)
            std::vector<std::string> _cellFieldNames;     ///<List of names for the global fields (cellId always included)
            std::vector<std::string> _tissFieldNames;     ///<List of names for the tissue fields
            std::map<std::string,int> _mapNodeFieldNames; ///<Map name to field number for nodal fields
            std::map<std::string,int> _mapCellFieldNames; ///<Map name to field number for global fields
            std::map<std::string,int> _mapTissFieldNames; ///<Map name to field number for tissue fields
            /** @} */
        
            /** @name Ghosts
            *  @{ */
            std::vector<int>  _inGhostLabels;                ///<Labels of the ghosts in this partition
            std::vector<int>  _inGhostOffsetPart;            ///<List kind of object to detect boundaries of partitions in _ghost_labels
            std::vector<Teuchos::RCP<Cell>> _inGhostCells;   ///<Ghost cells in this partition (same order as _ghost_labels)

            std::vector<int>  _outGhostLabels;               ///<Labels of the outgoing ghosts (cells to be sent to other partitions)
            std::vector<int>  _outGhostOffsetPart;           ///<List kind of object to detect boundaries of partitions in _ghost_labels
            
            //Information sent by this process (it may contain repeated data) -> it has to be copied whenever information neeeds to be sent
            std::vector<double> _outGhost_nodeFields;   ///<Long vector with all DOFs to be sent to other partitions
            std::vector<double> _outGhost_nodeFields0;  ///<Long vector with all DOFs0 to be sent to other partitions
            std::vector<double> _outGhost_cellFields;   ///<Long vector with all globFields to be sent to other partitions
            std::vector<double> _outGhost_cellFields0;  ///<Long vector with all globFields0 to be sent to other partitions
        
            std::vector<int> _outGhost_connec;  ///<Long vector with all globFields0 to be sent to other partitions

            std::vector<int> _outGhost_nodeFields_offsetCells;  ///<List kind of object containing the boundaries of Cells in _OutGhostDOFs and _OutGhostDOFs0
            std::vector<int> _outGhost_cellFields_offsetCells;  ///<List kind of object containing the boundaries of Cells in _OutGhostglobFields and _OutGhostglobFields0
            
            std::vector<int> _outGhost_connec_offsetCells;  ///<Long vector with all globFields0 to be sent to other partitions
        
            std::vector<int> _outGhost_nodeFields_offsetParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostDOFs and _OutGhostDOFs0
            std::vector<int> _outGhost_cellFields_offsetParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostglobFields and _OutGhostglobFields0
            
            std::vector<int> _outGhost_connec_offsetParts;  ///<Long vector with all globFields0 to be sent to other partitions

        
            std::vector<int> _outGhost_nodeFields_countParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostDOFs and _OutGhostDOFs0
            std::vector<int> _outGhost_cellFields_countParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostglobFields and _OutGhostglobFields0
            
            std::vector<int> _outGhost_connec_countParts;  ///<Long vector with all globFields0 to be sent to other partitions

        
            //Information received by this process (no repetition) -> cell DOFs point to this data (no copy needed)
            std::vector<double> _inGhost_nodeFields;   ///<Long vector with all DOFs received from other partitions
            std::vector<double> _inGhost_nodeFields0;  ///<Long vector with all DOFs0 received from other partitions
            std::vector<double> _inGhost_cellFields;   ///<Long vector with all globFields received from other partitions
            std::vector<double> _inGhost_cellFields0;  ///<Long vector with all globFields0 received from other partitions
            
            std::vector<int> _inGhost_connec;          ///<Long vector with all DOFs received from other partitions

            std::vector<int> _inGhost_nodeFields_offsetCells; ///<List kind of object containing the boundaries of Cells in _InGhostDOFs and _InGhostDOFs0
            std::vector<int> _inGhost_cellFields_offsetCells; ///<List kind of object containing the boundaries of Cells in _InGhostglobFields and _InGhostglobFields0
            
            std::vector<int> _inGhost_connec_offsetCells;  ///<Long vector with all globFields0 to be sent to other partitions
        
            std::vector<int> _inGhost_nodeFields_offsetParts; ///<List kind of object containing the boundaries of Cells in _OutGhostDOFs and _OutGhostDOFs0
            std::vector<int> _inGhost_cellFields_offsetParts; ///<List kind of object containing the boundaries of Cells in _OutGhostglobFields and _OutGhostglobFields0
            
            std::vector<int> _inGhost_connec_offsetParts;  ///<Long vector with all globFields0 to be sent to other partitions

        
            std::vector<int> _inGhost_nodeFields_countParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostDOFs and _OutGhostDOFs0
            std::vector<int> _inGhost_cellFields_countParts;  ///<List kind of object containing the boundaries of Cells in _OutGhostglobFields and _OutGhostglobFields0

            std::vector<int> _inGhost_connec_countParts;  ///<Long vector with all globFields0 to be sent to other partitions

        
            /*! @brief Generate the importers/exporters to get ghost cells */
            void _genGhostImportersExporters();
            void _checkFieldNames();
            size_t _checkCellIds();
            /** @} */
        
        friend class Integration;
        friend class TissueGen;
    };
}

#endif //_Tissue_h
