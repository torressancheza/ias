//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _Cell_h
#define _Cell_h

#include <Teuchos_RCP.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellLocator.h>
#include <vtkUnstructuredGrid.h>


#include "Tensor.h"
#include "ias_BasisFunction.h"

namespace ias
{


    /*! @class Cell
     *  @brief A cell is made of a mesh, basis functions for  interpolation, and set of fields both global (such as pressure or polarity) and at the nodes (such as the nodes' positions, velocities, or some concentration)*/
    class Cell
    {
        public:
            /** @name Constructor/destructor
            *  @{ */
            /*! @brief Constructor */
            Cell(){};
            /*! @brief Destructor */
            ~Cell(){};
            /** @} */
        
            /** @name Copy/move constructors/assignments
            *  @{ */
            /*! @brief Copy contructor deleted   */
            Cell(const Cell&)             = delete;
            /*! @brief Copy assignment deleted */
            Cell& operator=(const Cell&)  = delete;
            /*! @brief Move constructor defaulted */
            Cell(Cell&&)                  = default;
            /*! @brief Copy constructor defaulted */
            Cell& operator=(Cell&&)       = default;
            /** @} */
            
            /** @name Update
            *  @{ */
            /*! @brief Use the information given by the user through setters to initialise (or resize) internal variables */
            void Update();
            /** @} */
            
            /** @name Setters (prior to Update())
            *  @{ */
            /*! @brief Add a new field to the nodes with name newField*/
            void addNodeField(std::string newField)
            {    _nodeFieldNames.push_back(newField);    }
            /*! @brief Add a new cell field with name newField */
            void addCellField(std::string newField)
            {    _cellFieldNames.push_back(newField);    }
            /*! @brief Set node positions as a matrix (number of points * 3)*/
            void setNodePositions(Tensor::tensor<double,2> nodePos)
            {    _nodeFields.size()==0 ? _nodeFields = nodePos : _nodeFields(Tensor::all,Tensor::range(0,2)) = nodePos;    }
            /*! @brief Set mesh connectivity*/
            void setConnectivity(Tensor::tensor<int,2> connec)
            {    _connec = connec;    }
            /*! @brief Set type of basis functions*/
            void setBasisFunctionType(BasisFunctionType bfType)
            {    _bfType = bfType;    }
            /** @} */
        
            /** @name Mesh generation
            *  @{ */
            /*! Generate a sphere of radius r by subdividing an octahedron nSubdiv times*/
            void generateSphereFromOctahedron(int nSubdiv, double r);
            /** @} */

        
            /** @name Getters
             *  @{ */
            /*! @brief Get the number of points*/
            int getNumberOfPoints( ) const
            {    return _nodeFields.shape()[0];    }
            /*! @brief Get the number of fields at the nodes*/
            int getNumberOfNodeFields( ) const
            {    return _nodeFields.shape()[1];    }
            /*! @brief Get the number of cell fields*/
            int getNumberOfCellFields( ) const
            {    return _cellFields.shape()[0];    }
            /*! @brief Get the number of elements in the mesh*/
            int getNumberOfElements( ) const
            {    return _connec.shape()[0];    }
            /*! @brief Get the number of vertices per element*/
            int getNumberOfVerticesPerElement() const
            {    return _connec.shape()[1];    }
        
            /*! @brief Get the node fields (a shallow copy is created so that the user cannot change the internal structure of the tensor)*/
            Tensor::tensor<double,2> getNodeFields()
            {    return Tensor::tensor<double,2>(_nodeFields.data(),_nodeFields.shape()[0], _nodeFields.shape()[1]);    }
            /*! @brief Get the node fields interpolated at a given parametric coordinates of an element*/
            Tensor::tensor<double,1> getInterpolatedNodeFields(std::vector<double> xi, int e);   
            /*! @brief Get  the fields of node n*/
            Tensor::tensor<double,1> getNodeFields(int n)
            {
                if(n<_nodeFields.shape()[0])
                    return _nodeFields(n,Tensor::all);
                else
                    throw std::runtime_error("Cell::getNodeFields: Node " + std::to_string(n) + " does not exist!" + " Did you call Update()?");
            }
            /*! @brief Get the ith node field*/
            Tensor::tensor<double,1> getNodeField(int i)
            {
                if(i<_nodeFields.shape()[1])
                    return _nodeFields(Tensor::all,i);
                else
                    throw std::runtime_error("Cell::getNodeField: Field " + std::to_string(i) + " is beyond the size of node fields (" + std::to_string(_nodeFields.shape()[1])+"). Did you call Update()?");
            }
            /*! @brief Get the node field with the given label*/
            Tensor::tensor<double,1> getNodeField(std::string label)
            {
                try
                {
                    return getNodeField(_mapNodeFieldNames.at(label));
                }
                catch (const std::out_of_range& error)
                {
                    throw std::runtime_error("Cell::getNodeField: Field \"" + label + "\" is not in the list of node fields. Did you call Update()?");
                }
            }

            /*! @brief Get the cell fields (a shallow copy is created so that the user cannot change the internal structure of the tensor)*/
            Tensor::tensor<double,1> getCellFields()
            {    return Tensor::tensor<double,1>(_cellFields.data(),_cellFields.size());    }
            /*! @brief Get the ith cell field */
            double& getCellField(int i)
            {
                if(i<_cellFields.size())
                    return _cellFields(i);
                else
                    throw std::runtime_error("Cell::getCellField: Field " + std::to_string(i) + " is beyond the size of cell fields (" + std::to_string(_cellFields.size())+"). Did you call Update()?");
            }
            /*! @brief Get the cell field with the given label*/
            double& getCellField(std::string label)
            {
                try
                {
                    return getCellField(_mapCellFieldNames.at(label));
                }
                catch (const std::out_of_range& error)
                {
                    throw std::runtime_error("Cell::getCellField: Field \"" + label + "\" is not in the list of cell fields. Did you call Update()?");
                }
            }
            /*! @brief Get the type of basis functions*/
            BasisFunctionType getBasisFunctionType() const
            {    return _bfType;    }

            /*! @brief Get the bounding box of the cell*/
            Tensor::tensor<double,2> getBoundingBox(double eps = 0.0);

            /*! @brief Get a copy*/
            Teuchos::RCP<Cell> getCopy();

            /*! @brief Get list of elements in the box*/
            std::vector<int> getElementsInBox(Tensor::tensor<double,2> box);
        
            /*! @brief Get a copy of the node field names*/
            std::vector<std::string> getNodeFieldNames() const
            {    return _nodeFieldNames;    }
            /*! @brief Get a copy of the cell field names*/
            std::vector<std::string> getCellFieldNames() const
            {    return _cellFieldNames;    }
            /** @} */

            /** @name IO
            *  @{ */
            /*! @brief Save cell in VTK format (.vtu file)*/
            void saveVTK(std::string namefile);
            /*! @brief Load cell in VTK format (.vtu file) and assign*/
            void loadVTK(std::string namefile);
            /*! @brief Get a polydata for further manipulation*/
            vtkSmartPointer<vtkPolyData> getPolyData(bool withNodeFields = false, bool withGlobFields  = false);
            /** @} */

        
            /** @name Mesh modifiers
            *  @{ */
            /*! @brief Compute the mesh quality*/
            double getMeshQuality();
            /*! @brief Create a new mesh to reduce element distortion */
            void remesh(double tArea = 0.05);
            /*! @brief Divide the cell creating two daughters separated by a distance sep. First daughter replaces the cell, the other daughter is given as an output.  */
            Teuchos::RCP<Cell> cellDivision(double sep, double elArea, std::vector<double> planeNormal = {}, std::vector<double> planeCentre = {});
            /** @} */

        private:
        
            Tensor::tensor<int,2>    _connec; ///<Mesh connectivity for the cell (Input)
            BasisFunctionType _bfType = BasisFunctionType::Undefined; ///<Type of basis functions (Input)
            std::vector<std::string> _nodeFieldNames = {"x", "y", "z"}; ///<List of names for the nodal fields (x,y,z always included)
            std::vector<std::string> _cellFieldNames = {"cellId"};      ///<List of names for the cell fields (cellId always included)
        
            Teuchos::RCP<BasisFunction> _bfs = Teuchos::null; ///<Pointer to basis function object

            Tensor::tensor<double,2> _nodeFields;  ///<Tensor storing the fields at the nodes
            Tensor::tensor<double,1> _cellFields;  ///<Tensor storing  the fields at the nodes
        
            std::map<std::string,int> _mapNodeFieldNames; ///<Map name to field number for nodal fields
            std::map<std::string,int> _mapCellFieldNames; ///<Map name to field number for cell fields

            void _removePoints3Neighbours(vtkSmartPointer<vtkPolyData> polydata);

        friend class Tissue;
        friend class TissueGen;
        friend class Integration;
        friend void mapFields(Teuchos::RCP<Cell> cell1, Teuchos::RCP<Cell> cell2);

    };

    void mapFields(Teuchos::RCP<Cell> cell1, Teuchos::RCP<Cell> cell2);

}
#endif //_Cell_h
