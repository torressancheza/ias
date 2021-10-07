//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _TissueGen_h
#define _TissueGen_h

#include <Teuchos_RCP.hpp>
#include <Epetra_FECrsGraph.h>

#include <iostream>
#include <stdexcept>

#include "mpi.h"
#include "ias_Tissue.h"

namespace ias
{
    class TissueGen
    {
        public:
            Teuchos::RCP<Tissue> genRegularGridSpheres(int nx, int ny, int nz, double deltax, double deltay, double deltaz, double r, int nSubdiv, int type = VTK_SOLID_ICOSAHEDRON);
            Teuchos::RCP<Tissue> genTripletSpheres(double r, double delta,int nSubdiv, int type = VTK_SOLID_ICOSAHEDRON);
            Teuchos::RCP<Tissue> genSpheroid(double r,double delta, int cellsubdiv, int type, int spheroidsubdiv);

            /** @name Setters (prior to Update())
            *  @{ */
            /*! @brief Add a new field to the nodes with name newField*/
            void addNodeField(std::string newField)
            {    _nodeFieldNames.push_back(newField);    }
            /*! @brief Add new fields to nodes with name newField*/
            void addNodeFields(std::vector<std::string> newFields)
            {
                for(auto& f: newFields)
                    _nodeFieldNames.push_back(f);
            }
            /*! @brief Add a new global field with name newField */
            void addCellField(std::string newField)
            {    _cellFieldNames.push_back(newField);    }
            /*! @brief Add new fields to cells with name newField*/
            void addCellFields(std::vector<std::string> newFields)
            {
                for(auto& f: newFields)
                    _cellFieldNames.push_back(f);
            }
            /*! @brief Add a new tissue field with name newField */
            void addTissField(std::string newField)
            {    _tissFieldNames.push_back(newField);    }
            /*! @brief Add new fields to tissue with name newField*/
            void addTissFields(std::vector<std::string> newFields)
            {
                for(auto& f: newFields)
                    _tissFieldNames.push_back(f);
            }
            void setBasisFunctionType(BasisFunctionType bfType)
            {    _bfType = bfType;    }
            /** @} */
        
        private:
            BasisFunctionType _bfType; ///<Type of basis functions
            std::vector<std::string> _nodeFieldNames; ///<List of names for the nodal fields (x,y,z always included)
            std::vector<std::string> _cellFieldNames; ///<List of names for the cell fields (cellId always included)
            std::vector<std::string> _tissFieldNames; ///<List of names for the tissue fields 
        
            /*! @brief Check field names*/
            void _checkFieldNames();
    };

}
#endif //_TissueGen_h
