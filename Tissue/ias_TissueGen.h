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
            Teuchos::RCP<Tissue> genRegularGridSpheres(int nx, int ny, int nz, double deltax, double deltay, double deltaz, double r, int nSubdiv);
        
            /** @name Setters (prior to Update())
            *  @{ */
            /*! @brief Add a new field to the nodes with name newField*/
            void addNodeField(std::string newField)
            {    _nodeFieldNames.push_back(newField);    }
            /*! @brief Add a new global field with name newField */
            void addGlobField(std::string newField)
            {    _globFieldNames.push_back(newField);    }
            /*! @brief Add a new tissue field with name newField */
            void addTissField(std::string newField)
            {    _tissFieldNames.push_back(newField);    }
            void setBasisFunctionType(BasisFunctionType bfType)
            {    _bfType = bfType;    }
            /** @} */

        private:
            BasisFunctionType _bfType; ///<Type of basis functions
            std::vector<std::string> _nodeFieldNames; ///<List of names for the nodal fields (x,y,z always included)
            std::vector<std::string> _globFieldNames;      ///<List of names for the global fields (cellId always included)
            std::vector<std::string> _tissFieldNames;      ///<List of names for the global fields (cellId always included)
    };

}
#endif //_TissueGen_h
