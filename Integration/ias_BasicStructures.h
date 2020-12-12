//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _BasicStructures_h
#define _BasicStructures_h

#include "Tensor.h"

namespace ias
{
    struct SingleIntegralStr
    {
        int     pDim;
        int     elemID;
        int     sampID;
        int     eNN;
        int     cellID;
        double  w_sample;
        
        Tensor::tensor<double,2> nborFields{};
        Tensor::tensor<double,1> cellFields{};
        Tensor::tensor<double,1> tissFields{};

        std::vector<std::vector<double>> bfs{};

        Tensor::tensor<double,1> tissIntegrals{};
        Tensor::tensor<double,1> cellIntegrals{};

        Tensor::tensor<double,2> vec_n{};
        Tensor::tensor<double,1> vec_c{};

        Tensor::tensor<double,4> mat_nn{};
        Tensor::tensor<double,3> mat_nc{};
        Tensor::tensor<double,3> mat_cn{};
        Tensor::tensor<double,2> mat_cc{};
        
        std::vector<std::string> _nodeFieldNames; ///<List of names for the nodal fields
        std::vector<std::string> _cellFieldNames;      ///<List of names for the global fields
        std::vector<std::string> _tissFieldNames;      ///<List of names for the global fields
        std::map<std::string,int> _mapNodeFieldNames; ///<Map name to field number for nodal fields
        std::map<std::string,int> _mapCellFieldNames; ///<Map name to field number for global fields
        std::map<std::string,int> _mapTissFieldNames; ///<Map name to field number for global fields

        std::vector<std::string> _nodeDOFNames; ///<List of names for the nodal DOFs
        std::vector<std::string> _cellDOFNames;      ///<List of names for the global DOFs
        std::vector<std::string> _cellIntegralNames;      ///<List of names for the global DOFs
        std::vector<std::string> _tissIntegralNames;      ///<List of names for the global DOFs
        std::map<std::string,int> _mapNodeDOFNames; ///<Map name to field number for nodal DOFs
        std::map<std::string,int> _mapCellDOFNames; ///<Map name to field number for global DOFs
        std::map<std::string,int> _mapCellIntegralNames; ///<Map name to field number for nodal DOFs
        std::map<std::string,int> _mapTissIntegralNames; ///<Map name to field number for global DOFs
        
        
        int idxNodeField(std::string field)
        {    return _mapNodeFieldNames[field];    }
        int idxCellField(std::string field)
        {    return _mapCellFieldNames[field];    }
        int idxNodeDOF(std::string field)
        {    return _mapNodeDOFNames[field];    }
        int idxCellDOF(std::string field)
        {    return _mapCellDOFNames[field];    }
        int idxTissField(std::string field)
        {    return _mapTissFieldNames[field];    }
        
        int idxTissIntegral(std::string field)
        {    return _mapTissIntegralNames[field];    }
        int idxCellIntegral(std::string field)
        {    return _mapCellIntegralNames[field];    }
    };

    struct DoubleIntegralStr
    {
        Teuchos::RCP<SingleIntegralStr> fillStr1{Teuchos::null};
        Teuchos::RCP<SingleIntegralStr> fillStr2{Teuchos::null};
        
        //Interactions
        Tensor::tensor<double,4> mat_n1n2{};
        Tensor::tensor<double,4> mat_n2n1{};
        
        Tensor::tensor<double,3> mat_n2c1{};
        Tensor::tensor<double,3> mat_n1c2{};
        Tensor::tensor<double,3> mat_c2n1{};
        Tensor::tensor<double,3> mat_c1n2{};
        
        Tensor::tensor<double,2> mat_c2c1{};
        Tensor::tensor<double,2> mat_c1c2{};
    };
}

#endif  // _BasicStructures_h
