#ifndef _BasicStructures_h
#define _BasicStructures_h

#include "Tensor.h"

namespace ias
{
//    struct AuxIntegralStr
//    {
//        std::vector<double> dparam;
//        std::vector<std::vector<double>> dparam_cell;
//    };

    struct SingleIntegralStr
    {
        int     pDim;
        int     elemID;
        int     sampID;
        int     eNN;
        int     cellID;
        double  w_sample;
        
        Tensor::tensor<double,2> nborFields{};
        Tensor::tensor<double,1> globFields{};
        Tensor::tensor<double,1> tissFields{};

        std::vector<std::vector<double>> bfs{};

        std::vector<double> globIntegrals{};
        std::vector<double> cellIntegrals{};

        Tensor::tensor<double,2> vec_n{};
        Tensor::tensor<double,1> vec_g{};

        Tensor::tensor<double,4> mat_nn{};
        Tensor::tensor<double,3> mat_ng{};
        Tensor::tensor<double,3> mat_gn{};
        Tensor::tensor<double,2> mat_gg{};
        
        std::vector<std::string> _nodeFieldNames; ///<List of names for the nodal fields
        std::vector<std::string> _globFieldNames;      ///<List of names for the global fields
        std::vector<std::string> _tissFieldNames;      ///<List of names for the global fields
        std::map<std::string,int> _mapNodeFieldNames; ///<Map name to field number for nodal fields
        std::map<std::string,int> _mapGlobFieldNames; ///<Map name to field number for global fields
        std::map<std::string,int> _mapTissFieldNames; ///<Map name to field number for global fields

        std::vector<std::string> _nodeDOFNames; ///<List of names for the nodal DOFs
        std::vector<std::string> _globDOFNames;      ///<List of names for the global DOFs
        std::map<std::string,int> _mapNodeDOFNames; ///<Map name to field number for nodal DOFs
        std::map<std::string,int> _mapGlobDOFNames; ///<Map name to field number for global DOFs

        int idxNodeField(std::string field)
        {    return _mapNodeFieldNames[field];    }
        int idxGlobField(std::string field)
        {    return _mapGlobFieldNames[field];    }
        int idxNodeDOF(std::string field)
        {    return _mapNodeDOFNames[field];    }
        int idxGlobDOF(std::string field)
        {    return _mapGlobDOFNames[field];    }
        int idxTissField(std::string field)
        {    return _mapTissFieldNames[field];    }
    };

    struct DoubleIntegralStr
    {
        Teuchos::RCP<SingleIntegralStr> fillStr1{Teuchos::null};
        Teuchos::RCP<SingleIntegralStr> fillStr2{Teuchos::null};
        
        //Interactions
        Tensor::tensor<double,4> mat_n1n2{};
        Tensor::tensor<double,4> mat_n2n1{};
        
        Tensor::tensor<double,3> mat_n2g1{};
        Tensor::tensor<double,3> mat_n1g2{};
        Tensor::tensor<double,3> mat_g2n1{};
        Tensor::tensor<double,3> mat_g1n2{};
        
        Tensor::tensor<double,2> mat_g2g1{};
        Tensor::tensor<double,2> mat_g1g2{};
    };
}

#endif  // _BasicStructures_h
