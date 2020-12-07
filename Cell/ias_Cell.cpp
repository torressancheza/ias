#include <iostream>
#include <math.h>
#include <stdexcept>
#include <string>

#include <Teuchos_RCP.hpp>

#include "ias_LinearElements.h"
#include "ias_LoopSubdivision.h"
#include "ias_Cell.h"

namespace ias
{
    void Cell::Update()
    {
        using namespace std;
        using namespace Tensor;
        
        
        if(_connec.size() == 0)
            throw runtime_error("Cell::Update: Mesh connectivity has not been set");
        
        if(_bfType == BasisFunctionType::Linear)
            _bfs = Teuchos::rcp(new LinearElements(_connec));
        else if (_bfType == BasisFunctionType::LoopSubdivision)
            _bfs = Teuchos::rcp(new LoopSubdivision(_connec));
        else
            throw runtime_error("Cell::Update: Basis Function type has not been set");
        
        if(_nodeFields.size() == 0)
            throw runtime_error("Cell::Update: Node positions have not been set");

        if(_nodeFields.shape()[1] < int(_nodeFieldNames.size()))
        {
            tensor<double,2> prevFields = _nodeFields;
            _nodeFields.resize(prevFields.shape()[0],int(_nodeFieldNames.size()));
            
            if(prevFields.shape()[1] > 0)
                _nodeFields(all,range(0,prevFields.shape()[1]-1)) = prevFields;
            
            if(int(_nodeFieldNames.size()) > prevFields.shape()[1])
                _nodeFields(all,range(prevFields.shape()[1],_nodeFieldNames.size()-1)) = 0.0;

        }
        if(_globFields.shape()[0] < int(_globFieldNames.size()))
        {
            tensor<double,1> prevFields = _globFields;
           _globFields.resize(int(_globFieldNames.size()));
    
            if(prevFields.shape()[0] > 0)
                _globFields(range(0,prevFields.shape()[0]-1)) = prevFields;
            
            _globFields(range(prevFields.shape()[0],_globFieldNames.size()-1)) = 0.0;
        }
        
        _mapNodeFieldNames.clear();
        for(size_t i = 0; i < _nodeFieldNames.size(); i++)
            _mapNodeFieldNames[_nodeFieldNames[i]] = i;
        
        _mapGlobFieldNames.clear();
        for(size_t i = 0; i < _globFieldNames.size(); i++)
            _mapGlobFieldNames[_globFieldNames[i]] = i;
    }

    Tensor::tensor<double,2> Cell::getBoundingBox(double eps)
    {
        using namespace std;
        using namespace Tensor;
        
        tensor<double,2> box(3,2);
        for(int m = 0; m < 3; m++)
        {
            box(m,0) = _nodeFields(0,m);
            box(m,1) = _nodeFields(0,m);
        }

        for(int i = 1; i < getNumberOfPoints(); i++)
        {
            for(int m = 0; m < 3; m++)
            {
                box(m,0) = min(box(m,0),_nodeFields(i,m))-eps;
                box(m,1) = max(box(m,1),_nodeFields(i,m))+eps;
            }
        }
        
        return box;
    }

    std::vector<int> Cell::getElementsInBox(Tensor::tensor<double,2> box)
    {
        using namespace std;
        using namespace Tensor;
        
        std::vector<int> elems;
        for(int e = 0; e < getNumberOfElements(); e++)
        {
            int eNN_1 = _bfs->getNumberOfNeighbours(e);

            for(int i=0; i < eNN_1; i++)
            {
                int ii = _bfs->getNeighbours(e)[i];
                tensor<double,1> x = _nodeFields(ii,range(0,2));


                if(x(0)>box(0,0) and x(0)<box(0,1) and
                   x(1)>box(1,0) and x(1)<box(1,1) and
                   x(2)>box(2,0) and x(2)<box(2,1))
                {
                    elems.push_back(e);
                    break;
                }
            }
        }
        
        return elems;
    }
}

//            /** @name Setters (after Update())
//            *  @{ */
//            /*! @brief Set the ith global field*/
//            void setGlobField(int i, double gfield)
//            {
//                if(i<_globFields.size())
//                    _globFields(i) = gfield;
//                else
//                    throw std::runtime_error("Cell::setGlobField: Field " + std::to_string(i) + " is beyond the size of global fields (" + std::to_string(_globFields.size())+"). Did you call Update()?");
//            }
//            /*! @brief Set the ith nodal field at node n*/
//            void setNodeField(int i, int n, double value)
//            {
//                if(i<_nodeFields.shape()[0] )
//                {
//                    if (n < _nodeFields.shape()[1])
//                        _nodeFields(i,n) = value;
//                    else
//                        throw std::runtime_error("Cell::setNodeField: Node " + std::to_string(n) + " does not exist!" + " Did you call Update()?");
//
//                }
//                else
//                    throw std::runtime_error("Cell::setNodeField: Field " + std::to_string(i) + " is beyond the size of node fields (" + std::to_string(_nodeFields.shape()[0])+"). Did you call Update()?");
//            }
//            /*! @brief Set the ith nodal field with a value*/
//            void setNodeField(int i, double value)
//            {
//                if(i<_nodeFields.shape()[0])
//                    _nodeFields(i,Tensor::all) = value;
//                else
//                    throw std::runtime_error("Cell::setNodeField: Field " + std::to_string(i) + " is beyond the size of node fields (" + std::to_string(_nodeFields.shape()[0])+"). Did you call Update()?");
//            }
//            /*! @brief Set the ith nodal field with a tensor*/
//            void setNodeField(int i, Tensor::tensor<double,1> vals)
//            {
//                if(i<_nodeFields.shape()[0])
//                {
//                    if (vals.shape()[0] == _nodeFields.shape()[1])
//                        _nodeFields(i,Tensor::all) = vals;
//                    else
//                        throw std::runtime_error("Cell::setNodeField: The given vector has a number of components " + std::to_string(vals.shape()[0]) + " that does not match the number of nodes " + std::to_string(_nodeFields.shape()[1]) + ". Did you call Update()?");
//
//                }
//                else
//                    throw std::runtime_error("Cell::setNodeField: Field " + std::to_string(i) + " is beyond the size of node fields (" + std::to_string(_nodeFields.shape()[0])+"). Did you call Update()?");
//            }
//            /** @} */
