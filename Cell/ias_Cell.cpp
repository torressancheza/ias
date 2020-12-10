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
        if(_cellFields.shape()[0] < int(_cellFieldNames.size()))
        {
            tensor<double,1> prevFields = _cellFields;
           _cellFields.resize(int(_cellFieldNames.size()));
    
            if(prevFields.shape()[0] > 0)
                _cellFields(range(0,prevFields.shape()[0]-1)) = prevFields;
            
            _cellFields(range(prevFields.shape()[0],_cellFieldNames.size()-1)) = 0.0;
        }
        
        _mapNodeFieldNames.clear();
        for(size_t i = 0; i < _nodeFieldNames.size(); i++)
            _mapNodeFieldNames[_nodeFieldNames[i]] = i;
        
        _mapCellFieldNames.clear();
        for(size_t i = 0; i < _cellFieldNames.size(); i++)
            _mapCellFieldNames[_cellFieldNames[i]] = i;
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
