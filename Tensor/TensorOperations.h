/*
 **************************************************************************************************
 hiperlife - High Performance Library for Finite Elements
 Project homepage: https://git.lacan.upc.edu/HPLFEgroup/hiperlifelib.git
 Copyright (c) 2018 Daniel Santos-Olivan, Alejandro Torres-Sanchez and Guillermo Vilanova
 **************************************************************************************************
 hiperlife is under GNU General Public License ("GPL").
 GNU General Public License ("GPL") copyright permissions statement:
 This file is part of hiperlife, hiperlife is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by the Free Software Foundation,
 either version 3 of the License, or (at your option) any later version.
 hiperlife is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************************************
 */


#ifndef HPLFE_TENSOROPERATIONS_H
#define HPLFE_TENSOROPERATIONS_H

    namespace Tensor
    {
        
        class TensorOperation
        {
            private:
            
                template < class T, class S, class U, int Order1, int Order2, int contOrder, int n, int phase >
                struct _doProduct;
            
                template < class T, class S, class U, int Order1, int Order2, int contOrder, int n, int phase >
                struct _doProductScalar;
            
                template<class T, class S, int Order1, int Order2, int contOrder, bool isScalar>
                struct _productLastFirst;
            
                template< class T, class S, int Order1, int Order2, int contOrder >
                static auto _product(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2, int const (&contIndices)[contOrder][2]) ->  decltype(_productLastFirst<T,S,Order1,Order2,contOrder,Order1+Order2-2*contOrder==0>::function(tensor1, tensor2));
            
                template < class T, class U, int Order, int contOrder, int n, int phase >
                struct _doContract;
            
                template < class T, class U, int Order, int contOrder, int n, int phase >
                struct _doContractScalar;
            
                template<class T, int Order, int contOrder, bool isScalar>
                struct _contractLastFirst;
            
                template< class T, int Order, int contOrder >
                static auto _contract(tensor<T,Order> const&  input, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_contractLastFirst<T,Order,contOrder,Order-2*contOrder==0>::function(input));
            
                template < class T, class S, class U, int Order, int OpType, int n, int Phase>
                struct _doElementWiseOperationScalar;
            
            
                template < class T, class S, class U, int Order, int OpType, int n, int Phase>
                struct _doElementWiseOperationTensor;
            
                //FRIENDS (same functions calling the internal functions for usage from outsise)
                template<class T, int Order>
                friend class tensor;
            
                template<class S, int O, class... Indices>
                friend class tensorindices;
            
                template< class T, class S, int Order1, int Order2, int contOrder >
                friend auto product(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,Order1+Order2-2*contOrder==0>::function(tensor1,tensor2));

                template <class T, class S, int Order>
                friend auto operator+(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]+tensor2._data[0]),Order>;

                template <class T, class S, int Order>
                friend auto operator-(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]-tensor2._data[0]),Order>;

            template< class T, int Order, int contOrder >
                friend auto contract(tensor<T,Order> const& input, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_contractLastFirst<T,Order,contOrder,Order-2*contOrder==0>::function(input));
            
                template <class T, class S, int Order1, int Order2>
                friend auto operator*(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2 ) -> decltype(_productLastFirst<T,S,Order1,Order2,1,Order1+Order2-2==0>::function(tensor1,tensor2));
            
                template <class T, class S, int Order>
                friend auto eproduct(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order>;
            
                template <class T, int Order>
                friend auto trace(tensor<T,Order> const& input ) -> decltype(TensorOperation::_contractLastFirst<T,Order,1,Order-2==0>::function(input));
            
                template <class T, class S, int Order>
                friend auto operator+(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]+scalar),Order>;
                template <class T, class S, int Order>
                friend auto operator-(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]-scalar),Order>;
                template <class T, class S, int Order>
                friend auto operator*(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]*scalar),Order>;
                template <class T, class S, int Order>
                friend auto operator/(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]/scalar),Order>;

                template< class T, class S, int Order1, int Order2 >
                friend auto outer(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2) ->  decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,0,false>::function(tensor1,tensor2));

        };

        template< class T, class S, int Order1, int Order2 >
        auto outer(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2) ->  decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,0,false>::function(tensor1,tensor2))
        {
            return TensorOperation::_productLastFirst<T,S,Order1,Order2,0,false>::function(tensor1,tensor2);
        }

        template< class T, class S, int Order1, int Order2, int contOrder >
        auto product(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,Order1+Order2-2*contOrder==0>::function(tensor1,tensor2))
        {
            return TensorOperation::_product(tensor1,tensor2,contIndices);
        }

        template< class T, int Order, int contOrder >
        auto contract(tensor<T,Order> const& input, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_contractLastFirst<T,Order,contOrder,Order-2*contOrder==0>::function(input))
        {
            return TensorOperation::_contract(input,contIndices);
        }


        template< class T, int Order>
        tensor<T,Order> transpose(tensor<T,Order> const& tensor, std::array<int,Order> indices, bool copy = true )
        {
            return tensor.transpose(indices,copy);
        }

        template <class T, class S, int Order>
        auto eproduct(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order>;


        //Sum of two tensors
        template <class T, class S, int Order>
        auto operator+(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]+tensor2._data[0]),Order>;

        //Subtraction of two tensors
        template <class T, class S, int Order>
        auto operator-(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]-tensor2._data[0]),Order>;


        //Sum of a tensor and a scalar
        template <class T, class S, int Order>
        auto operator+(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]+scalar),Order>;

        template <class T, class S, int Order>
        auto operator+(S const& scalar, tensor<T,Order> const& input) -> tensor<decltype(input._data[0]+scalar),Order>
        {
            return input + scalar;
        }


        //Substract scalar from a tensor
        template <class T, class S, int Order>
        auto operator-(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]-scalar),Order>;

        template <class T, class S, int Order>
        auto operator-(S const& scalar, tensor<T,Order> const& input) -> tensor<decltype(input._data[0]-scalar),Order>
        {
            return input - scalar;
        }

        template <class S, class U, int Order>
        auto operator*(tensor<S,Order> const& input, U const& scalar) -> tensor<decltype(input._data[0]*scalar),Order>;

        template <class S, class U, int Order>
        auto operator*(U const& scalar,tensor<S,Order> const& input) -> tensor<decltype(input._data[0]*scalar),Order>
        {
            return input * scalar;
        }


        template <class S, class U, int Order>
        auto operator/(tensor<S,Order> const& input, U const& scalar) -> tensor<decltype(input._data[0]/scalar),Order>;


        template <class T, class S, int Order1, int Order2>
        auto operator*(tensor<T,Order1> const& tensor1, tensor<S,Order2> const& tensor2 ) -> decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,1, Order1+Order2-2==0>::function(tensor1,tensor2))
        {
            return TensorOperation::_productLastFirst<T,S,Order1,Order2,1, Order1+Order2-2==0>::function(tensor1,tensor2);
        }


        template <class T, int Order>
        auto trace(tensor<T,Order> const& input ) -> decltype(TensorOperation::_contractLastFirst<T,Order,1,Order-2==0>::function(input))
        {
            return TensorOperation::_contractLastFirst<T,Order,1,Order-2==0>::function(input);
        }
    
    }

#include "TensorOperations-impl.h"

#endif //HPLFE_TENSOROPERATIONS_H

