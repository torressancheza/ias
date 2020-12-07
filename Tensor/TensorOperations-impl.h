#ifndef HPLFE_TENSOROPERATIONSIMPL_H
#define HPLFE_TENSOROPERATIONSIMPL_H

#include <list>
#include "TensorRunningIndex.h"

    namespace Tensor
    {
        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProduct<T,S,U,Order1,Order2,contOrder,n,0>
        {
            //First phase, free indices in tensor1
            _doProduct(tensor<U,Order1+Order2-2*contOrder>& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int idx1, int idx2, int idxr )
            {
                for ( int i = 0; i < result._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+1 < Order1-contOrder) ? 0 : ((n+1 < Order1+Order2-2*contOrder) ? 1 : ((n+2 < Order1+Order2-contOrder) ? 2 : (contOrder>0) ? 3: 4));

                    int idxI1 = idx1 + i*tensor1._jumps[n];
                    int idxIr = idxr + i*result._jumps[n];
                    _doProduct<T,S,U,Order1,Order2,contOrder,n+1,nextPhase>(result,tensor1,tensor2, idxI1, idx2, idxIr);
                }
            }
        };

        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProduct<T,S,U,Order1,Order2,contOrder,n,1>
        {
            //Second phase, free indices in tensor2
            _doProduct(tensor<U,Order1+Order2-2*contOrder>& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int idx1, int idx2, int idxr )
            {
                for ( int i = 0; i < result._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+1 < Order1+Order2-2*contOrder) ? 1 : ((n+2 < Order1+Order2-contOrder) ? 2 : (contOrder>0) ? 3: 4);

                    int idxIr = idxr + i*result._jumps[n];
                    int idxI2 = idx2 + i*tensor2._jumps[2*contOrder-Order1+n];

                    _doProduct<T,S,U,Order1,Order2,contOrder,n+1,nextPhase>(result,tensor1,tensor2, idx1, idxI2, idxIr);
                }
            }
        };

        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProduct<T,S,U,Order1,Order2,contOrder,n,2>
        {
            //Third phase, sum paired indices
            _doProduct(tensor<U,Order1+Order2-2*contOrder>& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int idx1, int idx2, int idxr )
            {
                for ( int i = 0; i < tensor1._dims[n-Order2+contOrder]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order1+Order2-contOrder) ? 2 : (contOrder>0) ? 3: 4;

                    int idxI1 = idx1 + i*tensor1._jumps[n-Order2+contOrder];
                    int idxI2 = idx2 + i*tensor2._jumps[n-Order1-Order2+2*contOrder];

                    _doProduct<T,S,U,Order1,Order2,contOrder,n+1,nextPhase>(result,tensor1,tensor2, idxI1, idxI2, idxr);
                }
            }
        };


        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProduct<T,S,U,Order1,Order2,contOrder,n,3>
        {
            //Third phase, sum paired indices
            _doProduct(tensor<U,Order1+Order2-2*contOrder>& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2>const&  tensor2, int idx1, int idx2, int idxr )
            {
                for ( int i = 0; i < tensor1._dims[Order1-1]; i++ )
                {
                    int idxI1 = idx1 + i * tensor1._jumps[Order1-1];
                    int idxI2 = idx2 + i * tensor2._jumps[contOrder-1];

                    result._data[idxr] += tensor1._data[idxI1] * tensor2._data[idxI2];
                }
            }
        };

        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProduct<T,S,U,Order1,Order2,contOrder,n,4>
        {
            //Third phase, sum paired indices
            _doProduct(tensor<U,Order1+Order2-2*contOrder>& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2>const&  tensor2, int idx1, int idx2, int idxr )
            {
                    result._data[idxr] = tensor1._data[idx1] * tensor2._data[idx2];
            }
        };

        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProductScalar<T,S,U,Order1,Order2,contOrder,n,0>
        {
            //Third phase, sum paired indices
            _doProductScalar(U& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int idx1, int idx2 )
            {
                for ( int i = 0; i < tensor1._dims[n-Order2+contOrder]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order1+Order2-contOrder) ? 0 : 1;

                    int idxI1 = idx1 + i*tensor1._jumps[n-Order2+contOrder];
                    int idxI2 = idx2 + i*tensor2._jumps[n-Order1-Order2+2*contOrder];
                    _doProductScalar<T,S,U,Order1,Order2,contOrder,n+1,nextPhase>(result,tensor1,tensor2, idxI1, idxI2 );
                }
            }
        };


        template < class T, class S, class U, int Order1, int Order2, int contOrder, int n >
        struct TensorOperation::_doProductScalar<T,S,U,Order1,Order2,contOrder,n,1>
        {
            //Third phase, sum indices
            _doProductScalar(U& result, tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int idx1, int idx2 )
            {
                for ( int i = 0; i < tensor1._dims[Order1-1]; i++ )
                {
                    int idxI1 = idx1 + i * tensor1._jumps[Order1-1];
                    int idxI2 = idx2 + i * tensor2._jumps[contOrder-1];

                    result += tensor1._data[idxI1] * tensor2._data[idxI2];
                }
            }
        };

        template<class T, class S, int Order1, int Order2, int contOrder>
        struct TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,false>
        {
            static auto function(tensor<T,Order1> const&  tensor1, tensor<T,Order2> const&  tensor2 ) -> tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order1+Order2-2*contOrder>
            {
                //TODO: check that dimensions coincide!
                //TODO: In general: put all required checkers

                tensor<T,Order1+Order2-2*contOrder> result;

                std::copy_n(tensor1._dims.begin(), Order1-contOrder, result._dims.begin());
                std::copy_n(tensor2._dims.begin()+contOrder, Order2-contOrder, result._dims.begin()+Order1-contOrder);

                result._jumps[Order1+Order2-2*contOrder-1] = 1;
                for ( int i = Order1+Order2-2*contOrder-2; i >=0 ; i-- )
                    result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

                result._size = result._jumps[0]*result._dims[0];
                result._storage = true;
                result._data = new T [result._size]();

                //Initial phase?
                constexpr int iniPhase = (0 < Order1-contOrder) ? 0 : ((0 < Order1+Order2-2*contOrder) ? 1 : ((0 < Order1+Order2-contOrder) ? 2 : (contOrder>0) ? 3 : 4));

                _doProduct<T,S,decltype(tensor1._data[0]*tensor2._data[0]),Order1,Order2,contOrder,0,iniPhase>(result,tensor1,tensor2,0,0,0);

                return result;
            }
        };

        template<class T, class S, int Order1, int Order2, int contOrder>
        struct TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,true>
        {
            static auto function(tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2 ) -> decltype(tensor1._data[0]*tensor2._data[0])
            {
                T result{};
                //Initial phase?
                constexpr int iniPhase = (1 < Order1+Order2-contOrder) ? 0 : 1;
                _doProductScalar<T,S,decltype(tensor1._data[0]*tensor2._data[0]),Order1,Order2,contOrder,0,iniPhase>(result,tensor1,tensor2,0,0);

                return result;
            }
        };


        template< class T, class S, int Order1, int Order2, int contOrder >
        auto TensorOperation::_product(tensor<T,Order1> const&  tensor1, tensor<S,Order2> const&  tensor2, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,Order1+Order2-2*contOrder==0>::function(tensor1, tensor2))
        {

            std::array<int,Order1> allIndices1;
            std::array<int,Order2> allIndices2;

            std::list<int> sortedIndices1;
            std::list<int> sortedIndices2;

            for ( int i = 0; i < Order1; i++ )
                sortedIndices1.push_back(i);
            for ( int i = 0; i < Order2; i++ )
                sortedIndices2.push_back(i);

            for ( int i = 0; i < contOrder; i++ )
            {
                allIndices1[Order1-contOrder+i] = contIndices[i][0];
                allIndices2[i] = contIndices[i][1];

                sortedIndices1.remove(contIndices[i][0]);
                sortedIndices2.remove(contIndices[i][1]);
            }

            std::copy(sortedIndices1.begin(), sortedIndices1.end(),allIndices1.begin());
            std::copy(sortedIndices2.begin(), sortedIndices2.end(),allIndices2.begin()+contOrder);

            //-------------------------------------------------
            //TODO: If indices are in order this is not required!
            auto tensor1T = tensor1.transpose(allIndices1,false);
            auto tensor2T = tensor2.transpose(allIndices2,false);
            //-------------------------------------------------

            constexpr bool isScalar = Order1+Order2-2*contOrder==0;
            return TensorOperation::_productLastFirst<T,S,Order1,Order2,contOrder,isScalar>::function( tensor1T, tensor2T );
        }






        template < class T, class U, int Order, int contOrder, int n >
        struct TensorOperation::_doContract<T,U,Order,contOrder,n,0>
        {
            //First phase, free indices in tensor1
            _doContract(tensor<U,Order-2*contOrder>& result, tensor<T,Order> const&  input, int idx1, int idxr )
            {
                for ( int i = 0; i < result._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+1 < Order-2*contOrder) ? 0 : ((n+2 < Order-contOrder) ? 1 : 2 );

                    int idxIr = idxr + i*result._jumps[n];
                    int idxI1 = idx1 + i*input._jumps[n+contOrder];
                    _doContract<T,U,Order,contOrder,n+1,nextPhase>(result,input, idxI1, idxIr);
                }
            }
        };


        template < class T, class U, int Order, int contOrder, int n >
        struct TensorOperation::_doContract<T,U,Order,contOrder,n,1>
        {
            //First phase, free indices in tensor1
            _doContract(tensor<U,Order-2*contOrder>& result, tensor<T,Order> const&  input, int idx1, int idxr )
            {
                for ( int i = 0; i < input._dims[n+contOrder]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order-contOrder) ? 1 : 2;

                    int idxI1 = idx1 + i*(input._jumps[n-Order+2*contOrder]+input._jumps[n+contOrder]);
                    _doContract<T,U,Order,contOrder,n+1,nextPhase>(result, input, idxI1, idxr);
                }
            }
        };

        template < class T, class U, int Order, int contOrder, int n >
        struct TensorOperation::_doContract<T,U,Order,contOrder,n,2>
        {
            //First phase, free indices in tensor1
            _doContract(tensor<U,Order-2*contOrder>& result, tensor<T,Order> const&  input, int idx1, int idxr )
            {
                for ( int i = 0; i < input._dims[n+contOrder]; i++ )
                {
                    int idxI1 = idx1 + i*(input._jumps[n-Order+2*contOrder]+input._jumps[n+contOrder]);
                    result._data[idxr] += input._data[idxI1] ;
                }
            }
        };

        template < class T, class U, int Order, int contOrder, int n >
        struct TensorOperation::_doContractScalar<T,U,Order,contOrder,n,0>
        {
            //First phase, free indices in tensor1
            _doContractScalar( U& result, tensor<T,Order> const&  input, int idx1 )
            {
                for ( int i = 0; i < input._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order-contOrder) ? 0 : 1;

                    int idxI1 = idx1 + i*(input._jumps[n-Order+2*contOrder]+input._jumps[n+contOrder]);
                    _doContractScalar<T,U,Order,contOrder,n+1,nextPhase>(result,input, idxI1);
                }
            }
        };

        template < class T, class U, int Order, int contOrder, int n >
        struct TensorOperation::_doContractScalar<T,U,Order,contOrder,n,1>
        {
            //First phase, free indices in tensor1
            _doContractScalar(U& result, tensor<T,Order> const&  input, int idx1 )
            {
                for ( int i = 0; i < input._dims[n]; i++ )
                {
                    int idxI1 = idx1 + i*(input._jumps[n-Order+2*contOrder]+input._jumps[n+contOrder]);
                    result += input._data[idxI1] ;
                }
            }
        };



        template<class T, int Order, int contOrder>
        struct TensorOperation::_contractLastFirst<T,Order,contOrder,false>
        {
            static auto function(tensor<T,Order> const&  input ) -> tensor<T,Order-2*contOrder>
            {
                //TODO: check that dimensions coincide!
                //TODO: In general: put all required checkers

                tensor<T,Order-2*contOrder> result;

                std::copy_n(input._dims.begin()+contOrder, Order-2*contOrder, result._dims.begin());

                result._jumps[Order-2*contOrder-1] = 1;
                for ( int i = Order-2*contOrder-2; i >=0 ; i-- )
                    result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

                result._size = result._jumps[0]*result._dims[0];
                result._storage = true;
                result._data = new T [result._size]();

                //Initial phase?
                _doContract<T,T,Order,contOrder,0,0>(result,input,0,0);

                return result;
            }
        };

        template<class T, int Order, int contOrder>
        struct TensorOperation::_contractLastFirst<T,Order,contOrder,true>
        {
            static T function(tensor<T,Order> const&  input )
            {
                T result{};
                //Initial phase?
                constexpr int iniPhase = (0 < Order-contOrder-1) ? 0 : 1;
                _doContractScalar<T,T,Order,contOrder,0,iniPhase>(result,input,0);

                return result;
            }
        };







        template< class T, int Order, int contOrder >
        auto TensorOperation::_contract(tensor<T,Order> const&  input, int const (&contIndices)[contOrder][2]) ->  decltype(TensorOperation::_contractLastFirst<T,Order,contOrder,Order-2*contOrder==0>::function(input))
        {

            std::array<int,Order> allIndices;
            std::list<int> sortedIndices;

            for ( int i = 0; i < Order; i++ )
                sortedIndices.push_back(i);

            for ( int i = 0; i < contOrder; i++ )
            {
                allIndices[i] = contIndices[i][0];
                allIndices[Order-contOrder+i] = contIndices[i][1];

                sortedIndices.remove(contIndices[i][0]);
                sortedIndices.remove(contIndices[i][1]);
            }

            std::copy(sortedIndices.begin(), sortedIndices.end(),allIndices.begin()+contOrder);


            //-------------------------------------------------
            //TODO: If indices are in order this is not required!
            auto inputT = input.transpose(allIndices,false);

            constexpr bool isScalar = Order-2*contOrder==0;
            return TensorOperation::_contractLastFirst<T,Order,contOrder,isScalar>::function(inputT);
        }


        //This is just a way to make operations template-dependent to program tensor operations once in terms of OpType
        //OpType = 0 : sum
        //OpType = 1 : subtraction
        //OpType = 2 : product
        //OpType = 3 : quotient
        //OpType = 4 : equal to second
        template <class T, class S, int OpType>
        struct operation;

        template <class T, class S>
        struct operation<T,S,0>
        {
            static constexpr auto function( T first, S second ) -> decltype(first+second)
            {
                return first+second;
            }
        };

        template <class T, class S>
        struct operation<T,S,1>
        {
            static constexpr auto  function( T first, S second ) -> decltype(first-second)
            {
                return first-second;
            }
        };


        template <class T, class S>
        struct operation<T,S,2>
        {
            static constexpr auto  function( T first, S second ) -> decltype(first*second)
            {
                return first*second;
            }
        };

        template <class T, class S>
        struct operation<T,S,3>
        {
            static constexpr auto  function( T first, S second ) -> decltype(first/second)
            {
                return first/second;
            }
        };

        template <class T, class S>
        struct operation<T,S,4>
        {
            static constexpr auto  function( T first, S second ) -> decltype(second)
            {
                return (void) first, second;
            }
        };


        template < class T, class S, class U, int Order, int OpType, int n >
        struct TensorOperation::_doElementWiseOperationScalar<T,S,U,Order,OpType,n,0>
        {
            //First phase, free indices in tensor1
            _doElementWiseOperationScalar(tensor<T,Order>&  output, tensor<S,Order> const&  input, U const& scalar, int idxO, int idxI )
            {
                for ( int i = 0; i < input._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order) ? 0 : 1;

                    int idxOI = idxO + i*output._jumps[n];
                    int idxII = idxI + i*input._jumps[n];

                    _doElementWiseOperationScalar<T,S,U,Order,OpType,n+1,nextPhase>(output,input, scalar, idxOI, idxII);
                }
            }
        };

        template < class T, class S, class U, int Order, int OpType, int n >
        struct TensorOperation::_doElementWiseOperationScalar<T,S,U,Order,OpType,n,1> //FIXME: can I remove this n?
        {
            //First phase, free indices in tensor1
            _doElementWiseOperationScalar(tensor<T,Order>&  output, tensor<S,Order> const&  input, U const& scalar, int idxO, int idxI )
            {
                for ( int i = 0; i < input._dims[Order-1]; i++ )
                {
                    int idxOI = idxO + i*output._jumps[Order-1];
                    int idxII = idxI + i*input._jumps[Order-1];

                    output._data[idxOI] = operation<S,U,OpType>::function(input._data[idxII],scalar);
                }
            }
        };



        template < class T, class S, class U, int Order, int OpType, int n >
        struct TensorOperation::_doElementWiseOperationTensor<T,S,U,Order,OpType,n,0>
        {
            //First phase, free indices in tensor1
            _doElementWiseOperationTensor(tensor<T,Order>&  output, tensor<S,Order> const&  input1, tensor<U,Order> const&  input2, int idxO, int idx1, int idx2 )
            {
                for ( int i = 0; i < output._dims[n]; i++ )
                {
                    constexpr int nextPhase =  (n+2 < Order) ? 0 : 1;

                    int idxOI = idxO + i*output._jumps[n];
                    int idx1I = idx1 + i*input1._jumps[n];
                    int idx2I = idx2 + i*input2._jumps[n];

                    _doElementWiseOperationTensor<T,S,U,Order,OpType,n+1,nextPhase>(output, input1, input2, idxOI, idx1I, idx2I);
                }
            }
        };

        template < class T, class S, class U, int Order, int OpType, int n >
        struct TensorOperation::_doElementWiseOperationTensor<T,S,U,Order,OpType,n,1>
        {
            //First phase, free indices in tensor1
            _doElementWiseOperationTensor(tensor<T,Order>&  output, tensor<S,Order> const&  input1, tensor<U,Order> const&  input2, int idxO, int idx1, int idx2 )
            {
                for ( int i = 0; i < output._dims[Order-1]; i++ )
                {
                    int idxOI = idxO + i*output._jumps[Order-1];
                    int idx1I = idx1 + i*input1._jumps[Order-1];
                    int idx2I = idx2 + i*input2._jumps[Order-1];

                    output._data[idxOI] = operation<S,U,OpType>::function(input1._data[idx1I],input2._data[idx2I]);
                }
            }
        };



        template< class T, class S, int Order>
        auto operator+(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]+tensor2._data[0]),Order>
        {

            //Check if orders and sizes match
#ifdef   DEBUG

#endif //DEBUG

            tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order> result;

            result._dims = tensor1._dims;

            result._size = tensor1._size;
            result._storage = true;
            result._data = new decltype(tensor1._data[0]+tensor2._data[1]) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<decltype(tensor1._data[0]+tensor2._data[1]),T,S,Order,0,0,nextPhase>(result,tensor1,tensor2, 0, 0, 0 );

            return result;
        }


        template< class T, class S, int Order>
        auto operator-(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]-tensor2._data[0]),Order>
        {

            //Check if orders and sizes match
#ifdef   DEBUG

#endif //DEBUG

            tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order> result;

            result._dims = tensor1._dims;

            result._size = tensor1._size;
            result._storage = true;
            result._data = new decltype(tensor1._data[0]-tensor2._data[1]) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<decltype(tensor1._data[0]-tensor2._data[1]),T,S,Order,1,0,nextPhase>(result,tensor1,tensor2, 0, 0, 0 );

            return result;
        }

        template <class T, class S, int Order>
        auto eproduct(tensor<T,Order> const& tensor1, tensor<S,Order> const& tensor2 ) -> tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order>
        {
            //Check if orders and sizes match
#ifdef   DEBUG

#endif //DEBUG

            tensor<decltype(tensor1._data[0]*tensor2._data[0]),Order> result;

            result._dims = tensor1._dims;

            result._size = tensor1._size;
            result._storage = true;
            result._data = new decltype(tensor1._data[0]*tensor2._data[1]) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<decltype(tensor1._data[0]*tensor2._data[1]),T,S,Order,2,0,nextPhase>(result,tensor1,tensor2, 0, 0, 0 );

            return result;
        }

        //Operations with scalars

        //Sum of a tensor and a scalar
        template <class T, class S, int Order>
        auto operator+(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]+scalar),Order>
        {
            tensor<decltype(input._data[0]+scalar),Order> result;

            result._dims = input._dims;

            result._size    = input._size;
            result._storage = true;
            result._data    = new decltype(input._data[0]+scalar) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;

            TensorOperation::_doElementWiseOperationScalar<decltype(input._data[0]+scalar),T,S,Order,0,0,nextPhase>(result, input, scalar, 0, 0 );

            return result;
        }

        //Substract a scalar from a tensor
        template <class T, class S, int Order>
        auto operator-(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]-scalar),Order>
        {
            tensor<decltype(input._data[0]-scalar),Order> result;

            result._dims = input._dims;

            result._size    = input._size;
            result._storage = true;
            result._data    = new decltype(input._data[0]-scalar) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;

            TensorOperation::_doElementWiseOperationScalar<decltype(input._data[0]-scalar),T,S,Order,1,0,nextPhase>(result, input, scalar, 0, 0 );

            return result;
        }

        //Sum of a tensor and a scalar
        template <class T, class S, int Order>
        auto operator*(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]*scalar),Order>
        {
            tensor<decltype(input._data[0]*scalar),Order> result;

            result._dims = input._dims;

            result._size    = input._size;
            result._storage = true;
            result._data    = new decltype(input._data[0]*scalar) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;

            TensorOperation::_doElementWiseOperationScalar<decltype(input._data[0]*scalar),T,S,Order,2,0,nextPhase>(result, input, scalar, 0, 0 );

            return result;
        }

        //Sum of a tensor and a scalar
        template <class T, class S, int Order>
        auto operator/(tensor<T,Order> const& input, S const& scalar ) -> tensor<decltype(input._data[0]/scalar),Order>
        {
            tensor<decltype(input._data[0]/scalar),Order> result;

            result._dims = input._dims;

            result._size    = input._size;
            result._storage = true;
            result._data    = new decltype(input._data[0]/scalar) [result._size];

            result._jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                result._jumps[i] = result._jumps[i+1] * result._dims[i+1];

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;

            TensorOperation::_doElementWiseOperationScalar<decltype(input._data[0]/scalar),T,S,Order,3,0,nextPhase>(result, input, scalar, 0, 0 );

            return result;
        }

        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator=(U const& scalar)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationScalar<Type,Type,U,Order,4,0,nextPhase>(*this, *this,scalar, 0, 0 );
        }

        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator+=(U const& scalar)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationScalar<Type,Type,U,Order,0,0,nextPhase>(*this, *this,scalar, 0, 0 );
        }
        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator-=(U const& scalar)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationScalar<Type,Type,U,Order,1,0,nextPhase>(*this, *this,scalar, 0, 0 );
        }
        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator*=(U const& scalar)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationScalar<Type,Type,U,Order,2,0,nextPhase>(*this, *this,scalar, 0, 0 );
        }
        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator/=(U const& scalar)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationScalar<Type,Type,U,Order,3,0,nextPhase>(*this, *this,scalar, 0, 0 );
        }

        template<class Type, int Order>
        template<class U>
        void tensor<Type,Order>::operator+=(tensor<U,Order> const& tensor2)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<Type,Type,U,Order,0,0,nextPhase>(*this,*this,tensor2, 0, 0, 0 );
        }

        template<class Type, int Order>
        template <class U>
        void tensor<Type,Order>::operator-=(tensor<U,Order> const& tensor2)
        {
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<Type,Type,U,Order,1,0,nextPhase>(*this,*this,tensor2, 0, 0, 0 );
        }
    }

#endif //HPLFE_TENSOROPERATIONSIMPL_H
