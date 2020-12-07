
#ifndef _HPLFE_TENSOR_IMPL_H
#define _HPLFE_TENSOR_IMPL_H


#include <list>
#include <type_traits>
#include <cstring>

    namespace Tensor
    {
        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------
        //
        //                                      CONSTRUCTORS AND DESTRUCTORS...
        //
        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------

        template <class Type, int Order>
        template <class... Dims>
        tensor<Type,Order>::tensor ( int dim1, const  Dims... dims )
        {
            _storage = true;
            resize(dim1,dims...);
        }

        template <class Type, int Order>
        template <class... Dims>
        tensor<Type,Order>::tensor ( Type* pointer, const Dims... dims)
        {
            _storage = false;
            _data = pointer;
            resize(dims...);
        }


        template <class Type, int Order>
        void tensor<Type,Order>::clear()
        {
            if ( _storage )
                delete [] _data;

            _data = nullptr;
            _storage = false;
            std::fill(_dims.begin(),_dims.end(),0);
            std::fill(_jumps.begin(),_jumps.end(),0);
            _size = 0;
        }

        template <class Type, int Order>
        template <class... Dims>
        void tensor<Type,Order>::resize(int dim1, const  Dims... dims)
        {

            static_assert(sizeof...(Dims)==Order-1, "Number of dimensions do not match with tensor order");
            static_assert(areInts<Dims...>::value,"Non-integers passed as dimensions.");

            _dims = {dim1, dims...};
#ifdef DEBUG
            for( int i = 0; i < Order; i++ )
            {
                if ( _dims[i] < 0 )
                    throw std::runtime_error("Tensor::Operator(): Dimension " + std::to_string(i) + " is negative: " + std::to_string(_dims[i]) + ".");
            }
#endif
                                        
            _jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                _jumps[i] = _jumps[i+1] * _dims[i+1];
                      
            //If there is no data, storage should be true
            if(_data == nullptr)
                _storage =true;
            
            if(_storage)
            {
                if(_size != _jumps[0]*_dims[0])
                {
                    _size = _jumps[0]*_dims[0];

                    if(_data != nullptr)
                        delete [] _data;
                    _data = new Type [_size];
                }
            }
            else
                _size = _jumps[0]*_dims[0];
        }
    
        template <class Type, int Order>
        void tensor<Type,Order>::set_pointer(Type* pointer)
        {
            if(_storage and _data != nullptr)
                delete [] _data;
            _data = pointer;
            _storage = false;
        }

    
        //Copy constructor
        template <class Type, int Order>
        template <class S>
        tensor<Type,Order>::tensor(tensor<S,Order>& other)
        {
            if(_dims != other._dims)
            {
                _dims = other._dims;
                _jumps[Order-1] = 1;
                for ( int i = Order-2; i >=0 ; i-- )
                    _jumps[i] = _jumps[i+1] * _dims[i+1];
                
                _size = _jumps[0]*_dims[0];

                if(_data != nullptr)
                    delete [] _data;
                _data = new Type [_size];
                _storage = true;
                
            }
            
            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<Type,Type,S,Order,4,0,nextPhase>(*this,*this,other, 0, 0, 0 );
        }

        //Copy assignment
        template <class Type, int Order>
        template <class S>
        tensor<Type,Order>& tensor<Type,Order>::operator=(tensor<S,Order>& other)
        {
            if(_data == nullptr) //tensor was empty
            {
                _dims = other._dims;
                _jumps[Order-1] = 1;
                for ( int i = Order-2; i >=0 ; i-- )
                    _jumps[i] = _jumps[i+1] * _dims[i+1];
            
                _size = _jumps[0]*_dims[0];

                _data = new Type [_size];
                _storage = true;
            }

            constexpr int nextPhase =  (1 < Order) ? 0 : 1;
            TensorOperation::_doElementWiseOperationTensor<Type,Type,S,Order,4,0,nextPhase>(*this, *this, other, 0, 0, 0 );

            return *this;
        }
        //--------------------------------------------------



        //Moving constructor. This is used to shallow copy the information
        template <class Type, int Order>
        tensor<Type,Order>::tensor(tensor<Type,Order>&& other)
        {
            auto dims = _dims;
            auto jumps = _jumps;
            auto data = _data;
            auto storage = _storage;
            auto size = _size;

            //Copy size and _vec
            _dims    = other._dims;
            _jumps   = other._jumps;
            _data    = other._data;
            _storage = other._storage;
            _size    = other._size;

            //Set the other size and _vec to 0 and nullptr
            other._data    = data;
            other._storage = storage;
            other._size    = size;
            other._jumps   = jumps;
            other._dims    = dims;
        }

        //Moving assignment. This is used to shallow copy the information
        template <class Type, int Order>
        tensor<Type,Order>& tensor<Type,Order>::operator=(tensor<Type,Order>&& other)
        {
            if(_data == nullptr) //tensor was empty
            {
                auto dims = _dims;
                auto jumps = _jumps;
                auto data = _data;
                auto storage = _storage;
                auto size = _size;

                //Copy size and _vec
                _dims    = other._dims;
                _jumps   = other._jumps;
                _data    = other._data;
                _storage = other._storage;
                _size    = other._size;

                //Set the other size and _vec to 0 and nullptr
                other._data    = data;
                other._storage = storage;
                other._size    = size;
                other._jumps   = jumps;
                other._dims    = dims;
            }
            else
            {
                constexpr int nextPhase =  (1 < Order) ? 0 : 1;
                TensorOperation::_doElementWiseOperationTensor<Type,Type,Type,Order,4,0,nextPhase>(*this, *this, other, 0, 0, 0 );
            }
                         
            return *this;
        }


        template<class Type, int Order>
        template<int n>
        typename std::enable_if<n==Order-1, void>::type
        tensor<Type,Order>::_initialize( int vidx,  NestedInitializerListsT<Type,1> init_list )
        {
            auto it = init_list.begin();
            for ( int i = 0; i < _dims[n]; i++)
            {
                int vidxI = vidx + i * _jumps[n];

                _data[vidxI] = *it;
                it++;
            }
        }


        template<class Type, int Order>
        template<int n>
        typename std::enable_if<n<Order-1, void>::type
        tensor<Type,Order>::_initialize( int vidx, NestedInitializerListsT<Type,Order-n> init_list )
        {
            auto it = init_list.begin();
            for ( int i = 0; i < _dims[n]; i++)
            {
                int vidxI = vidx + i * _jumps[n];
                _initialize<n+1>(vidxI,*it);

                it++;
            }
        }

        template<class Type, int Order>
        void tensor<Type,Order>::operator=(NestedInitializerListsT<Type,Order> init_list )
        {
            _initialize<0>(0,init_list);
        }


        template<class Type, int Order>
        template<int n>
        typename std::enable_if<n==Order-1, void>::type
        tensor<Type,Order>::_initializeAndDim( std::vector<Type>& data,  NestedInitializerListsT<Type,1> init_list )
        {
            _dims[n] = init_list.size();

            for ( auto& it : init_list)
                data.push_back(it);
        }


        template<class Type, int Order>
        template<int n>
        typename std::enable_if<n<Order-1, void>::type
        tensor<Type,Order>::_initializeAndDim( std::vector<Type>& data, NestedInitializerListsT<Type,Order-n> init_list )

        {
            _dims[n] = init_list.size();
            for ( auto& it : init_list)
                _initializeAndDim<n+1>(data,it);
        }

        template<class Type, int Order>
        tensor<Type,Order>::tensor(NestedInitializerListsT<Type,Order> init_list)
        {
            std::vector<Type> data{};
            _initializeAndDim<0>(data,init_list);

            _jumps[Order-1] = 1;
            for ( int i = Order-2; i >=0 ; i-- )
                _jumps[i] = _jumps[i+1] * _dims[i+1];

            _size = _jumps[0] *_dims[0];
#ifdef DEBUG

            if ( static_cast<unsigned int>(_size) != data.size())
                throw std::runtime_error("tensor::tensor: the nested initializer list has a shape incompatible with a tensor.");

#endif //DEBUG

            _storage = true;
            _data = new Type [_size];

            std::memcpy(_data,data.data(), _size * sizeof(Type));
        }


        //--------------------------------------------------


        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------
        //
        //                                         TENSOR ACCESORS
        //
        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------

        template <class Type, int Order>
        template <typename... Indices>
        auto tensor<Type,Order>::operator() ( const Indices... indices ) -> decltype(_getSubspace<areInts<Indices...>::value,Indices...>::function(*this,indices...))
        {
            return _getSubspace<areInts<Indices...>::value,Indices...>::function(*this,indices...);
        }


        template <class Type, int Order>
        template <typename... Indices>
        Type& tensor<Type,Order>::_getSubspace<true,Indices...>::function(tensor<Type,Order>& thetensor,const Indices... indices)
        {
            static_assert(sizeof...(Indices)==Order, "Number of dimensions do not match with tensor order");
            static_assert(areInts<Indices...>::value,"Non-integers passed as indices.");
            int vidx{};
            thetensor._getVecIndex(vidx, indices...);
            return thetensor._data[vidx];
        }

        template <class Type, int Order>
        template <typename... Indices>
        auto tensor<Type,Order>::_getSubspace<false,Indices...>::function(tensor<Type,Order>& thetensor,const Indices... indices) -> tensor<Type,_nElementsOfClass<sizeof...(Indices)==1,runningIndex,Indices...>::value>
        {
            constexpr int nInts = _nElementsOfClass<sizeof...(Indices)==1,int,Indices...>::value;
            constexpr int nRunI = _nElementsOfClass<sizeof...(Indices)==1,runningIndex,Indices...>::value;
            static_assert(sizeof...(Indices)==Order, "Number of dimensions do not match with tensor order");
            static_assert(nInts+nRunI==Order, "Unrecognized symbol. Indices should be an integer or runningIndex (e.g. all, range(i,j))");

            tensor<Type,nRunI> output;

            int vidx{};
            int dim{};
            thetensor._getVecIndexAndInitDim(vidx, dim, output._dims, output._jumps, indices...);

            output._size = output._jumps[0]*output._dims[0]; //FIXME: is this a problem?


            output._storage = false;
            output._data = &thetensor._data[vidx];

            return output;
        }


        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------
        //
        //                                         TENSOR PRINTING
        //
        //-------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------

        template <class Type, int Order>
        template<class S, int n>
        struct tensor<Type,Order>::_print<S,n,true>
        {
            static void function( S const& t, std::ostream& out, std::string opening, std::string closing, std::string separator, int idx=0 )
            {

                out << opening;
                for ( int i = 0; i < t._dims[n]-1; i++ )
                {
                    int idxI = idx + i*t._jumps[n];

                    _print<S, n+1, (n+1 < Order-1) >::function(t,out,opening,closing,separator,idxI);
                    out << separator;
                }

                int idxI = idx + (t._dims[n]-1)*t._jumps[n];

                _print<S,n+1,(n+1< Order-1)>::function(t,out,opening,closing,separator,idxI);

                idx += t._jumps[n];

                out << closing;

            }
        };

        template <class Type, int Order>
        template<class S, int n>
        struct tensor<Type,Order>::_print<S,n,false>
        {
            static void function( S const& t, std::ostream& out, std::string opening, std::string closing, std::string separator, int idx=0 )
            {

                out << opening;
                for ( int i = 0; i < t._dims[n]-1; i++ )
                {
                    int idxI = idx + i*t._jumps[n];

                        out << t._data[idxI];
                    out << separator;
                }

                int idxI = idx + (t._dims[n]-1)*t._jumps[n];

                out << t._data[idxI];

                idx += t._jumps[n];

                out << closing;

            }
        };

        template<class Type, int Order>
        template <class Idx>
        void tensor<Type,Order>::_getVecIndex(int &vidx, Idx idx)
        {
            (std::is_same<int, Idx>()) ? vidx += idx *_jumps[Order-1] : 0 ;
        }

        template<class Type, int Order>
        template <class Idx, class... Indices>
        void tensor<Type,Order>::_getVecIndex(int &vidx, Idx idx, Indices... indices)
        {
            constexpr int remaining = sizeof...(indices);
#ifdef DEBUG
            if(idx < 0)
                throw std::runtime_error("Tensor::_getVecIndex(): Index " + std::to_string(Order-remaining-1) + " is negative: " + std::to_string(idx) + ".");
#endif
            (std::is_same<int, Idx>()) ? vidx += idx*_jumps[Order-remaining-1] : 0;
            _getVecIndex(vidx, indices...);
        }



        template<class Type, int Order>
        template <size_t childOrder,class Idx>
        typename std::enable_if<std::is_same<int, Idx>::value, void>::type
        tensor<Type,Order>::_getVecIndexAndInitDim(int& vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx)
        {
            (void) i; //This is just to silence the nasty warning about the variable not being used
            (void) childDims; //This is just to silence the nasty warning about the variable not being used
            (void) childJumps; //This is just to silence the nasty warning about the variable not being used
            vidx += idx *_jumps[Order-1];
        }


        template<class Type, int Order>
        template <size_t childOrder,class Idx>
        typename std::enable_if< std::is_same<runningIndex, Idx>::value, void>::type
        tensor<Type,Order>::_getVecIndexAndInitDim(int& vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx)
        {
            if ( idx.size() > 1 )
                childDims[i] = idx.size();
            else
                childDims[i] = _dims[Order-1];

            childJumps[i] = _jumps[Order-1];
            vidx += idx.first *_jumps[Order-1];

            i++;
        }

        template<class Type, int Order>
        template <size_t childOrder,class Idx, class... Indices>
        typename std::enable_if< std::is_same<int, Idx>::value, void>::type
        tensor<Type,Order>::_getVecIndexAndInitDim(int &vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx, Indices... indices)
        {
            constexpr int remaining = sizeof...(indices);

            vidx += idx *_jumps[Order-remaining-1];

            _getVecIndexAndInitDim(vidx, i, childDims, childJumps, indices...);
        }

        template<class Type, int Order>
        template <size_t childOrder,class Idx, class... Indices>
        typename std::enable_if< std::is_same<runningIndex, Idx>::value, void>::type
        tensor<Type,Order>::_getVecIndexAndInitDim(int &vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx, Indices... indices)
        {
            constexpr int remaining = sizeof...(indices);

            if ( idx.size() > 1 )
                childDims[i] = idx.size();
            else
                childDims[i] = _dims[Order-remaining-1];

            childJumps[i] = _jumps[Order-remaining-1];
            vidx += idx.first *_jumps[Order-remaining-1];

            i++;

            _getVecIndexAndInitDim(vidx, i, childDims, childJumps, indices...);
        }

        template<class Type, int Order>
        tensor<Type,Order> tensor<Type,Order>::transpose(std::array<int,Order> indices, bool copy) const
        {
            tensor<Type,Order> newTens;

#ifdef DEBUG
            std::set<int> sIndices(indices.begin(), indices.end());
            int n{};
            for ( auto i : sIndices )
            {
                if (i != n )
                    throw std::runtime_error("Tensor::transpose: there is something wrong with the transposing indices");
                n++;
            }
#endif

            //First copy the dimensions
            int i = 0;
            for ( auto it = indices.begin(); it != indices.end(); ++it )
            {
                newTens._dims[i] = _dims[*it];
                i++;
            }

            if ( copy )
            {
                //If we have to copy the data, mantain usual order
                newTens._jumps[Order-1] = 1;
                for ( int i = Order-2; i >=0 ; i-- )
                    newTens._jumps[i] = newTens._jumps[i+1] * newTens._dims[i+1];

                newTens._size = newTens._jumps[0]*newTens._dims[0];
                newTens._storage = true;
                newTens._data = new Type [newTens._size];

                //Make the copy
                constexpr int nextPhase =  (1 < Order) ? 0 : 1;
                TensorOperation::_doElementWiseOperationTensor<Type,Type,Type,Order,4,0,nextPhase>(newTens,newTens,(*this).transpose(indices,false), 0, 0, 0 ); //FIXME: this is a mess
            }
            else
            {
                //Otherwise, just copy the jumps
                int i = 0;
                for ( auto it = indices.begin(); it != indices.end(); ++it )
                {
                    newTens._jumps[i] = _jumps[*it];
                    i++;
                }

                newTens._data = _data;
                newTens._size = _size;
                newTens._storage = false;
            }

            return newTens;
        }



        template<class Type, int Order>
        tensor<Type,Order> tensor<Type,Order>::transpose(bool copy) const
        {
            tensor<Type,Order> newTens;


            std::copy(_dims.begin()+1,_dims.end()-1,newTens._dims.begin()+1);
            newTens._dims[0]  = _dims[Order-1];
            newTens._dims[Order-1]  = _dims[0];

            if ( copy )
            {
                newTens._jumps[Order-1] = 1;
                for ( int i = Order-2; i >=0 ; i-- )
                    newTens._jumps[i] = newTens._jumps[i+1] * newTens._dims[i+1];

                newTens._size = newTens._dims[0] * newTens._jumps[0];
                newTens._storage = true;
                newTens._data = new Type [newTens._size];

                constexpr int nextPhase =  (1 < Order) ? 0 : 1;
                TensorOperation::_doElementWiseOperationTensor<Type,Type,Type,Order,4,0,nextPhase>(newTens,newTens,(*this).transpose(false), 0, 0, 0 );
            }
            else
            {
                std::copy(_jumps.begin()+1,_jumps.end()-1,newTens._jumps.begin()+1);
                newTens._jumps[0] = _jumps[Order-1];
                newTens._jumps[Order-1] = _jumps[0];

                newTens._data = _data;
                newTens._storage = false;
            }

            return newTens;
        }



        template<class Type, int Order>
        Type tensor<Type,Order>::det()
        {
            static_assert(Order==2, "Determinant does not make sense for order different than 2.");

            Type det{};

            if ( _dims[0] == 2 and _dims[1] == 2 )
                det = (*this)(0,0) * (*this)(1,1) - (*this)(0,1) * (*this)(1,0);
            else if  ( _dims[0] == 3 and _dims[1] == 3  )
            {
                det  =    (*this)(0,0) * ((*this)(1,1)*(*this)(2,2) - (*this)(1,2) * (*this)(2,1))
                        - (*this)(1,0) * ((*this)(0,1)*(*this)(2,2) - (*this)(0,2) * (*this)(2,1))
                        + (*this)(2,0) * ((*this)(0,1)*(*this)(1,2) - (*this)(0,2) * (*this)(1,1));
            }
            else
            {
                std::cout << _dims[0] << " " << _dims[1] << std::endl;
                throw std::runtime_error("Tensor::det: Only programmed for 2x2 and 3x3 at the moment.");
            }

            return det;
        }

        template<class Type, int Order>
        tensor<Type,Order> tensor<Type,Order>::inv ( )
        {
            static_assert(Order==2, "Inverse cannot be computed for order different than 2.");

            tensor<Type,2> inv;

            if ( _dims[0] == 2 and _dims[1] == 2 )
            {
                inv._dims[0] = 2;
                inv._dims[1] = 2;

                inv._jumps[0] = 2;
                inv._jumps[1] = 1;

                inv._storage = true;
                inv._size = 4;
                inv._data = new Type [4];

                Type idet = 1.0/det();
                inv._data[0] =  (*this)(1,1)*idet;
                inv._data[3] =  (*this)(0,0)*idet;
                inv._data[1] = -(*this)(0,1)*idet;
                inv._data[2] = -(*this)(1,0)*idet;
            }
            else
                throw std::runtime_error("Tensor::det: Only programmed for 2x2 at the moment.");

            return inv;
        }

        static const tensor<double,3> antisym3D={{{ 0, 0, 0},
                                                  { 0, 0, 1},
                                                  { 0,-1, 0}},
                                                 {{ 0, 0,-1},
                                                  { 0, 0, 0},
                                                  { 1, 0, 0}},
                                                 {{ 0, 1, 0},
                                                  {-1, 0, 0},
                                                  { 0, 0, 0}}};

        class Identity : public tensor<double,2>
        {
            public:
                Identity(int dim)
                {
                    _dims[0]  = dim;
                    _dims[1]  = dim;
                    _jumps[0] = dim;
                    _jumps[1] = 1;
                    _size     = dim*dim;
                    _storage = true;
                    _data = new double [dim*dim];

                    for(int i = 0; i < dim*dim; i++)
                        _data[i] = 0.0;

                    for(int i = 0; i < dim; i++)
                        _data[i*dim+i] = 1.0;
                }
        };
    }


#endif //_HPLFE_TENSOR_IMPL_H
