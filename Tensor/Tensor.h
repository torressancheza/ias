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


#ifndef HPLFE_TENSOR_H
#define HPLFE_TENSOR_H

#include <iostream>
#include <set>
#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <algorithm>
#include <cassert>
#include <utility>
#include <type_traits>
#include <stddef.h>

#include "TensorRunningIndex.h"


    namespace Tensor
    {

        /*!
        * \class tensor
        * \brief Class for tensor operations
        * <b>hiperlife</b>.
        *
        * This class admits two template parameters (Type and Order) which determine the type of
        * data of the tensor and the order of the tensor respectively.
        * The tensor class has two main constructors that can be accessed from the user and allow
        * the user to create a new tensor with internal storage or pass a pointer to where
        * data is stored
        */
        template <class Type, int Order>
        class tensor
        {
            
            static_assert(Order>0, "Order must be larger than 0.");

            /** @name Some helpers
            *  @{ */
            protected:
                template<bool final, typename Ref, typename S, typename... Ts>
                struct _nElementsOfClass;

                template<typename Ref, typename S>
                struct _nElementsOfClass<true,Ref,S>
                {
                    static constexpr int value = std::is_same<Ref,S>::value ? 1 : 0;
                };

                template<typename Ref, typename S, typename... Ts>
                struct _nElementsOfClass<false,Ref,S,Ts...>
                {
                    static constexpr int value = std::is_same<Ref,S>::value ? (_nElementsOfClass<sizeof...(Ts)==1,Ref,Ts...>::value + 1) : (_nElementsOfClass<sizeof...(Ts)==1,Ref,Ts...>::value);
                };

                template<typename... Conds>
                struct _and : std::true_type {};

                template<typename Cond, typename... Conds>
                struct _and<Cond, Conds...> : std::conditional<Cond::value, _and<Conds...>,std::false_type>::type {};

                template<typename... Ts>
                using areInts = _and<std::is_same<int,Ts>...>;


                template<class T, std::size_t T_levels>
                struct NestedInitializerLists
                {
                    using type = std::initializer_list<typename NestedInitializerLists<T, T_levels - 1>::type>;
                };

                template<class T>
                struct NestedInitializerLists<T, 0>
                {
                    using type = T;
                };


                template<class T, std::size_t T_levels>
                using NestedInitializerListsT = typename NestedInitializerLists<T, T_levels>::type;
            /** @} */ // end of some helpers

            /** @name Constructor / Destructor / Assignment Operator
            *  @{ */
            protected:

                template<int n>
                typename std::enable_if<n<Order-1, void>::type
                _initialize( int vidx,  NestedInitializerListsT<Type,Order-n> init_list );

                template<int n>
                typename std::enable_if<n==Order-1, void>::type
                _initialize( int vidx, NestedInitializerListsT<Type,1> init_list );
            
                template<int n>
                typename std::enable_if<n<Order-1, void>::type
                _initializeAndDim( std::vector<Type>& data, NestedInitializerListsT<Type,Order-n> init_list );

                template<int n>
                typename std::enable_if<n==Order-1, void>::type
                _initializeAndDim( std::vector<Type>& data, NestedInitializerListsT<Type,1> init_list );
            
            public:
                /*! @brief Constructor (main)
                 * This is an empty constructor
                 */
                tensor ( ){}
                /*! @brief Constructor (main)
                 *  @param dims Dimensions (size of Order or fails)
                 *
                 *  Usage examples:
                 *  @code
                 *  tensor<double,1> vector(2);
                 *  tensor<double,2> matrix(2,3);
                 *  @endcode
                 *  generates a vector (tensor of rank 1) of size 2 and a matrix (tensor of rank 2) of size (2*3)
                 */
                template <class... Dims>
                tensor ( int dim1, const Dims... dims );
            

                /*! @brief Constructor with pointer to data (useful to encapsulate data as a tensor)
                 * @param pointer where data is stored (data is not own by tensor if this constructor is called)
                 * @param dims Dimensions (size of Order or fails)
                 *  Usage examples:
                 * @code
                 * double data[9];
                 * tensor<double,2> mytensor(data,3,3);
                 * @endcode
                 * creates a tensor of order 2 (matrix) of size (3*3) and assumes that data for that tensor is provided by the array data
                 */
                template <class... Dims>
                tensor ( Type* pointer, const Dims... dims);
            
                /*!@brief Destructor */
                ~tensor()
                {
                    clear();
                }
            
                /*!@brief Copy constructor */
                template<class S>
                tensor(tensor<S,Order>& other);
            
                /*!@brief Copy assignment */
                template<class S>
                tensor& operator=(tensor<S,Order>& other);
            
                /*!@brief Moving constructor */
                tensor(tensor&& other);
            
                /*!@brief Moving assignment */
                tensor& operator=(tensor&& other);
            
                /*!@brief Initialize with initialize_list */
                void operator=(NestedInitializerListsT<Type,Order> init_list );
            
                tensor(NestedInitializerListsT<Type,Order> init_list);
            
            
                void clear();
                template <class... Dims>
                void resize(int dim1, const  Dims... dims);
                void set_pointer(Type* pointer);

            
                //--------------------------------------------------
                /** @} */ // end of constructors and assignment operators

                /** @name Helpers for Accessor to stored data
                *  @{ */
            protected:
                template <int isScalar, typename... Indices>
                struct _getSubspace;

                template <typename... Indices>
                struct _getSubspace<false, Indices...>
                {
                    static auto function(tensor<Type,Order>& thetensor, const Indices... indices) -> tensor<Type,_nElementsOfClass<sizeof...(Indices)==1,runningIndex,Indices...>::value>;
                };


                template <typename... Indices>
                struct _getSubspace<true, Indices...>
                {
                    static Type& function(tensor<Type,Order>& thetensor,const Indices... indices);
                };
                /** @} */ // end of Helpers for Accessor to stored data

                /** @name Accessor to stored data
                *  @{ */
            public:
                /*! @brief Get access to raw data
                 *
                 * @return Pointer to the location where data is stored (emulating std::vector behavior)
                 */
                Type* data() const
                {   return _data;   }


                 /*! @brief Main getter
                  *
                  * @tparam Indices general template parameter to accept integers or runningIndices
                  * @param indices Integers or runningIndices (e.g. all or range(init,end))
                  * @return Reference to a value if indices integers or a tensor of order equal to the number of free
                  * indices (subspace of the tensor) sharing memory with the parent tensor
                  *
                  * Usage example:
                  * @code
                  *  tensor<double,2> mytensor2(3,4);
                  *  for ( int i = 0; i < 3; i++ )
                  *      for (int j = 0; j < 4; j++ )
                  *          mytensor2(i,j) = 2*i+j;
                  *  tensor<double,3> mytensor3(3,3,4);
                  *  mytensor3(0,all,all) = mytensor2;
                  *  mytensor3(1,all,all) = 2*mytensor2;
                  *  mytensor3(2,all,all) = 3*mytensor2;
                  *  std::cout << "mytensor2 is : " << mytensor2 << std::endl;
                  *  std::cout << "mytensor3 is : " << mytensor3 << std::endl;
                  * @endcode
                  * will print:
                  * @code
                  * mytensor2 is: [[0,1,2,3],[2,3,4,5],[4,5,6,7]]
                  * mytensor3 is: [[[0,1,2,3],[2,3,4,5],[4,5,6,7]],[[0,2,4,6],[4,6,8,10],[8,10,12,14]],[[0,3,6,9],[6,9,12,15],[12,15,18,21]]]
                  * @endcode
                 g */
                template <class... Indices>
                auto operator() ( const Indices... indices ) -> decltype(_getSubspace<areInts<Indices...>::value,Indices...>::function(*this,indices...));
            
                std::array<int,Order> shape ( ) const 
                {   return _dims;    }
            
                int size() const
                {    return _size;    }
                /** @} */ // end of accessor


                /** @name Printing
                *  @{ */
                /*! @brief print
                 *
                 * @param out ostream where to print
                 * @param opening each dimension of the tensor will be opened with this string
                 * @param closing each dimension of the tensor will be closed with this string
                 * @param separator each dimension of the tensor will be separated with this string
                 */
                void print( std::ostream& out, const std::string opening, const std::string closing, const std::string separator) const
                {
                    _print<tensor,0,(0<Order-1)>::function( (*this), out, opening, closing, separator );
                }
                /*! @brief overload << operator for printing
                 *
                 * @param out out ostream where to print
                 * @param tens tensor to be printed
                 * @return ostream
                 */
                friend std::ostream& operator<<(std::ostream& out, tensor const& tens)
                {
                    tens.print( out, "[", "]", ",");
                    return out;
                }
                /** @} */ // end of accessor

                /** @name Simple Tensor operations (complex operations in TensorOperations.h)
                *  @{ */
                /*! @brief transpose of a tensor
                 *
                 * @param indices (contains the order in which the initial dimensions will acquire when transposed). If no indices are passed, then it swaps the first with last index
                 * @param copy (make a copy of the tensor data or just point to it)
                 * @return tensor of the same type and order but with tansposed dimensions
                 *
                 * [1] A simple example for a third-order tensor:
                 * @code
                 * tensor<double,3> sometensor = {{{0.0,1.0},{2.0,3.0},{4.0,5.0}},{{6.0,7.0},{8.0,9.0},{10.0,11.0}}};;
                 * cout << "sometensor_abc: " << sometensor << endl;
                 * cout << "sometensor_acb: " << sometensor.transpose({0,2,1}) << endl;
                 * cout << "sometensor_cba: " << sometensor.transpose({2,1,0}) << endl;
                 * cout << "sometensor_cba: " << sometensor.transpose() << endl;
                 * @endcode
                 * It will print:
                 * @code
                 * sometensor_abc: [[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]
                 * sometensor_acb: [[[0,2,4],[1,3,5]],[[6,8,10],[7,9,11]]]
                 * sometensor_cba: [[[0,6],[2,8],[4,10]],[[1,7],[3,9],[5,11]]]
                 * sometensor_cba: [[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]
                 * @endcode
                 * [2] This is an example of a transposed without copy (modifying values of the transpose changes the initial matrix):
                 * @code
                 * tensor<int,2> sometensor(2,2);
                 * sometensor = {0,1,2,3};
                 * cout << "sometensor: " << sometensor << endl;
                 * tensor<int,2> sometensort = sometensor.transpose({1,0},false);
                 * cout << "sometensort: " << sometensort << endl;
                 * sometensort(1,0) *= -1;
                 * cout << "sometensor: " << sometensor << endl;
                 * cout << "sometensort: " << sometensort << endl;
                 * @endcode
                 * It will print:
                 * @code
                 * sometensor: [[0,1],[2,3]]
                 * sometensort: [[0,2],[1,3]]
                 * sometensor: [[0,-1],[2,3]]
                 * sometensort: [[0,2],[-1,3]]
                 * @endcode
                 *
                 */
                tensor<Type,Order> transpose ( std::array<int,Order> indices, bool copy = true ) const;
                tensor<Type,Order> transpose ( bool copy = true ) const;

                tensor<Type,Order> T ( bool copy = false ) const
                {   return transpose( copy );   }
            
                Type det();
            
                tensor<Type,Order> inv ( );
                /** @} */ // end of tensor operations

            protected:
                /** @name Internal data
                *  @{ */
                Type* _data{nullptr};             //!< Real data stored in a long array
                int   _size{};                    //!< Size of the array (equal to _dims[0]*_jumps[0])
                bool  _storage{false};            //!< Is data stored? Otherwise tensor is just a wrapper for data
                std::array<int,Order> _dims{};    //!< Size of the dimensions
                std::array<int,Order> _jumps{};   //!< jumps[i] gives the jump in _data one has to do to jump 1 element in the ith dimension
                /** @} */ // end of internal data


                /** @name Internal Helpers
                *  @{ */
                template<class S, int n, bool noFinal>
                struct _print;

                template<class Idx>
                void _getVecIndex(int &vidx, Idx idx);
                template <class Idx, class... Indices>
                void _getVecIndex(int &vidx, Idx idx, Indices... indices);

                template <size_t childOrder,class Idx>
                typename std::enable_if<std::is_same<int, Idx>::value, void>::type
                _getVecIndexAndInitDim(int& vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx);

                template <size_t childOrder,class Idx>
                typename std::enable_if< std::is_same<runningIndex, Idx>::value, void>::type
                _getVecIndexAndInitDim(int& vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx);

                template <size_t childOrder,class Idx, class... Indices>
                typename std::enable_if< std::is_same<int, Idx>::value, void>::type
                _getVecIndexAndInitDim(int &vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx, Indices... indices);

                template <size_t childOrder,class Idx, class... Indices>
                typename std::enable_if< std::is_same<runningIndex, Idx>::value, void>::type
                _getVecIndexAndInitDim(int &vidx, int& i, std::array<int,childOrder>& childDims, std::array<int,childOrder>& childJumps, Idx idx, Indices... indices);

                /** @} */ // end of internal helpers
            
            public:

                template< class S, class U, int Order2 >
                friend auto operator+(tensor<S,Order2> const& tensor1, tensor<U,Order2> const& tensor2 ) -> tensor<decltype(tensor1._data[0]+tensor2._data[0]),Order2>;
            
                template< class S, class U, int Order2 >
                friend auto operator-(tensor<S,Order2> const& tensor1, tensor<U,Order2> const& tensor2 ) -> tensor<decltype(tensor1._data[0]-tensor2._data[0]),Order2>;
            
            
                //Operations with scalars
                template <class S, class U, int Order2>
                friend auto operator+(tensor<S,Order2> const& tensor1, U const& scalar ) -> tensor<decltype(tensor1._data[0]*scalar),Order2>;
                template <class S, class U, int Order2>
                friend auto operator+(U const& scalar,tensor<S,Order2> const& tensor1) -> tensor<decltype(tensor1._data[0]+scalar),Order2>;

                template <class S, class U, int Order2>
                friend auto operator-(tensor<S,Order2> const& tensor1, U const& scalar ) -> tensor<decltype(tensor1._data[0]*scalar),Order2>;
                template <class S, class U, int Order2>
                friend auto operator-(U const& scalar,tensor<S,Order2> const& tensor1) -> tensor<decltype(tensor1._data[0]-scalar),Order2>;

            
                template <class S, class U, int Order2>
                friend auto operator*(tensor<S,Order2> const& tensor1, U const& scalar ) -> tensor<decltype(tensor1._data[0]*scalar),Order2>;
                template <class S, class U, int Order2>
                friend auto operator*(U const& scalar,tensor<S,Order2> const& tensor1) -> tensor<decltype(tensor1._data[0]*scalar),Order2>;

                template <class S, class U, int Order2>
                friend auto operator/(tensor<S,Order2> const& tensor1, U const& scalar ) -> tensor<decltype(tensor1._data[0]/scalar),Order2>;

                template <class U>
                void operator=(U const& scalar);


                template <class U>
                void operator+=(U const& scalar);
                template <class U>
                void operator-=(U const& scalar);
                template <class U>
                void operator*=(U const& scalar);
                template <class U>
                void operator/=(U const& scalar);

                tensor<Type,Order> operator-()
                {
                    return (-1.0) * (*this); //FIXME: change this (it's only working well for doubles)
                }


                template <class U>
                void operator+=(tensor<U,Order> const& tensor2);
                template <class U>
                void operator-=(tensor<U,Order> const& tensor2);
                
                template <class T, class S, int O>
                friend auto eproduct(tensor<T,O> const& tensor1, tensor<S,O> const& tensor2 ) -> tensor<decltype(tensor1._data[0]*tensor2._data[0]),O>;
            
//                template <char C, class... Indices>
//                tensorindices<Type,Order,index<C>,Indices...> operator() ( index<C> firstIndex, const Indices... indices );
//
//                template<class T, class S, int Order1, int Order2, class ...Indices, class ...Indices2>
//                friend auto operator*(tensorindices<T,Order1,Indices...> tensorindices1, tensorindices<S,Order2,Indices2...> tensorindices2 ) -> tensor<decltype(tensorindices1.thetensor->_data[0] * tensorindices2.thetensor->_data[0]),countNonRepeatedIndices<Indices...,Indices2...>()>;


                /** @name Friend classes
                *  @{ */
                template<class S, int O, class... Indices>
                friend class tensorindices;
                template<class S, int Order2>
                friend class tensor;
                friend class TensorOperation;
                /** @} */ // end of friend classes

        };
        
    }
    

#include "TensorOperations.h"
#include "Tensor-impl.h"

#endif //HPLFE_TENSOR_H
