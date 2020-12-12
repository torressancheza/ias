//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _Assembly_h
#define _Assembly_h

namespace ias
{

    inline void AssembleElementalVector(size_t offset, size_t eNN, size_t nDOFs, int* loc2glo, double* bel, Teuchos::RCP<Epetra_FEVector> b)
    {
        for(size_t i=0; i < eNN; i++)
        {
            for(size_t p=0; p < nDOFs; p++)
            {
                int lrow = i*nDOFs+p;
                int grow = offset + nDOFs * loc2glo[i]+p;

                b->SumIntoGlobalValues(1, &grow, &bel[lrow]);
            }
        }
    }

    template <bool insert>
    inline void AssembleElementalMatrix(size_t offset_1, size_t offset_2, size_t eNN_1, size_t eNN_2, size_t nDOFs_1, size_t nDOFs_2, int* loc2glo_1, int* loc2glo_2, double* Ael, Teuchos::RCP<Epetra_FECrsMatrix> A)
    {
        std::vector<int> rows;
        std::vector<int> cols;
        
        for(size_t i=0; i < eNN_1; i++)
        {
            for(size_t p=0; p < nDOFs_1; p++)
            {
                rows.push_back(offset_1 + nDOFs_1*loc2glo_1[i]+p);
            }
            
        }
        
        for(size_t j=0; j < eNN_2; j++)
        {
            for(size_t q=0; q < nDOFs_2; q++)
            {
                cols.push_back(offset_2 + nDOFs_2*loc2glo_2[j]+q);
            }
        }

        insert ?  A->InsertGlobalValues(eNN_1*nDOFs_1, rows.data(), eNN_2*nDOFs_2, cols.data(), Ael, Epetra_FECrsMatrix::ROW_MAJOR) : A->SumIntoGlobalValues(eNN_1*nDOFs_1, rows.data(), eNN_2*nDOFs_2, cols.data(), Ael, Epetra_FECrsMatrix::ROW_MAJOR);

    }
}

#endif //_Assembly_h
