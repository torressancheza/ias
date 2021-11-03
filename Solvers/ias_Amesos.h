//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _amesos_h
#define _amesos_h

#include <vector>
#include <string>

#include <Amesos.h>


#include "ias_Solver.h"

namespace ias
{
    namespace solvers
    {
    /*! @class TrilinosAmesos
     *  @brief Wrapper for the Amesos solver from Trilinos */
        class TrilinosAmesos : public Solver
        {
            public:

                /** @name Constructor/destructor
                *  @{ */
                /*! @brief Constructor */
                TrilinosAmesos(){};
                /*! @brief Destructor */
                ~TrilinosAmesos(){};
                /** @} */
            
                /** @name Copy/move constructors/assignments
                *  @{ */
                /*! @brief Copy contructor deleted   */
                TrilinosAmesos(const TrilinosAmesos&)             = delete;
                /*! @brief Copy assignment deleted */
                TrilinosAmesos& operator=(const TrilinosAmesos&)  = delete;
                /*! @brief Move constructor defaulted */
                TrilinosAmesos(TrilinosAmesos&&)                  = default;
                /*! @brief Copy constructor defaulted */
                TrilinosAmesos& operator=(TrilinosAmesos&&)       = default;
                /** @} */
            
                /** @name Update
                *  @{ */
                /*! @brief Use the information given by the user through setters to initialise (or resize) internal variables */
                void Update();
                /** @} */
            
                /** @name Solve
                *  @{ */
                /*! @brief Solve the system */
                void solve();
                /** @} */
            
            private:
                std::vector<std::string> _nameParams;
                std::vector<std::string> _params;

                Amesos _AmesFact;
                Teuchos::RCP<Amesos_BaseSolver> _solver = Teuchos::null;

        };
    }
}

#endif //_aztecoo_h
