//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef _ParametrisationUpdate_h
#define _ParametrisationUpdate_h

#include <string>
#include <iostream>

#include <Teuchos_RCP.hpp>

#include "ias_Tissue.h"
#include "ias_BasicStructures.h"
#include "ias_Integration.h"
#include "ias_AztecOO.h"
#include "ias_NewtonRaphson.h"

namespace ias
{
    
    class ParametrisationUpdate
    {
        public:

            enum class Method
            {
                Undefined = 0,
                Lagrangian = 1,
                Eulerian = 2,
                ALE = 3
            };

            /** @name Important setters/getters
             *  @{ */
             /*! @brief Set tissue */
            void setTissue( Teuchos::RCP<Tissue> tissue )
            {    _tissue = tissue;    }
             /*! @brief Set method */
            void setMethod( Method method)
            {    _method = method;    }
             /*! @brief Set inplane viscosity (ALE) */
            void setInplaneViscosity(double viscosity)
            {    _viscosity = viscosity;    }
             /*! @brief Set penalty shear */
            void setPenaltyShear(double penaltyShear)
            {    _penaltyShear = penaltyShear;    }
             /*! @brief Set penalty stretch */
            void setPenaltyStretch(double penaltyStretch)
            {    _penaltyStretch = penaltyStretch;    }
            /*! @brief Set maximum shear */
            void setMaximumShear(double maxShear)
            {    _maxShear = maxShear;    }
            /*! @brief Set maximum stretch */
            void setMaximumStretch(double maxStretch)
            {    _maxStretch = maxStretch;    }
            /*! @brief Set maximum stretch */
            void setMinimumStretch(double minStretch)
            {    _minStretch = minStretch;    }


            /*! @brief Set names of the velocity fields */
            void setVelocityFieldNames(std::vector<std::string> velFieldNames)
            {    _velFieldNames = velFieldNames;    }
            /*! @brief Set names of the fields storing the previous position*/
            void setPreviousPositionFieldNames(std::vector<std::string> prevPosFieldNames)
            {    _prevPosFieldNames = prevPosFieldNames;    }
            /*! @brief Set names of the fields storing the reference position*/
            void setReferencePositionFieldNames(std::vector<std::string> refPosFieldNames)
            {    _refPosFieldNames = refPosFieldNames;    }
            /*  }@ */

            /*! @brief Creates all the structure  */
            void Update();
            /** @} */
            void UpdateParametrisation();

        protected:
            Teuchos::RCP<Tissue> _tissue = Teuchos::null;
            Method _method = Method::Undefined;

            double _viscosity{};
            double _penaltyShear{};
            double _penaltyStretch{};
            double _maxShear{};
            double _maxStretch{};
            double _minStretch{};
            std::vector<std::string> _velFieldNames;
            std::vector<std::string> _prevPosFieldNames;
            std::vector<std::string> _refPosFieldNames;
            std::vector<Teuchos::RCP<Tissue>> _tissues;
            std::vector<Teuchos::RCP<Integration>> _integrations;
            std::vector<Teuchos::RCP<solvers::TrilinosAztecOO>> _linearSolvers;
            std::vector<Teuchos::RCP<solvers::NewtonRaphson>> _newtonRaphsons;
            
            void _checkSettings();

            void _eulerianUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill);
            void _arbLagEulUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill);
            
    };
}

#endif //_ParametrisationUpdate_h