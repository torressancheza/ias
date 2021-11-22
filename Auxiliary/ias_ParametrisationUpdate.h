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
#include "ias_Amesos.h"
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

            /*! @brief Set Maximum deltat (ALE) */
            void setMaximumTimeStep(double deltatMax)
            {    _deltatMax = deltatMax;    _ale_param_set = true;    }

            /*! @brief Set inplane friction (ALE) */
            void setInPlaneFriction(double friction)
            {    _tfriction = friction;    _ale_param_set = true;    }

             /*! @brief Set penalty shear (ALE)*/
            void setPenaltyShear(double penaltyShear)
            {    _penaltyShear = penaltyShear;    _ale_param_set = true;    }
             /*! @brief Set penalty stretch (ALE)*/
            void setPenaltyStretch(double penaltyStretch)
            {    _penaltyStretch = penaltyStretch;    _ale_param_set = true;    }
            
            /*! @brief Set penalty stretch (ALE)*/
            void setElasticSpringConstantReference(double elastRef)
            {    _elastRef = elastRef;    _ale_param_set = true;    }

            /*! @brief Set remove rigid translation */
            void setRemoveRigidBodyTranslation(bool remove_RBT)
            {    _remove_RBT = remove_RBT;    }

            /*! @brief Set remove rigid rotations */
            void setRemoveRigidBodyRotation(bool remove_RBR)
            {    _remove_RBR = remove_RBR;    }

            /*! @brief Set names of the displacement fields */
            void setDisplacementFieldNames(std::vector<std::string> dispFieldNames)
            {    _dispFieldNames = dispFieldNames;    }

            /*! @brief Creates all the structure  */
            void Update();
            /** @} */
            bool UpdateParametrisation();

        protected:
            Teuchos::RCP<Tissue> _tissue = Teuchos::null;
            Method _method = Method::Undefined;

            double _tfriction{1.E0};
            double _elastRef{1.E-1};
            double _penaltyShear{1.E0};
            double _penaltyStretch{1.E-1};
            double _deltatMax{1.E-2};
            std::vector<std::string> _dispFieldNames;
            std::vector<Teuchos::RCP<Cell>> _cells;
            std::vector<Teuchos::RCP<Tissue>> _tissues;
            std::vector<Teuchos::RCP<Integration>> _integrations;
            std::vector<Teuchos::RCP<solvers::Solver>> _linearSolvers;
            std::vector<Teuchos::RCP<solvers::NewtonRaphson>> _newtonRaphsons;

            bool _remove_RBT{};
            bool _remove_RBR{};
            bool _ale_param_set{};
            
            void _checkSettings();
                    
            std::function<void(Teuchos::RCP<SingleIntegralStr>)> _updateFunction;
            static void _centreOfMass(Teuchos::RCP<ias::SingleIntegralStr> fill);
            static void _eulerianUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill);
            static void _arbLagEulUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill);
    };
}

#endif //_ParametrisationUpdate_h