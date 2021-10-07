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
            /*! @brief Set normal friction (ALE) */
            void setNormalFriction(double friction)
            {    _nfriction = friction;    _ale_param_set = true;    }
            /*! @brief Set inplane friction (ALE) */
            void setInPlaneFriction(double friction)
            {    _tfriction = friction;    _ale_param_set = true;    }
             /*! @brief Set inplane viscosity (ALE) */
            void setInPlaneViscosity(double viscosity)
            {    _viscosity = viscosity;    _ale_param_set = true;    }
             /*! @brief Set penalty shear (ALE)*/
            void setPenaltyShear(double penaltyShear)
            {    _penaltyShear = penaltyShear;    _ale_param_set = true;    }
             /*! @brief Set penalty stretch (ALE)*/
            void setPenaltyStretch(double penaltyStretch)
            {    _penaltyStretch = penaltyStretch;    _ale_param_set = true;    }
            /*! @brief Set maximum shear  (ALE)*/
            void setMaximumShear(double maxShear)
            {    _maxShear = maxShear;    _ale_param_set = true;    }
            /*! @brief Set maximum stretch (ALE)*/
            void setMaximumStretch(double maxStretch)
            {    _maxStretch = maxStretch;    _ale_param_set = true;    }
            /*! @brief Set maximum stretch (ALE)*/
            void setMinimumStretch(double minStretch)
            {    _minStretch = minStretch;    _ale_param_set = true;    }

            /*! @brief Set remove rigid translation */
            void setRemoveRigidBodyTranslation(bool remove_RBT)
            {    _remove_RBT = remove_RBT;    }

            /*! @brief Set remove rigid rotations */
            void setRemoveRigidBodyRotation(bool remove_RBR)
            {    _remove_RBR = remove_RBR;    }

            /*! @brief Set names of the displacement fields */
            void setDisplacementFieldNames(std::vector<std::string> dispFieldNames)
            {   
                _dispFieldNames = dispFieldNames;    
            }

            /*! @brief Creates all the structure  */
            void Update();
            /** @} */
            bool UpdateParametrisation();

        protected:
            Teuchos::RCP<Tissue> _tissue = Teuchos::null;
            Method _method = Method::Undefined;

            double _tfriction{1.E0};
            double _nfriction{1.E3};
            double _viscosity{1.E0}; //FIXME: delete it
            double _penaltyShear{1.E0};
            double _penaltyStretch{1.E-1};
            double _maxShear{0.5};
            double _maxStretch{1.25};
            double _minStretch{0.0};
            std::vector<std::string> _dispFieldNames;
            std::vector<Teuchos::RCP<Tissue>> _tissues;
            std::vector<Teuchos::RCP<Integration>> _integrations;
            std::vector<Teuchos::RCP<solvers::TrilinosAztecOO>> _linearSolvers;
            std::vector<Teuchos::RCP<solvers::NewtonRaphson>> _newtonRaphsons;
            std::vector<Tensor::tensor<double,2>> _x0;
            std::vector<Tensor::tensor<double,1>> _normalErr;

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