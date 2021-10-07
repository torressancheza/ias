//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef aux_SingleCell_h
#define aux_SingleCell_h

#include <random>

#include "ias_Tissue.h"
#include "ias_TissueGen.h"
#include "ias_Integration.h"
#include "ias_AztecOO.h"
#include "ias_NewtonRaphson.h"
#include "ConfigFile.h"

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill);
void leastSquares(Teuchos::RCP<ias::SingleIntegralStr> fill);

constexpr double friction  = 1.E-1;
constexpr double pressure  = 0.0;
constexpr double kappa     = 1.E-1;
constexpr double C0 = 1.0;


inline void computeFieldsSpherHarm(Tensor::tensor<double,1>& x0, double& tension, double& phi, double& vn)
{
    double theta = acos(x0(2)/sqrt(x0(0)*x0(0)+x0(1)*x0(1)+x0(2)*x0(2)));
    double varphi = atan2(x0(1),x0(0));
    double integrate{};
    for(int A = 0; A < 4; A++)
    {
        for(int a = -A; a <= A; a++)
        {
            double Afactor = A*(A+1);
            double f_tension = 0.0;//A>=2 ? (0.5*a*a)/(A*A): (A==0) ? 1.0: 0.0;
            double f_phi = f_tension * friction; 
            f_phi += (A == 0) ? (pressure + kappa*(2.0-C0)*C0)*sqrt(4*M_PI) : 0;
            f_phi /= Afactor*(1+friction)+(2+friction)*(friction-1);
            double f_vn = Afactor*f_phi-2.*f_tension;
            f_vn += (A == 0) ? (pressure + kappa*(2.0-C0)*C0)*sqrt(4*M_PI): 0;
            f_vn /= (2+friction);

            //Real spherical harmonic
            double Y = (a > 0) ? sqrt(2) * std::sph_legendre ( A, a, theta ) * cos(a*varphi): (a==0) ? std::sph_legendre ( A, a, theta ) : sqrt(2) * std::sph_legendre ( A, a, theta ) * sin(a*varphi);

            tension += f_tension * Y;
            phi += f_phi * Y;
            vn += f_vn * Y;
        }
    }
}
#endif //
