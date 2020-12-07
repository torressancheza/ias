#ifndef aux_mzg_h
#define aux_mzg_h

#include <random>

#include "ias_Tissue.h"
#include "ias_TissueGen.h"
#include "ias_Integration.h"
#include "ias_NewtonRaphson.h"
#include "ConfigFile.h"

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill);
void interaction(Teuchos::RCP<ias::DoubleIntegralStr> fill);
void eulerianUpdate(Teuchos::RCP<ias::SingleIntegralStr> fill);

inline double SmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1)
    {
        return 1.0;
    }
    else if (r<rc2)
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x3 = x*x*x;
        double x4 = x3*x;
        double x5=x4*x;
        return -6.0*x5 - 15.0*x4 - 10.0 * x3;
    }
    else
    {
        return 0.0;
    }
}

inline double dSmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1 or r>rc2)
    {
        return 0.0;
    }
    else
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x2 = x*x;
        double x3 = x2*x;
        double x4 = x3*x;
        return (-30.0*x4 - 60.0*x3 - 30.0 * x2)/(rc2 - rc1);
    }
}

inline double ddSmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1 or r>rc2)
    {
        return 0.0;
    }
    else
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x2 = x*x;
        double x3 = x2*x;
        return (-120.0*x3 - 180.0*x2 - 60.0 * x)/(rc2 - rc1)/(rc2 - rc1);
    }
}

inline double MorsePotential(double D, double r0, double w, double r)
{
    return D * ((exp(-2.0*(r-r0)/w)-2.0*exp(-(r-r0)/w)));
}

inline double dMorsePotential(double D, double r0, double w, double r)
{
    return D * 2.0/w * (-exp(-2.0*(r-r0)/w)+exp(-(r-r0)/w));
}

inline double ddMorsePotential(double D, double r0, double w, double r)
{
    return D * 2.0/(w*w) * (2.0 * exp(-2.0*(r-r0)/w)-exp(-(r-r0)/w));
}

inline double ModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return MorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r);
}
inline double dModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return dMorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r) + MorsePotential(D, r0, w, r) * dSmoothStep(rc1, rc2, r);
}
inline double ddModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return ddMorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r) + 2.0 * dMorsePotential(D, r0, w, r) * dSmoothStep(rc1, rc2, r) + MorsePotential(D, r0, w, r) * ddSmoothStep(rc1, rc2, r);
}

#endif //aux_mzg_h
