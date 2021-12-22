//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <math.h>
#include <iostream>
#include <stdexcept>

#include "ias_CubatureGauss.h"

using namespace std;

namespace ias
{

    //Object creation management
    CubatureGauss::CubatureGauss(int iPts, int pDim)
    {
        _iPts = iPts;
        _pDim = pDim;

        _checkSettings();
        
        _wSamples.resize(_iPts);
        _xSamples.resize(_iPts);
        for(auto& x: _xSamples)
            x.resize(pDim);
        
        // Performing the quadrature in the reference element
        switch(_pDim)
        {
            case 1:
//                _lineRule();
                _quadratureRule();
                break;
            case 2:
                _triangularRule();
                break;
        }
    }

    void CubatureGauss::_checkSettings()
    {
        using namespace std;

        // Line quadrature rule
        if (_pDim==1)
        {
            if (this->_iPts == 1){}
            else if (this->_iPts ==  2){}
            else if (this->_iPts ==  3){}
            else if (this->_iPts ==  4){}
            else if (this->_iPts ==  5){}
            else if (this->_iPts ==  6){}
            else if (this->_iPts ==  7){}
            else if (this->_iPts ==  8){}
            else if (this->_iPts ==  9){}
            else if (this->_iPts == 10){}
            else if (this->_iPts == 15){}
            else
                throw runtime_error("CubatureGauss: CheckSettings failed!!");
        }
        else if (_pDim==2)
        {
            if (this->_iPts == 1){}
            else if (this->_iPts  ==   3){}
            else if (this->_iPts  ==   4){}
            else if (this->_iPts  ==   6){}
            else if (this->_iPts  ==   7){}
            else if (this->_iPts  ==  12){}
            else
                throw runtime_error("CubatureGauss: CheckSettings failed!!");
        }

    }

    // Weights and canonical variables for integration on line
    void CubatureGauss::_lineRule()
    {
        int np;
        np = (int) (this->_iPts);

        double *pt = new double[np];
        double *gw = new double[np];

        if (this->_iPts == 1)
        {
            pt[0] = 0.0;
            gw[0] = 2.0;
        }
        else if (this->_iPts == 2)
        {
            pt[0] =-1.0/sqrt(3.0);
            pt[1] = 1.0/sqrt(3.0);
            gw[0] = 1.0;
            gw[1] = 1.0;
        }
        else if (this->_iPts == 3)
        {
            pt[0] =-sqrt(3.0/5.0);
            pt[1] = 0.0;
            pt[2] = sqrt(3.0/5.0);

            gw[0] = 5.0/9.0;
            gw[1] = 8.0/9.0;
            gw[2] = 5.0/9.0;
        }
        else if (this->_iPts == 4)
        {
            pt[0] =-sqrt((3.0 + 2.0*sqrt(6.0/5.0))/7.0);
            pt[1] =-sqrt((3.0 - 2.0*sqrt(6.0/5.0))/7.0);
            pt[2] = sqrt((3.0 - 2.0*sqrt(6.0/5.0))/7.0);
            pt[3] = sqrt((3.0 + 2.0*sqrt(6.0/5.0))/7.0);

            gw[0] = 1.0/2.0 - sqrt(5.0/6.0)/6.0;
            gw[1] = 1.0/2.0 + sqrt(5.0/6.0)/6.0;
            gw[2] = 1.0/2.0 + sqrt(5.0/6.0)/6.0;
            gw[3] = 1.0/2.0 - sqrt(5.0/6.0)/6.0;
        }
        else if (this->_iPts == 5)
        {
            pt[0] =-sqrt(5.0 + 2.0*sqrt(10.0/7.0))/3.0;
            pt[1] =-sqrt(5.0 - 2.0*sqrt(10.0/7.0))/3.0;
            pt[2] = 0.0;
            pt[3] = sqrt(5.0 - 2.0*sqrt(10.0/7.0))/3.0;
            pt[4] = sqrt(5.0 + 2.0*sqrt(10.0/7.0))/3.0;

            gw[0] = (322.0 - 13.0*sqrt(70.0))/900.0;
            gw[1] = (322.0 + 13.0*sqrt(70.0))/900.0;
            gw[2] = 128.0 / 225.0;
            gw[3] = (322.0 + 13.0*sqrt(70.0))/900.0;
            gw[4] = (322.0 - 13.0*sqrt(70.0))/900.0;
        }
        else if (this->_iPts == 6)
        {
            pt[0] = -0.9324695142031520;
            pt[1] = -0.6612093864662645;
            pt[2] = -0.2386191860831969;
            pt[3] = 0.2386191860831969;
            pt[4] = 0.6612093864662645;
            pt[5] = 0.9324695142031520;

            gw[0] = 0.1713244923791703;
            gw[1] = 0.3607615730481386;
            gw[2] = 0.4679139345726911;
            gw[3] = 0.4679139345726911;
            gw[4] = 0.3607615730481386;
            gw[5] = 0.1713244923791703;
        }
        else if (this->_iPts == 7)
        {
            pt[0] = -0.9491079123427585;
            pt[1] = -0.7415311855993944;
            pt[2] = -0.4058451513773972;
            pt[3] = 0e0;
            pt[4] = 0.4058451513773972;
            pt[5] = 0.7415311855993944;
            pt[6] = 0.9491079123427585;

            gw[0] = 0.1294849661688697;
            gw[1] = 0.2797053914892767;
            gw[2] = 0.3818300505051189;
            gw[3] = 0.4179591836734694;
            gw[4] = 0.3818300505051189;
            gw[5] = 0.2797053914892767;
            gw[6] = 0.1294849661688697;
        }
        else if (this->_iPts == 8)
        {
            pt[0] = -0.9602898564975362;
            pt[1] = -0.7966664774136267;
            pt[2] = -0.5255324099163290;
            pt[3] = -0.1834346424956498;
            pt[4] = 0.1834346424956498;
            pt[5] = 0.5255324099163290;
            pt[6] = 0.7966664774136267;
            pt[7] = 0.9602898564975362;

            gw[0] = 0.1012285362903763;
            gw[1] = 0.2223810344533745;
            gw[2] = 0.3137066458778873;
            gw[3] = 0.3626837833783620;
            gw[4] = 0.3626837833783620;
            gw[5] = 0.3137066458778873;
            gw[6] = 0.2223810344533745;
            gw[7] = 0.1012285362903763;
        }
        else if (this->_iPts == 9)
        {
            pt[0] = -0.9681602395076261;
            pt[1] = -0.8360311073266358;
            pt[2] = -0.6133714327005904;
            pt[3] = -0.3242534234038089;
            pt[4] = 0e0;
            pt[5] = 0.3242534234038089;
            pt[6] = 0.6133714327005904;
            pt[7] = 0.8360311073266358;
            pt[8] = 0.9681602395076261;

            gw[0] = 0.0812743883615744;
            gw[1] = 0.1806481606948574;
            gw[2] = 0.2606106964029354;
            gw[3] = 0.3123470770400029;
            gw[4] = 0.3302393550012598;
            gw[5] = 0.3123470770400028;
            gw[6] = 0.2606106964029355;
            gw[7] = 0.1806481606948574;
            gw[8] = 0.0812743883615744;
        }
        else if (this->_iPts == 10)
        {
            pt[0] = -0.9739065285171717;
            pt[1] = -0.8650633666889845;
            pt[2] = -0.6794095682990244;
            pt[3] = -0.4333953941292471;
            pt[4] = -0.1488743389816312;
            pt[5] = 0.1488743389816312;
            pt[6] = 0.4333953941292471;
            pt[7] = 0.6794095682990244;
            pt[8] = 0.8650633666889845;
            pt[9] = 0.9739065285171717;

            gw[0] = 0.0666713443086881;
            gw[1] = 0.1494513491505805;
            gw[2] = 0.2190863625159820;
            gw[3] = 0.2692667193099963;
            gw[4] = 0.2955242247147528;
            gw[5] = 0.2955242247147528;
            gw[6] = 0.2692667193099963;
            gw[7] = 0.2190863625159820;
            gw[8] = 0.1494513491505805;
            gw[9] = 0.0666713443086881;
        }
        else if (this->_iPts == 15)
        {
            pt[0] = -0.9879925180204854;
            pt[1] = -0.9372733924007059;
            pt[2] = -0.8482065834104272;
            pt[3] = -0.7244177313601700;
            pt[4] = -0.5709721726085388;
            pt[5] = -0.3941513470775634;
            pt[6] = -0.2011940939974345;
            pt[7] = 0e0;
            pt[8] = 0.2011940939974345;
            pt[9] = 0.3941513470775634;
            pt[10] = 0.5709721726085388;
            pt[11] = 0.7244177313601700;
            pt[12] = 0.8482065834104272;
            pt[13] = 0.9372733924007059;
            pt[14] = 0.9879925180204854;

            gw[0] = 0.03075324199611807;
            gw[1] = 0.07036604748811134;
            gw[2] = 0.1071592204671351;
            gw[3] = 0.1395706779261761;
            gw[4] = 0.1662692058169852;
            gw[5] = 0.1861610000155741;
            gw[6] = 0.1984314853271374;
            gw[7] = 0.2025782419255562;
            gw[8] = 0.1984314853271374;
            gw[9] = 0.1861610000155741;
            gw[10] = 0.1662692058169852;
            gw[11] = 0.1395706779261761;
            gw[12] = 0.1071592204671351;
            gw[13] = 0.07036604748811134;
            gw[14] = 0.03075324199611807;
        }

        for (int i = 0; i< np; i++)
        {
            this->_xSamples[i][0] = pt[i];
            this->_wSamples[i] = gw[i];
        }

        delete [] pt;
        delete [] gw;

        return;
    }

    void CubatureGauss::_quadratureRule ()
    {
        int i, k, m, iback;
        int    mp1mi, ncopy, nmove, order2;
        double d1, d2pn, d3pn, d4pn;
        double dp, dpn;
        double e1, fx, h;
        double p, pk;
        double pkm1, pkp1;
        double t, u, v;
        double x0, xtemp, iorder;

        e1 = ( double ) ( this->_iPts * ( this->_iPts + 1 ) );
        m  = ( this->_iPts + 1 ) / 2;

        order2 = this->_iPts*this->_iPts;
        iorder = 1.0/((double)this->_iPts);


        for ( i = 1; i <= m; i++ )
        {
            mp1mi = m + 1 - i;
            t     = (double) ( 4 * i - 1 ) * M_PI/((double) (4 * this->_iPts + 2));
            x0    = cos(t) * ( 1.0 - (1.0 - iorder)/( (double)(8*order2) ));
            pkm1  = 1.0;
            pk    = x0;

            for ( k = 2; k <= this->_iPts; k++ )
            {
                pkp1 = 2.0*x0*pk - pkm1 - ( x0*pk - pkm1 )/( (double) k );
                pkm1 = pk;
                pk = pkp1;
            }

            d1 = ( double ) ( this->_iPts ) * ( pkm1 - x0 * pk );

            dpn  = d1 / ( 1.0-x0*x0 );
            d2pn = ( 2.0*x0*dpn  - e1 * pk ) / (1.0 - x0*x0);
            d3pn = ( 4.0*x0*d2pn + ( 2.0 - e1 ) * dpn )  / (1.0 - x0*x0);
            d4pn = ( 6.0*x0*d3pn + ( 6.0 - e1 ) * d2pn ) / (1.0 - x0*x0);

            u = pk / dpn;
            v = d2pn / dpn;
            //
            //  Initial approximation H:
            //
            h = -u * ( 1.0 + 0.5*u * ( v + u * ( v*v - d3pn/( 3.0 * dpn ) ) ) );
            //
            //  Refine H using one step of Newton's method:
            //
            p  =  pk + h * ( dpn + 0.5*h * ( d2pn + (h/3.0) * ( d3pn + 0.25*h*d4pn ) ) );
            dp = dpn + h * ( d2pn + 0.5*h * ( d3pn + h*d4pn/3.0 ) );
            h  = h - p / dp;

            xtemp = x0 + h;
            fx    = d1 - h * e1 * ( pk + 0.5 * h * ( dpn + h / 3.0
                    * ( d2pn + 0.25 * h * ( d3pn + 0.2 * h * d4pn ) ) ) );

            this->_xSamples[mp1mi-1][0] = xtemp;
            this->_wSamples[mp1mi-1] = 2.0 * ( 1.0 - xtemp * xtemp ) / ( fx * fx );
        }

        if ( ( this->_iPts % 2 ) == 1 )
        {
            this->_xSamples[0][0] = 0.0;
        }

        //  Shift the data up.
        //
        nmove = ( this->_iPts + 1 ) / 2;
        ncopy = this->_iPts - nmove;

        for ( i = 1; i <= nmove; i++ )
        {
            iback = this->_iPts + 1 - i;
            this->_xSamples[iback-1] = this->_xSamples[iback-ncopy-1];
            this->_wSamples[iback-1] = this->_wSamples[iback-ncopy-1];
        }

        //  Reflect values for the negative abscissas.
        //
        for ( i = 0; i < (this->_iPts - nmove); i++ )
        {
            this->_xSamples[i][0] = -(this->_xSamples[this->_iPts-i-1][0]);
            this->_wSamples[i] =  this->_wSamples[this->_iPts-i-1];
        }

        for (i = 0; i < this->_iPts; i++ )
        {
            this->_xSamples[i][0] *= 0.5;
            this->_xSamples[i][0] += 0.5;
            this->_wSamples[i] *= 0.5;
        }
        return;
    }


    void CubatureGauss::_triangularRule()
    {
        double a, b, c;
        double P1, P2, P3;


        switch(this->_iPts)
        {
            case 1:     //order: 1
                this->_xSamples[0][0] = 1.0/3.0;      this->_xSamples[0][1] = 1.0/3.0;
                this->_wSamples[0] = 1.0;
                break;

            case 3:     //order: 2
                this->_xSamples[0][0] = 2.0/3.0;      this->_xSamples[0][1] = 1.0/6.0;
                this->_xSamples[1][0] = 1.0/6.0;      this->_xSamples[1][1] = 2.0/3.0;
                this->_xSamples[2][0] = 1.0/6.0;      this->_xSamples[2][1] = 1.0/6.0;

                this->_wSamples[0] = 1.0/3.0;
                this->_wSamples[1] = 1.0/3.0;
                this->_wSamples[2] = 1.0/3.0;

                break;

             case 4:     //order: 3
                 this->_xSamples[0][0] = 1.0/3.0;  this->_xSamples[0][1] = 1.0/3.0;
                 this->_xSamples[1][0] = 0.6;      this->_xSamples[1][1] = 0.2;
                 this->_xSamples[2][0] = 0.2;      this->_xSamples[2][1] = 0.6;
                 this->_xSamples[3][0] = 0.2;      this->_xSamples[3][1] = 0.2;

                 this->_wSamples[0] = -27.0/48.0;
                 this->_wSamples[1] =  25.0/48.0;
                 this->_wSamples[2] =  25.0/48.0;
                 this->_wSamples[3] =  25.0/48.0;

                 break;

            case 6:     //order: 4
                a = 0.816847572980459;
                b = 0.091576213509771;
                this->_xSamples[0][0] = a;         this->_xSamples[0][1] = b;
                this->_xSamples[1][0] = b;         this->_xSamples[1][1] = a;
                this->_xSamples[2][0] = b;         this->_xSamples[2][1] = b;

                a = 0.108103018168070;
                b = 0.445948490915965;
                this->_xSamples[3][0] = a;        this->_xSamples[3][1] = b;
                this->_xSamples[4][0] = b;        this->_xSamples[4][1] = a;
                this->_xSamples[5][0] = b;        this->_xSamples[5][1] = b;

                P1 = 0.109951743655322;
                P2 = 0.223381589678011;
                for (int i=0; i< 3; i++) this->_wSamples[i] = P1;
                for (int i=3; i< 6; i++) this->_wSamples[i] = P2;
                break;

            case 7:     //order: 5
                a = 0.797426985353087;
                b = 0.101286507323456;
                this->_xSamples[0][0] = a;           this->_xSamples[0][1] = b;
                this->_xSamples[1][0] = b;           this->_xSamples[1][1] = a;
                this->_xSamples[2][0] = b;           this->_xSamples[2][1] = b;

                a = 0.059715871789770;
                b = 0.470142064105115;
                this->_xSamples[3][0] = a;           this->_xSamples[3][1] = b;
                this->_xSamples[4][0] = b;           this->_xSamples[4][1] = a;
                this->_xSamples[5][0] = b;           this->_xSamples[5][1] = b;

                this->_xSamples[6][0] = 1.0/3.0;     this->_xSamples[6][1] = 1.0/3.0;

                P1 = 0.125939180544827;
                P2 = 0.132394152788506;
                for (int i=0; i< 3; i++) this->_wSamples[i] = P1;
                for (int i=3; i< 6; i++) this->_wSamples[i] = P2;
                this->_wSamples[6] = 0.225;

                break;

            case 12:        //order: 6
                a = 0.873821971016996;
                b = 0.063089014491502;
                this->_xSamples[0][0] = a;           this->_xSamples[0][1] = b;
                this->_xSamples[1][0] = b;           this->_xSamples[1][1] = a;
                this->_xSamples[2][0] = b;           this->_xSamples[2][1] = b;

                a = 0.501426509658179;
                b = 0.249286745170910;
                this->_xSamples[3][0] = a;           this->_xSamples[3][1] = b;
                this->_xSamples[4][0] = b;           this->_xSamples[4][1] = a;
                this->_xSamples[5][0] = b;           this->_xSamples[5][1] = b;

                a = 0.636502499121399;
                b = 0.310352451033785;
                c = 0.053145049844816;
                this->_xSamples[6] [0] = a;           this->_xSamples[6] [1] = b;
                this->_xSamples[7] [0] = b;           this->_xSamples[7] [1] = a;
                this->_xSamples[8] [0] = b;           this->_xSamples[8] [1] = c;
                this->_xSamples[9] [0] = a;           this->_xSamples[9] [1] = c;
                this->_xSamples[10][0] = c;           this->_xSamples[10][1] = a;
                this->_xSamples[11][0] = c;           this->_xSamples[11][1] = b;

                P1 = 0.050844906370207;
                P2 = 0.116786275726379;
                P3 = 0.082851075618374;
                for (int i=0; i< 3; i++) this->_wSamples[i] = P1;
                for (int i=3; i< 6; i++) this->_wSamples[i] = P2;
                for (int i=6; i<12; i++) this->_wSamples[i] = P3;

                break;

        }

        //due to the fact that the weight sum up to one we half them to mach the area of
        //a isoparametric triangle of side 1: vertices [(0,0),(1,0),(0,1)]
        for ( int i=0; i<_iPts; i++ )
            this->_wSamples[i] *= 0.5;
        return;
    }

}// end namespace
