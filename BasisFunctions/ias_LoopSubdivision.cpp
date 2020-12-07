#include <math.h>
#include <set>

#include "ias_LoopSubdivision.h"

namespace ias
{
    LoopSubdivision::LoopSubdivision(Tensor::tensor<int,2> connec)
    {
        using namespace std;
        using namespace Tensor;
        
        //First calculate the total number of points in the connectivity.
        //FIXME: this could be problematic in the case the mesh is not simply connected
        int nPts{};
        for(int n = 0; n < connec.size(); n++)
        {
            if(nPts < connec.data()[n])
                nPts = connec.data()[n];
        }
        nPts++;
        
        //We first create an adjacency for every node with the elements it belongs to
        std::vector<std::vector<int>> adjyNE;
        adjyNE.resize(nPts);
        
        for(int e = 0; e < connec.shape()[0]; e++)
        {
            auto nbors = connec(e,all);
            adjyNE[nbors(0)].push_back(e);
            adjyNE[nbors(1)].push_back(e);
            adjyNE[nbors(2)].push_back(e);
        }
        
        //We now construct an adjacency with the first neighbours of every node. Note that we build it preserving the original mesh orientation. This is important so that all parametrisations are oriented in the same way
        std::vector<std::vector<int>> adjyNN;
        for(int i = 0; i < nPts; i++)
        {
            vector<int> nbors_n;
            set<int> nbors_e(adjyNE[i].begin(),adjyNE[i].end());
            
            bool finished = false;
            int e = adjyNE[i][0];
            nbors_e.erase(e);
            
            int j0 = connec(e,0) == i ? connec(e,2) : (connec(e,1) == i ? connec(e,0) : connec(e,1));
            int j  = connec(e,0) == j0 ? connec(e,2) : (connec(e,1) == j0 ? connec(e,0) : connec(e,1));
            
            while (not finished)
            {
                nbors_n.push_back(j);
                int jn = j;
                for(auto g: nbors_e)
                {
                    j = connec(g,0) == j ? connec(g,2) : (connec(g,1) == j ? connec(g,0) : (connec(g,2) == j ? connec(g,1) : j));
                    
                    if(j != jn)
                    {
                        nbors_e.erase(g);
                        break;
                    }
                }
                
                if(j == j0)
                    finished = true;
            }
            nbors_n.push_back(j0);

            adjyNN.emplace_back(nbors_n);
        }
        
        //Compute the adjacency (nodes whose basis functions are not zero in the given element)
        int nElem = connec.shape()[0];
        _etype.resize(nElem);
        _adjyEN_i.resize(nElem+1);
        _adjyEN_i[0] = 0;
        for(int e = 0; e < connec.shape()[0]; e++)
        {
            array<int,3> conn = {connec(e,0), connec(e,1), connec(e,2)};

            //Check if all nodes are irregular and throw an error if they are
            if(adjyNN[conn[0]].size() != 6 and adjyNN[conn[1]].size() != 6 and adjyNN[conn[2]].size() != 6)
                throw runtime_error("All nodes are irregular in element " + to_string(e) + " (" + to_string(adjyNN[conn[0]].size()) + ", " + to_string(adjyNN[conn[1]].size()) + ", " + to_string(adjyNN[conn[2]].size()) + "). You need to subdivide the mesh at least once!");
            
            //First check for the irregular node
            for( int i = 0; i < 3; i++)
            {
                if(adjyNN[conn[0]].size() != 6)
                    break;
             
                int aux = conn[0];
                conn[0] = conn[1];
                conn[1] = conn[2];
                conn[2] = aux;
            }

            _adjyEN_j.push_back(conn[0]);
            //First find where node 1 is in the list of neighbours of node 0
            auto it1 = find(adjyNN[conn[0]].begin(),adjyNN[conn[0]].end(),conn[1]);
            int j1 = it1-adjyNN[conn[0]].begin();
            int nNN1 = adjyNN[conn[0]].size();
            for(int i = 0; i < nNN1; i++)
            {
                int node = adjyNN[conn[0]][(i+j1) % nNN1];
                _adjyEN_j.push_back(node);
            }
            
            //Now find where node 2 is in the list of neighbours of node 1 (we start in the next position)
            auto it2 = find(adjyNN[conn[1]].begin(),adjyNN[conn[1]].end(),conn[2]);
            int j2 = it2-adjyNN[conn[1]].begin();
            int nNN2 = adjyNN[conn[1]].size(); //FIXME: This must be six anyway
            for(int i = 0; i < nNN2-3; i++)
            {
                int node = adjyNN[conn[1]][(i+j2+1) % nNN2];
                _adjyEN_j.push_back(node);
            }
            
            //Now find where node 0 is in the list of neighbours of node 2 (we start two positions way)
            auto it3 = find(adjyNN[conn[2]].begin(),adjyNN[conn[2]].end(),conn[0]);
            int j3 = it3-adjyNN[conn[2]].begin();
            int nNN3 = adjyNN[conn[2]].size(); //This must be six anyway
            _adjyEN_j.push_back(adjyNN[conn[2]][(j3+2+1) % nNN3]);
            _adjyEN_j.push_back(adjyNN[conn[2]][(j3+2+0) % nNN3]);

            _etype[e] = adjyNN[conn[0]].size()-4;
            _adjyEN_i[e+1] = _adjyEN_i[e] + adjyNN[conn[0]].size() + 6;
        }
        
        
        //TODO: I mark this as a todo but it is mostly to stress what we are doing here. Instead of computing only the valances present in the mesh (which would be more efficient here) we calculate all valances from 4 to 12 (any mesh with less than 4 or more than 12 as valance is a distaster anyway). That way every class will have the same valances and we don't need to send this information via MPI later (which saves time). Types are then valance-4 (0 to 8).
        for(int e = 0; e < nElem; e++)
            if(_etype[e] < 0 or _etype[e] > 8)
                throw runtime_error("Element " + to_string(e) + " has a valence " + to_string(_etype[e]) + " (<4 or > 12)" );
        _types = {0,1,2,3,4,5,6,7,8};
    }



    // allocate space and fill the the picking matrix 1
    Tensor::tensor<double,2> pickmtrx(int eval, int type)
    {
        using namespace Tensor;
        tensor<double,2> picking(12,eval+12);

        picking = 0.0;
        switch (type)
        {
            case 0:
            {
                picking( 3,      2) = 1.0;         picking( 4,     0) = 1.0;
                picking( 2, eval+3) = 1.0;         picking( 0,     1) = 1.0;
                picking( 5, eval  ) = 1.0;         picking( 9,eval+8) = 1.0;
                picking( 1, eval+2) = 1.0;         picking( 6,eval+1) = 1.0;
                picking(11, eval+4) = 1.0;         picking( 8,eval+7) = 1.0;
                picking( 7, eval+6) = 1.0;         picking(10,eval+9) = 1.0;
                break;
            }
            case 1:
            {
                picking( 3, eval+9) = 1.0;         picking( 4, eval+6) = 1.0;
                picking( 2, eval+4) = 1.0;         picking( 0, eval+1) = 1.0;
                picking( 5, eval+2) = 1.0;         picking( 9, eval+5) = 1.0;
                picking( 1, eval  ) = 1.0;         picking( 6,      1) = 1.0;
                picking(11, eval+3) = 1.0;         picking( 8, eval-1) = 1.0;
                picking( 7,      0) = 1.0;         picking(10,      2) = 1.0;
                break;
            }
            case 2:
            {
                picking( 3,       0) = 1.0;        picking( 4, eval- 1) = 1.0;
                picking( 2,       1) = 1.0;        picking( 0, eval   ) = 1.0;
                picking( 5, eval+ 5) = 1.0;        picking( 9, eval+ 2) = 1.0;
                picking( 1, eval+ 1) = 1.0;        picking( 6, eval+ 4) = 1.0;
                picking(11, eval+11) = 1.0;        picking( 8, eval+ 6) = 1.0;
                picking( 7, eval+ 9) = 1.0;        picking(10, eval+10) = 1.0;
                break;
            }
        }
       
        return picking;
    }


    Tensor::tensor<double,2> subdivisionmatrix (int eval)
    {
        using namespace std;
        using namespace Tensor;

        tensor<double,2> subdiv(eval+12,eval+12);
        subdiv = 0.0;
        
        subdiv(range(eval+1,eval+5),range(eval+1,eval+5)) =  {{0.125,  0.0,    0.0,    0.0,    0.0},
                                                              {0.0625, 0.0625, 0.0625, 0.0,    0.0},
                                                              {0.0,    0.0,    0.125,  0.0,    0.0},
                                                              {0.0625, 0.0,    0.0,    0.0625, 0.0625},
                                                              {0.0,    0.0,    0.0,    0.0,    0.125}};
        
        subdiv(range(eval+6,eval+11),range(eval+1,eval+5)) = { {0.375,  0.125,   0.0,    0.0,   0.0},
                                                               {0.125,  0.375,   0.125,  0.0,   0.0},
                                                               {0.0,    0.125,   0.375,  0.0,   0.0},
                                                               {0.375,  0.0,     0.0,    0.125, 0.0},
                                                               {0.125,  0.0,     0.0,    0.375, 0.125},
                                                               {0.0,    0.0,     0.0,    0.125, 0.375}};

        double c = 0.375;
        double d = 0.125;

        // Second line of S
        subdiv(1,0)=c;      subdiv(1,1)=c;     subdiv(1,2)=d;          subdiv(1,eval)=d;
        // Last line of matrix S
        subdiv(eval,0)=c;   subdiv(eval,1)=d;  subdiv(eval,eval-1)=d;  subdiv(eval,eval)=c;

        for(int i = 2; i < eval; i++)
        {
            subdiv(i, 0)   = c;
            subdiv(i, i-1) = d;
            subdiv(i, i)   = c;
            subdiv(i, i+1) = d;
        }

        double cs    = cos(2.0*M_PI/(double(eval)));
        double alpha = 5.0/8.0-(3.0+2.0*cs)*(3.0+2.0*cs)/64.0;
        double b     = alpha/(double(eval));

        subdiv(0,0) = 1.-alpha;
        subdiv(0,range(1,eval)) = b;
        
        subdiv(eval+1,    0) = 0.125;    subdiv(eval+1,      1) = 0.375;   subdiv(eval+1,   eval)  = 0.375;
        subdiv(eval+2,    0) = 0.0625;   subdiv(eval+2,      1) = 0.625;   subdiv(eval+2,      2) = 0.0625;   subdiv(eval+2, eval) = 0.0625;
        subdiv(eval+3,    0) = 0.125;    subdiv(eval+3,      1) = 0.375;   subdiv(eval+3,      2) = 0.375;
        subdiv(eval+4,    0) = 0.0625;   subdiv(eval+4,      1) = 0.0625;  subdiv(eval+4, eval-1) = 0.0625;   subdiv(eval+4, eval) = 0.625;
        subdiv(eval+5,    0) = 0.125;    subdiv(eval+5, eval-1) = 0.375;   subdiv(eval+5,   eval) = 0.375;
        
        
        subdiv(eval+ 6,      1) = 0.375;     subdiv(eval+6,  eval) = 0.125;
        subdiv(eval+ 7,      1) = 0.375;
        subdiv(eval+ 8,      1) = 0.375;     subdiv(eval+8,     2) = 0.125;
        subdiv(eval+ 9,      1) = 0.125;     subdiv(eval+9,  eval) = 0.375;
        subdiv(eval+10, eval  ) = 0.375;
        subdiv(eval+11, eval-1) = 0.125;     subdiv(eval+11, eval) = 0.375;

        return subdiv;
    }

    std::vector<std::vector<double>>  basisFunctionsDerNode0 ( int eval )
    {
        using namespace std;

        vector<std::vector<double>> ders;

        ders.resize(3);
        ders[0].resize(12);
        ders[1].resize(2*12);
        ders[2].resize(3*12);


        double N = static_cast<double>(eval);

        double subParam = (0.375+0.25*cos(2.0*M_PI/N));
        subParam *= (-subParam);
        subParam += 0.625;
        subParam /= N;
        
        double l = 1.0/(3.0/(8.0*subParam)+N);
        
        ders[0][0] = 0.0;
        ders[0][2*0+0] = 0.0;
        ders[0][2*0+1] = 0.0;
        ders[0][3*0+0] = 0.0;
        ders[0][3*0+1] = 0.0;
        ders[0][3*0+2] = 0.0;

        for ( int i = 1; i < eval+1; i ++)
        {
            double ci = cos(2.0*M_PI*(i-1)/N);
            double si = sin(2.0*M_PI*(i-1)/N);
            
            ders[0][i] = l;
            ders[1][2*i+0] = 2.0*si/N;
            ders[1][2*i+1] = 2.0*ci/N;
            ders[2][3*i+0] = 0.0;
            ders[2][3*i+1] = 0.0;
            ders[2][3*i+2] = 0.0;
        }
        for ( int i = eval+1; i < eval+6; i ++)
        {
            ders[0][i] = 0.0;
            ders[1][2*i+0] = 0.0;
            ders[1][2*i+1] = 0.0;
            ders[2][3*i+0] = 0.0;
            ders[2][3*i+1] = 0.0;
            ders[2][3*i+2] = 0.0;
        }

        return ders;
    }

    std::vector<std::vector<double>> LoopSubdivision::_computeIrrBasisFunctions (std::vector<double> xi, int eval)
    {
        using namespace std;
        using namespace Tensor;

        //Barycentric coordinates
        double v = xi[0];
        double w = xi[1];
        double u = 1.0-xi[0]-xi[1];

        if ( abs(u-1.0) < _eps )
        {
            return basisFunctionsDerNode0(eval);
        }

        // evaluate the number of the required subdivisions
        int na  = 0;
        double min = 0.0;
        double max = 0.5;

        while ( !((u>(min-_eps)) && (u<(max+_eps))) )
        {
            na++;
            min  = max;
            max += pow(2.0,-na-1);
        }
        
        //barycentric coordinates after subdivision
        int potenz = na+1;
        double pow2   = pow(2.0, na);
        v  *= pow2;
        w  *= pow2;
        u  = 1.0 - v - w;

        //check new barycentric coordinates
        assert( (u<(0.5+_eps)) && (u>-_eps) );

        double jfac = pow(2.0, na+1);
        
        int type{};
        // coordinate transformation
        if (v > (0.5-_eps))
        {
            v    = 2.0*v - 1.0;
            w    = 2.0*w;
            type = 0;
        }
        else if (w >(0.5-_eps))
        {
            v    = 2.0*v;
            w    = 2.0*w - 1.0;
            type = 2;
        }
        else
        {
            v    = 1.0 - 2.0*v;
            w    = 1.0 - 2.0*w;
            type = 1;
            jfac *= -1.0;
        }
        u    = 1.0 - v - w;
        
        //check new barycentric coordinates
        assert((u<(1.0+_eps))&&(u>-_eps));

        
        tensor<double,2> picking = pickmtrx(eval, type);
        tensor<double,2> subdiv  = subdivisionmatrix(eval);
        
        for(int i = 0; i < potenz-1; i++)
            subdiv = subdiv * subdiv;
        picking = picking * subdiv;
        
        auto boxplines = _computeBoxSplines ({v,w});
        
        vector<std::vector<double>> ders;
        ders.resize(3);
        ders[0].resize(eval+6);
        ders[1].resize(2*(eval+6));
        ders[2].resize(3*(eval+6));
        
        // compute the shape function values for irregular mesh (compute (P*A^n)*Co)
        for (int j=0; j<(eval+6); j++)
        {

            double sum_shape  = 0.0;
            double sum_derx   = 0.0;
            double sum_dery   = 0.0;
            double sum_hessxx = 0.0;
            double sum_hessyy = 0.0;
            double sum_hessxy = 0.0;

            for (int i=0; i<12; i++)
            {
                sum_shape   += picking(i,j) * boxplines[0][i];
                sum_derx    += picking(i,j) * boxplines[1][2*i+0];
                sum_dery    += picking(i,j) * boxplines[1][2*i+1];
                sum_hessxx  += picking(i,j) * boxplines[2][3*i+0];
                sum_hessyy  += picking(i,j) * boxplines[2][3*i+1];
                sum_hessxy  += picking(i,j) * boxplines[2][3*i+2];
            }

            ders[0][j]     =  sum_shape;
            ders[1][j*2+0] =  sum_derx * jfac;
            ders[1][j*2+1] =  sum_dery * jfac;
            ders[2][j*3+0] =  sum_hessxx * jfac * jfac;
            ders[2][j*3+1] =  sum_hessyy * jfac * jfac;
            ders[2][j*3+2] =  sum_hessxy * jfac * jfac;
        }
        
        return ders;
    }

    std::vector<std::vector<double>> LoopSubdivision::_computeBoxSplines (std::vector<double> xi)
    {
        using namespace std;
        
        vector<std::vector<double>> ders;
        
        ders.resize(3);
        ders[0].resize(12);
        ders[1].resize(2*12);
        ders[2].resize(3*12);
        
        
        //Barycentric coordinates
        double v = xi[0];
        double w = xi[1];
        double u = 1.0-xi[0]-xi[1];
        
        double u2 = u*u;
        double u3 = u2*u;
        double u4 = u3*u;

        double v2 = v*v;
        double v3 = v2*v;
        double v4 = v3*v;

        double w2 = w*w;
        double w3 = w2*w;
        double w4 = w3*w;
        
        
        ders[0][0]     = (6. * u4 + 24. * u3 * w + 24. * u2 * w2 + 8. * u * w3 + w4 + 24. * u3 * v + 60. * u2 * v * w + 36. * u * v * w2 + 6. * v * w3 + 24. * u2 * v2 + 36. * u * v2 * w + 12. * v2 * w2 + 8. * u * v3 + 6. * v3 * w + v4)/12.0;
        ders[1][2*0+0] = (-4.0 * v3 - 24.0 * v2 * u - 24.0 * v * u2 - 18.0 * v2 * w - 48.0 * v * u * w - 12.0 * u2 * w - 12.0 * v * w2 - 12.0 * u * w2 - 2.0 * w3)/12.0;
        ders[1][2*0+1] = (-2.0 * v3 - 12.0 * v2 * u - 12.0 * v * u2 - 12.0 * v2 * w - 48.0 * v * u * w - 24.0 * u2 * w - 18.0 * v * w2 - 24.0 * u * w2 - 4.0 * w3)/12.0;
        ders[2][3*0+0] = v2 - 2.0 * u2 + v * w - 2.0 * u * w;
        ders[2][3*0+1] = -2.0 * v * u - 2.0 * u2 + v * w + w2;
        ders[2][3*0+2] = (6.0 * v2 - 12.0 * u2 + 24.0 * v * w + 6.0 * w2)/12.0;
        
        ders[0][1]     = (u4 + 6. * u3 * w + 12. * u2 * w2 + 6. * u * w3 + w4 + 8. * u3 * v + 36. * u2 * v * w + 36. * u * v * w2 + 8. * v * w3 + 24. * u2 * v2 + 60. * u * v2 * w + 24. * v2 * w2 + 24. * u * v3 + 24. * v3 * w + 6. * v4)/12.0;
        ders[1][2*1+0] = (24.0 * v2 * u + 24.0 * v * u2 + 4.0 * u3 + 12.0 * v2 * w + 48.0 * v * u * w + 18.0 * u2 * w + 12.0 * v * w2 + 12.0 * u * w2 + 2.0 * w3)/12.0;
        ders[1][2*1+1] = (12.0 * v2 * u + 12.0 * v * u2 + 2.0 * u3 - 12.0 * v2 * w + 6.0 * u2 * w - 12.0 * v * w2 - 6.0 * u * w2 - 2.0 * w3)/12.0;
        ders[2][3*1+0] = (-24.0 * v2 + 12.0 * u2 - 24.0 * v * w + 12.0 * u * w)/12.0;
        ders[2][3*1+1] = (-24.0 * v2 - 24.0 * v * u - 24.0 * v * w - 24.0 * u * w)/12.0;
        ders[2][3*1+2] = (-12.0 * v2 + 6.0 * u2 - 24.0 * v * w - 12.0 * u * w - 6.0 * w2)/12.0;
        
        ders[0][2]     = (u4 + 2. * u3 * w + 6. * u3 * v + 6. * u2 * v * w + 12. * u2 * v2 + 6. * u* v2 * w + 6. * u * v3 + 2. * v3 * w + v4)/12.0;
        ders[1][2*2+0] = (-2.0 * v3 - 6.0 * v2 * u + 6.0 * v * u2 + 2.0 * u3)/12.0;
        ders[1][2*2+1] = (-4.0 * v3 - 18.0 * v2 * u - 12.0 * v * u2 - 2.0 * u3 - 6.0 * v2 * w - 12.0 * v * u * w - 6.0 * u2 * w)/12.0;
        ders[2][3*2+0] = -2.0 * v * u;
        ders[2][3*2+1] = v2 + v * u + v * w + u * w;;
        ders[2][3*2+2] = (6.0 * v2 - 12.0 * v * u - 6.0 * u2)/12.0;
        
        ders[0][3]     = (u4 + 2. * u3 * v)/12.0;
        ders[1][2*3+0] = (-6.0 * v * u2 - 2.0 * u3)/12.0;
        ders[1][2*3+1] = (-6.0 * v * u2 - 4.0 * u3)/12.0;
        ders[2][3*3+0] = v*u;
        ders[2][3*3+1] = v*u + u2;
        ders[2][3*3+2] = (12.0 * v * u + 6.0 * u2)/12.0;

        ders[0][4]     = (u4 + 2. * u3 * w)/12.0;
        ders[1][2*4+0] = (-4.0 * u3 - 6.0 * u2 * w)/12.0;
        ders[1][2*4+1] = (-2.0 * u3 - 6.0 * u2 * w)/12.0;
        ders[2][3*4+0] = u2 + u * w;
        ders[2][3*4+1] = u * w;
        ders[2][3*4+2] = (6.0 * u2 + 12.0 * u * w)/12.0;
        
        ders[0][5]     = (u4 + 6. * u3 * w + 12. * u2 * w2 + 6. * u * w3 + w4 + 2. * u3 * v + 6. * u2 * v * w + 6. * u * v * w2 + 2. * v * w3)/12.0;
        ders[1][2*5+0] = (-6.0 * v * u2 - 2.0 * u3 - 12.0 * v * u * w - 12.0 * u2 * w - 6.0 * v * w2 - 18.0 * u * w2 - 4.0 * w3)/12.0;
        ders[1][2*5+1] = (2.0 * u3 + 6.0 * u2 * w - 6.0 * u * w2 - 2.0 * w3)/12.0;
        ders[2][3*5+0] = v * u + v * w + u * w + w2;
        ders[2][3*5+1] = -2.0 * u * w;
        ders[2][3*5+2] = (-6.0 * u2 - 12.0 * u * w + 6.0 * w2)/12.0;
        
        ders[0][6]     = (u4 + 8 * u3 * w + 24. * u2 * w2 + 24. * u * w3 + 6. * w4 + 6. * u3 * v + 36. * u2 * v * w + 60. * u * v * w2 + 24. * v * w3 + 12. * u2 * v2 + 36. * u * v2 * w + 24. * v2 * w2 + 6. * u * v3 + 8. * v3 * w + v4)/12.0;
        ders[1][2*6+0] = (-2.0 * v3 - 6.0 * v2 * u + 6.0 * v * u2 + 2.0 * u3 - 12.0 * v2 * w + 12.0 * u2 * w - 12.0 * v * w2 + 12.0 * u * w2)/12.0;
        ders[1][2*6+1] = (2.0 * v3 + 12.0 * v2 * u + 18.0 * v * u2 + 4.0 * u3 + 12.0 * v2 * w + 48.0 * v * u * w + 24.0 * u2 * w + 12.0 * v * w2 + 24.0 * u * w2)/12.0;
        ders[2][3*6+0] = -2.0 * v * u - 2.0 * v * w - 2.0 * u * w - 2.0 * w2;
        ders[2][3*6+1] = v * u + u2 - 2.0 * v * w - 2.0 * w2;
        ders[2][3*6+2] = (-6.0 * v2 - 12.0 * v * u + 6.0 * u2 - 24.0 * v * w - 12.0 * w2)/12.0;
        
        ders[0][7]     = (2. * u * w3 + w4 + 6. * u * v * w2 + 6. * v * w3 + 6. * u * v2 * w + 12. * v2 * w2 + 2. * u * v3 + 6. * v3 * w + v4)/12.0;
        ders[1][2*7+0] = (2.0 * v3 + 6.0 * v2 * u + 12.0 * v2 * w + 12.0 * v * u * w + 18.0 * v * w2 + 6.0 * u * w2 + 4.0 * w3)/12.0;
        ders[1][2*7+1] = (4.0 * v3 + 6.0 * v2 * u + 18.0 * v2 * w + 12.0 * v * u * w + 12.0 * v * w2 + 6.0 * u * w2 + 2.0 * w3)/12.0;
        ders[2][3*7+0] = (12.0 * v * u + 12.0 * v * w + 12.0 * u * w + 12.0 * w2)/12.0;
        ders[2][3*7+1] = v2 + v * u + v * w + u * w;
        ders[2][3*7+2] = (6.0 * v2 + 12.0 * v * u + 24.0 * v * w + 12.0 * u * w + 6.0 * w2)/12.0;
        
        ders[0][8]     = (2. * v3 * w + v4)/12.0;
        ders[1][2*8+0] = (4.0 * v3 + 6.0 * v2 * w)/12.0;
        ders[1][2*8+1] = (2.0 * v3)/12.0;
        ders[2][3*8+0] = (12.0*v2 + 12.0 * v * w)/12.0;
        ders[2][3*8+1] = 0.0;
        ders[2][3*8+2] = v2/2.0;
        
        ders[0][9]     = (2. * u * v3 + v4)/12.0;
        ders[1][2*9+0] = (2.0 * v3 + 6.0 * v2 * u)/12.0;
        ders[1][2*9+1] = (-2.0 * v3)/12.0;
        ders[2][3*9+0] = v * u;
        ders[2][3*9+1] = 0.0;
        ders[2][3*9+2] = -v2/2.0;
        
        ders[0][10]     = (w4 + 2. * v * w3)/12.0;
        ders[1][2*10+0] = (2.0 * w3)/12.0;
        ders[1][2*10+1] = (6.0 * v * w2 + 4.0 * w3)/12.0;
        ders[2][3*10+0] = 0.0;
        ders[2][3*10+1] = v*w + w2;
        ders[2][3*10+2] = w2/2.0;

        ders[0][11]     = (2. * u * w3 + w4)/12.0;
        ders[1][2*11+0] = (-2.0 * w3)/12.0;
        ders[1][2*11+1] = (6.0 * u * w2 + 2.0 * w3)/12.0;
        ders[2][3*11+0] = 0.0;
        ders[2][3*11+1] = u * w;
        ders[2][3*11+2] = -w2/2.0;

        return ders;
    }

}
