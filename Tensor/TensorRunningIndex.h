
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

#ifndef HPLFE_TENSORRUNNINGINDEX_H
#define HPLFE_TENSORRUNNINGINDEX_H

    namespace Tensor
    {
        struct runningIndex
        {
            friend int operator * (runningIndex r, int i)
            {
                return i*r.first;
            }
            
            friend int operator * (int i, runningIndex r)
            {
                return i*r.first;
            }
            
            int first{};
            int last{};
            
            
            runningIndex()
            {            }
            
            runningIndex(int _first, int _last)
            {
                first = _first;
                last  = _last;
            }
            
            int size()
            {
                return last-first+1;
            }
        };
        
        typedef runningIndex range;
        static const runningIndex all;
    }
#endif //HPLFE_TENSORRUNNINGINDEX_H
