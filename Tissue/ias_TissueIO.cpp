#include <fstream>
#include <omp.h>

#include "ias_Tissue.h"

namespace ias
{
    void Tissue::saveVTK(std::string prefix,std::string suffix, double time)
    {
        using namespace std;
        
        #pragma omp parallel for
        for(int i = 0; i < int(_cells.size()); i++)
            _cells[i]->saveVTK(prefix+std::to_string(int(_cells[i]->getGlobField("cellId")))+suffix+".vtu");

        vector<int> cellLabels;
        for(auto c: _cells)
            cellLabels.push_back(int(c->getGlobField("cellId")+0.5));
        
        vector<int> glo_cellLbls(_nCells);
        MPI_Gatherv(cellLabels.data(), cellLabels.size(), MPI_INT, glo_cellLbls.data(), _nCellPart.data(), _offsetPart.data(), MPI_INT, 0, _comm);
        
        
        if (_myPart == 0)
        {
            string vtmname = prefix+suffix + ".vtm";
            ofstream file;
            file.open(vtmname);
            
            file << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\"" << endl;
            file << "byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
            file << "<vtkMultiBlockDataSet>"<< endl;
            for(int i = 0; i < _nCells; i++)
            {
                file << "<DataSet index=\"" << glo_cellLbls[i];
                file << "\" name=\"" << glo_cellLbls[i];
                file << "\" file=\"" << prefix+std::to_string(glo_cellLbls[i])+suffix+".vtu";
                file << "\"></DataSet>" << endl;
            }
            file << "</vtkMultiBlockDataSet>" << endl;
            file << "</VTKFile>" << endl;
        }
        
        MPI_Barrier(_comm);
    }

//    void Tissue::openPVD(std::string suffix)
//    {
//        using namespace std;
//        
//        _pvdname = suffix+".pvd";
//
//        
//        if (_myPart == 0)
//        {
//            ofstream file;
//            file.open(_pvdname);
//            file << "<?xml version=\"1.0\"?>" << endl;
//            file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
//            file << "<Collection>" << endl;
//            file.close();
//        }
//    }
//
//    void Tissue::closePVD()
//    {
//        using namespace std;
//
//        if (_myPart == 0)
//        {
//            ofstream file;
//            file.open(_pvdname,ios_base::app);
//            file << "</Collection>" << endl;
//            file << "</VTKFile>" << endl;
//            file.close();
//        }
//        
//        _pvdname = {};
//    }
//
//    void Tissue::loadVTK(std::string location,std::string filename)
//    {
//        using namespace std;
//        using Teuchos::RCP;
//        using Teuchos::rcp;
//
//        int nCells = 0;
//        
//        vector<string> fileNames;
//        
//        if (_myPart == 0)
//        {
//            string name = location+filename + ".pvtu";
//            ifstream file;
//            string line;
//            file.open(name);
//            
//            if (! file.is_open())
//                throw runtime_error("Could not open file "+name+".");
//            
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            getline (file,line);
//            while(getline (file,line))
//            {
//                
//                if(line == "</PUnstructuredGrid>")
//                    break;
//                else
//                {
//                    auto it1 = std::find(line.begin(),line.end(),'"')+1;
//                    auto it2 = std::find(it1,line.end(),'"');
//                    
//                    string filename(it1,it2);
//                    
//                    fileNames.push_back(filename);
//                    
//                    nCells++;
//                }
//            }
//
//            file.close();
//        }
//        
//        MPI_Bcast(&nCells,1,MPI_INT,0,_comm);
//        
//        fileNames.resize(nCells);
//        for(int i = 0; i < nCells; i++)
//        {
//            int fnamesize = fileNames[i].size();
//            MPI_Bcast(&fnamesize, 1, MPI_INT, 0, MPI_COMM_WORLD);
//            
//            if (_myPart != 0)
//                fileNames[i].resize(fnamesize);
//            
//            MPI_Bcast(const_cast<char*>(fileNames[i].data()), fnamesize, MPI_CHAR, 0, MPI_COMM_WORLD);
//        }
//        
//        
//        
//        _nCells = nCells;
//        
//        int baseLocItem = _nCells / _nParts;
//        int remain_nItem = _nCells - baseLocItem * _nParts;
//        int currOffs = 0;
//        for (int p = 0; p < _nParts; p++)
//        {
//            int p_locnitem = (p < remain_nItem) ? (baseLocItem + 1) : (baseLocItem);
//
//            _nCellPart[p]  = p_locnitem;
//            _offsetPart[p] = currOffs;
//
//            currOffs += p_locnitem;
//        }
//        _offsetPart[_nParts] = _nCells;
//        int loc_nCells = _nCellPart[_myPart];
//        
//        for(int i = 0; i < loc_nCells; i++)
//        {
//            RCP<Cell> cell = rcp(new Cell());
////            cell->loadVTK(prefix+std::to_string(getGlobalIdx(i))+suffix+".vtu");
//            int label = cell->loadVTK(location+fileNames[getGlobalIdx(i)]);
//            _cells.emplace_back(cell);
//            _cellLabels.push_back(label);
//        }
//                
//        MPI_Barrier(_comm);
//    }
}
