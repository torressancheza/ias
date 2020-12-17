//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <fstream>
#include <thread>

#include "ias_Tissue.h"

namespace ias
{
    void Tissue::saveVTK(std::string prefix,std::string suffix)
    {
        using namespace std;
        
        #pragma omp parallel for
        for(int i = 0; i < int(_cells.size()); i++)
            _cells[i]->saveVTK(prefix+std::to_string(int(_cells[i]->getCellField("cellId")))+suffix+".vtu");

        vector<int> cellLabels;
        for(auto c: _cells)
            cellLabels.push_back(int(c->getCellField("cellId")+0.5));
        
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
                file << "<DataSet index=\"" << glo_cellLbls[i] << "\" ";
                file << "name=\"" << glo_cellLbls[i] << "\" ";
                file << "file=\"" << prefix+std::to_string(glo_cellLbls[i])+suffix+".vtu";
                file << "\"></DataSet>" << endl;
            }
            file << "</vtkMultiBlockDataSet>" << endl;
            file <<"<FieldData>" << endl;
            for(int i = 0; i < _tissFields.size(); i++)
            {
                file <<"<DataArray type=\"Float64\" Name=\"" << _tissFieldNames[i] <<"\" NumberOfTuples=\"1\" format=\"ascii\" RangeMin=\"" << _tissFields(i) << "\" RangeMax=\"" << _tissFields(i) << "\">" << endl;
                file << _tissFields(i) << endl;
                file << "</DataArray>" << endl;
            }
            file << "</FieldData>" << endl;
            file << "</VTKFile>" << endl;
            file.close();
        }
        
        MPI_Barrier(_comm);
    }

    void Tissue::loadVTK(std::string location,std::string filename, BasisFunctionType bfType)
    {
        using namespace std;
        using Teuchos::RCP;
        using Teuchos::rcp;

        int nCells = 0;
        
        vector<string> fileNames;
        vector<double> fieldValues;
        
        if (_myPart == 0)
        {
            string name = location+filename + ".vtm";
            ifstream file;
            string line;
            file.open(name);
            
            if (! file.is_open())
                throw runtime_error("Could not open file "+name+".");
            
            
            string errormsg = "Tissue::loadVTK: the file " + name + " is not in a format accepted by ias.";
            //Read first line
            getline(file,line);
            if(line != "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\"")
                throw runtime_error(errormsg);
            getline(file,line);
            if(line != "byte_order=\"LittleEndian\" header_type=\"UInt64\">")
                throw runtime_error(errormsg);
            getline(file,line);
            if(line != "<vtkMultiBlockDataSet>")
                throw runtime_error(errormsg);

            while(getline (file,line))
            {
                if(line.find("<DataSet", 0) == 0)
                {
                    size_t first = line.find("file=\"");
                    size_t last = line.find("\"><");
                    string file = line.substr(first+6,last-(first+6));
                    
                    fileNames.push_back(file);
                    
                    nCells++;
                }
                else
                {
                    break;
                }
            }
            if(line != "</vtkMultiBlockDataSet>")
                throw runtime_error(errormsg);
            
            getline (file,line);
            if(line != "<FieldData>")
                throw runtime_error(errormsg);
            
            while(getline (file,line))
            {
                if(line.find("<DataArray", 0) == 0)
                {
                    size_t first = line.find("Name=\"");
                    size_t last = line.find("\" NumberOfTuples");
                    string name = line.substr(first+6,last-(first+6));
                    
                    _tissFieldNames.push_back(name);
                    
                    getline (file,line);
                    double value{};
                    std::istringstream s( line );
                    s >> value;
                    fieldValues.push_back(value);
                    
                    getline (file,line);
                    if(line != "</DataArray>")
                        throw runtime_error(errormsg);
                }
                else
                {
                    break;
                }
            }
            
            file.close();
        }
        
        MPI_Bcast(&nCells, 1, MPI_INT, 0, _comm);

        int nTissFields{};
        nTissFields = _tissFieldNames.size();
        MPI_Bcast(&nTissFields, 1, MPI_INT, 0, _comm);
        
        string tissFieldNames{};
        vector<char> v_tissFieldNames;
        if(_myPart == 0)
        {
            for(auto s: _tissFieldNames)
                tissFieldNames += s + ",";
            v_tissFieldNames = vector<char>(tissFieldNames.begin(),tissFieldNames.end());
        }
        
        int sizeTissFieldNames = v_tissFieldNames.size();
        MPI_Bcast(&sizeTissFieldNames, 1, MPI_INT, 0, _comm);
        v_tissFieldNames.resize(sizeTissFieldNames);
        
        MPI_Bcast(v_tissFieldNames.data(), sizeTissFieldNames, MPI_CHAR, 0, _comm);

        tissFieldNames = string(v_tissFieldNames.begin(),v_tissFieldNames.end());
        
        _tissFieldNames.clear();
        std::istringstream ss(tissFieldNames);
        std::string token;
        while(std::getline(ss, token, ','))
            _tissFieldNames.push_back(token);
                
        fieldValues.resize(nTissFields);
        MPI_Bcast(fieldValues.data(), nTissFields, MPI_DOUBLE, 0, _comm);

        _tissFields.resize(nTissFields);
        for(int i = 0; i < nTissFields; i++)
            _tissFields(i) = fieldValues[i];
        
        fileNames.resize(nCells);
        for(int i = 0; i < nCells; i++)
        {
            int fnamesize = fileNames[i].size();
            MPI_Bcast(&fnamesize, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (_myPart != 0)
                fileNames[i].resize(fnamesize);

            MPI_Bcast(const_cast<char*>(fileNames[i].data()), fnamesize, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        _nCells = nCells;

        int baseLocItem = _nCells / _nParts;
        int remain_nItem = _nCells - baseLocItem * _nParts;
        int currOffs = 0;
        for (int p = 0; p < _nParts; p++)
        {
            int p_locnitem = (p < remain_nItem) ? (baseLocItem + 1) : (baseLocItem);

            _nCellPart[p]  = p_locnitem;
            _offsetPart[p] = currOffs;

            currOffs += p_locnitem;
        }
        _offsetPart[_nParts] = _nCells;
        int loc_nCells = _nCellPart[_myPart];

        for(int i = 0; i < loc_nCells; i++)
        {
            RCP<Cell> cell = rcp(new Cell());
            cell->loadVTK(location+fileNames[getGlobalIdx(i)]);
            cell->setBasisFunctionType(bfType);
            cell->Update();
            _cells.emplace_back(cell);
        }
                
        MPI_Barrier(_comm);
    }
}
