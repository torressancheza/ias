#include <numeric>

#include <vtkSmartPointer.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkCleanPolyData.h>


#include "ias_LinearElements.h"
#include "ias_LoopSubdivision.h"
#include "ias_TissueGen.h"

namespace ias
{
    Teuchos::RCP<Tissue> TissueGen::genRegularGridSpheres(int nx, int ny, int nz, double deltax, double deltay, double deltaz, double r, int nSubdiv)
    {
        using namespace std;
        using namespace Tensor;
        using Teuchos::RCP;
        using Teuchos::rcp;
        
        RCP<Tissue> tissue = rcp(new Tissue);

        tissue->_nCells = nx * ny * nz;

        int loc_nCells{};
        int baseLocItem = tissue->_nCells / tissue->_nParts;
        int remain_nItem = tissue->_nCells - baseLocItem * tissue->_nParts;
        int currOffs = 0;
        for (int p = 0; p < tissue->_nParts; p++)
        {
            int p_locnitem = (p < remain_nItem) ? (baseLocItem + 1) : (baseLocItem);

            tissue->_nCellPart[p]  = p_locnitem;
            tissue->_offsetPart[p] = currOffs;

            currOffs += p_locnitem;
        }
        tissue->_offsetPart[tissue->_nParts] = tissue->_nCells;
        loc_nCells = tissue->_nCellPart[tissue->getMyPart()];

        RCP<Cell> cell = rcp(new Cell);
        cell->generateSphereFromOctahedron(nSubdiv, r);
        cell->setBasisFunctionType(_bfType);
        for(auto f: _nodeFieldNames)
            cell->addNodeField(f);
        for(auto f: _globFieldNames)
            cell->addGlobField(f);
        cell->Update();
                                
        for(int n = 0; n < loc_nCells; n++)
        {
            int glo_n = tissue->getGlobalIdx(n);
            int k =  glo_n / (nx*ny);
            int j = (glo_n-k*nx*ny)/nx;
            int i =  glo_n-(k*ny+j)*nx;

            RCP<Cell> newcell = rcp(new Cell);
            newcell->_connec = cell->_connec;
            newcell->_nodeFields = cell->_nodeFields;
            newcell->_globFields = cell->_globFields;
            newcell->_bfType = _bfType;
            newcell->_nodeFieldNames = cell->_nodeFieldNames;
            newcell->_globFieldNames = cell->_globFieldNames;
            newcell->Update();
            
            newcell->getGlobField("cellId") = glo_n;

            newcell->_nodeFields(all,0) += i * deltax;
            newcell->_nodeFields(all,1) += j * deltay;
            newcell->_nodeFields(all,2) += k * deltaz;

            tissue->_cells.emplace_back(newcell);
        }
        
        tissue->_tissFields.resize(_tissFieldNames.size());
        tissue->_tissFieldNames = _tissFieldNames;
        
        tissue->Update();
        
        return tissue;
    }
}
    
