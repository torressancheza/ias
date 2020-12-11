#ifndef __vtkvmtkPolyDataSurfaceRemeshing_h
#define __vtkvmtkPolyDataSurfaceRemeshing_h

#include "vtkObject.h"
#include "vtkPolyDataAlgorithm.h"
#include "vtkCellLocator.h"
#include "vtkIdList.h"



class SurfaceRemeshing : public vtkPolyDataAlgorithm
{
    public:

        static SurfaceRemeshing *New();
        vtkTypeMacro(SurfaceRemeshing,vtkPolyDataAlgorithm)
        void PrintSelf(ostream& os, vtkIndent indent) override;

        vtkSetMacro(AspectRatioThreshold,double);
        vtkGetMacro(AspectRatioThreshold,double);

        vtkSetMacro(InternalAngleTolerance,double);
        vtkGetMacro(InternalAngleTolerance,double);

        vtkSetMacro(NormalAngleTolerance,double);
        vtkGetMacro(NormalAngleTolerance,double);

        vtkSetMacro(CollapseAngleThreshold,double);
        vtkGetMacro(CollapseAngleThreshold,double);

        vtkSetMacro(Relaxation,double);
        vtkGetMacro(Relaxation,double);

        vtkSetMacro(TargetArea,double);
        vtkGetMacro(TargetArea,double);

        vtkSetMacro(TargetAreaFactor,double);
        vtkGetMacro(TargetAreaFactor,double);

        vtkSetMacro(TriangleSplitFactor,double);
        vtkGetMacro(TriangleSplitFactor,double);

        vtkSetMacro(MinAreaFactor,double);
        vtkGetMacro(MinAreaFactor,double);

        vtkSetMacro(NumberOfIterations,int);
        vtkGetMacro(NumberOfIterations,int);

        vtkSetMacro(NumberOfConnectivityOptimizationIterations,int);
        vtkGetMacro(NumberOfConnectivityOptimizationIterations,int);

        vtkSetStringMacro(TargetAreaArrayName);
        vtkGetStringMacro(TargetAreaArrayName);

        vtkSetMacro(ElementSizeMode,int);
        vtkGetMacro(ElementSizeMode,int);
        void SetElementSizeModeToTargetArea()
        { this->SetElementSizeMode(TARGET_AREA); }
        void SetElementSizeModeToTargetAreaArray()
        { this->SetElementSizeMode(TARGET_AREA_ARRAY); }

        vtkSetMacro(PreserveBoundaryEdges,int);
        vtkGetMacro(PreserveBoundaryEdges,int);
        vtkBooleanMacro(PreserveBoundaryEdges,int);

        vtkSetStringMacro(CellEntityIdsArrayName);
        vtkGetStringMacro(CellEntityIdsArrayName);
    
        enum
        {
            SUCCESS = 0,
            EDGE_ON_BOUNDARY,
            EDGE_BETWEEN_ENTITIES,
            EDGE_LOCKED,
            NOT_EDGE,
            NON_MANIFOLD,
            NOT_TRIANGLES,
            DEGENERATE_TRIANGLES,
            TRIANGLE_LOCKED
        };

        enum
        {
            DO_CHANGE,
            DO_NOTHING
        };

        enum
        {
            TARGET_AREA,
            TARGET_AREA_ARRAY
        };

        enum
        {
            RELOCATE_SUCCESS,
            RELOCATE_FAILURE
        };

        enum
        {
            INTERNAL_POINT,
            POINT_ON_BOUNDARY,
            NO_NEIGHBORS
        };
      //ETX

    protected:
        SurfaceRemeshing();
        ~SurfaceRemeshing();

        virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *) override;

        void BuildEntityBoundary(vtkSmartPointer<vtkPolyData>& input, vtkSmartPointer<vtkPolyData>& entityBoundary);
    
        int EdgeFlipConnectivityOptimizationIteration();
        int EdgeFlipIteration();
        int EdgeCollapseIteration();
        int TriangleSplitIteration();
        int EdgeSplitIteration();
        int PointRelocationIteration(bool projectToSurface=true);

        int TestFlipEdgeValidity(vtkIdType pt1, vtkIdType pt2, vtkIdType cell1, vtkIdType cell2, vtkIdType pt3, vtkIdType pt4);
        int TestConnectivityFlipEdge(vtkIdType pt1, vtkIdType pt2);
        int TestDelaunayFlipEdge(vtkIdType pt1, vtkIdType pt2);
        int TestAspectRatioCollapseEdge(vtkIdType cellId, vtkIdType& pt1, vtkIdType& pt2);
        int TestTriangleSplit(vtkIdType cellId);
        int TestAreaSplitEdge(vtkIdType cellId, vtkIdType& pt1, vtkIdType& pt2);

        int IsElementExcluded(vtkIdType cellId);
        int GetEdgeCellsAndOppositeEdge(vtkIdType pt1, vtkIdType pt2, vtkIdType& cell1, vtkIdType& cell2, vtkIdType& pt3, vtkIdType& pt4);

        int SplitEdge(vtkIdType pt1, vtkIdType pt2);
        int CollapseEdge(vtkIdType pt1, vtkIdType pt2);
        int FlipEdge(vtkIdType pt1, vtkIdType pt2);

        int SplitTriangle(vtkIdType cellId);
        int CollapseTriangle(vtkIdType cellId);

        int RelocatePoint(vtkIdType pointId, bool projectToSurface);

        int IsPointOnBoundary(vtkIdType pointId);
        int IsPointOnEntityBoundary(vtkIdType pointId);

        int GetNumberOfBoundaryEdges(vtkIdType cellId);

        double ComputeTriangleTargetArea(vtkIdType cellId);

        int FindOneRingNeighbors(vtkIdType pointId, vtkSmartPointer<vtkIdList> neighborIds);

        vtkSmartPointer<vtkPolyData> Mesh;
        vtkSmartPointer<vtkPolyData> InputBoundary;
        vtkSmartPointer<vtkPolyData> InputEntityBoundary;
        vtkSmartPointer<vtkCellLocator> Locator;
        vtkSmartPointer<vtkCellLocator> EntityBoundaryLocator;
        vtkSmartPointer<vtkIntArray> CellEntityIdsArray;
        vtkSmartPointer<vtkDataArray> TargetAreaArray;
        vtkSmartPointer<vtkIdList> ExcludedEntityIds;

        double AspectRatioThreshold;
        double InternalAngleTolerance;
        double NormalAngleTolerance;
        double CollapseAngleThreshold;
        double Relaxation;
        int NumberOfConnectivityOptimizationIterations;
        int NumberOfIterations;

        int PreserveBoundaryEdges;

        int ElementSizeMode;
        double TargetArea;
        double TargetAreaFactor;
        double MinAreaFactor;
        double MaxAreaFactor;
        double TriangleSplitFactor;
        char* TargetAreaArrayName;

        char* CellEntityIdsArrayName;

    private:
        SurfaceRemeshing(const SurfaceRemeshing&);  // Not implemented.
        void operator=(const SurfaceRemeshing&);  // Not implemented.
};

#endif

