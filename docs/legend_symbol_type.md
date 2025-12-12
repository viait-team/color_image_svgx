# Implementation Plan: Legend Symbol Type Association

This plan outlines the steps to enhance `SVGXLineChartRendering.js` to accurately associate legend items with chart elements by distinguishing between line and marker symbols.

## 1. Identify Legend Symbol Type (Line vs. Marker)

**Location**: `_findLegendItems` method in `SVGXLineChartRendering.js`.

**Goal**: Determine if a discovered legend symbol is a "Line" or a "Marker".

**Strategy**:
1.  Iterate through the `items` found in `_findLegendItems`.
2.  For each `item.symbolElement`:
    *   Get the Bounding Box (`bbox`).
    *   Calculate **Aspect Ratio**: `width / height`.
    *   Calculate **Size**: `width` and `height`.
3.  **Classification Logic**:
    *   **Marker**: 
        *   Typically small size (e.g., `< 20px` in both dimensions).
        *   Aspect ratio close to 1.0 (e.g., `0.5 < ratio < 2.0`).
        *   Often a closed path or a specific shape (circle, square, diamond).
    *   **Line**:
        *   Can be wider (if it's a segment).
        *   Can be flat (small height, large width).
        *   Or simply "Not a Marker".
4.  **Storage**: Add a `lc_legend_type` property to the legend item symbol: `lc_legend_type: 'line' | 'marker'`.

--- 

## 2. Refined Association Logic

**Location**: `_associateLinesWithLegend` method in `SVGXLineChartRendering.js`.

**Overview**: Replace the current simple color-based loop with a multi-stage process.

### Step 2.1: Group Legend Items
*   Separate `legendItems` into two collections:
    *   `legendMarkers`: items where `lc_legend_type === 'marker'`.
    *   `legendLines`: items where `lc_legend_type === 'line'`.

### Step 2.2: Case Handling

#### Case A: Single Marker Type (`legendMarkers.length === 1`)

1.  **Filter Markers**:
    *   Iterate through all `dataLines` (candidate paths).
    *   Identify paths that match the **geometry** of a marker (small size, aspect ratio ~1).
    *   Refine match by **Color**: Check if path color matches the single legend marker's color.
    *   If match found: associate path with `legendMarkers[0]`.

2.  **Match Lines (Unassociated Paths)**:
    *   Iterate through remaining unassociated paths.
    *   Match against `legendLines` based on **Color** (using existing `_getColorDistance`).

#### Case B: Multi-Marker Types (`legendMarkers.length > 1`)

1.  **Iterate through all paths** (`dataLines`).

2.  **Step 3.1: Identify Path Type**:
    *   Classify the current path (`p`) as a potential **Marker** or **Line/other** based on bounding box dimensions.
    *   *Thresholds*: Define max width/height for a marker (e.g., 30px).

3.  **Step 3.2: Marker Scoring (Paper.js)**:
    *   If path is a **Marker Candidate**:
        *   Compare it against the symbol of *each* item in `legendMarkers`.
        *   **Scoring Function**:
            *   **Normalization**: Scale both the path and the legend symbol to a standard size (e.g., 20x20) and position (center at 0,0).
            *   **Shape Comparison (Paper.js)**:
                *   Create `paper.Path` objects for both.
                *   Use `path.getArea()` to compare areas.
                *   Use boolean operations (if reliable) or sample points to check overlap/distance.
                *   *Alternative (Simpler)*: Rasterize both to small canvas grids and compare pixel overlap (IoU - Intersection over Union).
            *   **Color Match**: Calculate color distance.
            *   **Final Score**: Weighted combination of Shape Similarity and Color Similarity.
        *   **Assignment**: Assign path to the legend marker with the highest score (if above a confidence threshold).

4.  **Step 3.3: Match Lines**:
    *   After marker processing is complete.
    *   Iterate through all remaining unassociated paths.
    *   Match against `legendLines` based on **Color** logic.

---

## 3. Implementation Details for Scoring (Step 3.2)

To allow robust shape matching without heavy overhead:

1.  **Paper.js Integration**:
    *   Ensure `paper` is initialized on a hidden canvas.
    *   Import/Create paths from the SVG path data (`d` attribute).
2.  **Robust Scoring Function**:
    ```javascript
    function calculateMarkerScore(candidatePath, legendSymbol) {
        // 1. Color Score (0-1)
        const colorDist = _getColorDistance(candidatePath.color, legendSymbol.color);
        const colorScore = Math.max(0, 1 - (colorDist / 100));

        // 2. Shape Score (0-1) using Paper.js attributes
        // - Normalize position (center)
        // - Normalize scale (fit in 1x1 box)
        // - Compare: Area, Perimeter, Convex vs Concave
        
        // Example: Compare Area ratio
        const areaRatio = Math.min(area1, area2) / Math.max(area1, area2);
        
        // Example: Shape sampling (check points)
        // Sample N points on candidate, check distance to nearest point on legend symbol.
        
        return (colorScore * 0.4) + (shapeScore * 0.6);
    }
    ```

## 4. Summary of Changes

1.  **Update `_findLegendItems`** to detect and tag `line` vs `marker`.
2.  **Rewrite `_associateLinesWithLegend`** to implement the multi-stage logic:
    *   Sort legends.
    *   Pass 1: Markers (Single vs Multi logic).
    *   Pass 2: Lines (Color logic).
3.  **Add Helper Functions**:
    *   `_isMarkerCandidate(path)`
    *   `_calculateShapeScore(path1, path2)` (wrapping Paper.js logic)
