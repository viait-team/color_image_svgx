# Walkthrough - Legend Symbol Association Implementation

I have implemented the improved legend symbol association logic in `SVGXLineChartRendering.js`.

## Changes

1.  **Legend Item Identification (`_findLegendItems`)**
    *   Added logic to determine if a legend symbol is a `marker` or a `line` based on its size and aspect ratio.
    *   Markers are identified if `width < 30`, `height < 30`, and `aspectRatio` is between 0.5 and 2.0.
    *   The type is stored in `lc_legend_type` property of the legend item and set as an attribute on the symbol element.

2.  **Association Logic (`_associateLinesWithLegend`)**
    *   Refactored to handle "Markers" and "Lines" separately.
    *   **Step 1: Marker Association**
        *   Filters `dataLines` to find marker candidates (`_isMarkerCandidate`).
        *   If single legend marker: Matches based on geometry and color specific to that marker.
        *   If multiple legend markers: Uses `_calculateMarkerScore` to find the best match based on Color + Shape.
    *   **Step 2: Line Association**
        *   Iterates through remaining unassociated paths.
        *   Matches against `legendLines` based on Color distance (backward compatibility with original logic).

3.  **Shape Scoring (`_calculateMarkerScore`)**
    *   Uses **Paper.js** (`paper.Path`) to normalize and compare shapes.
    *   Compares **Area Ratio** and **Perimeter (Length) Ratio** after normalizing scale and position.
    *   Combines shape score (60%) and color score (40%) for a robust match.

## Verification

### Manual Code Review
*   Checked that `paper.setup` is called (`_initializePaper`) before using Paper.js features.
*   Verified that `lc_legend_type` is correctly assigned.
*   Verified that the fallback to simple color matching exists for lines.

### Dependencies
*   Verified `webview.html` imports `paper-full.min.js`.

## files
- `webview/SVGXLineChartRendering.js`
- `docs/legend_symbol_type.md`
