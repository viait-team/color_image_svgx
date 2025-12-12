# Walkthrough - Chart Data Extraction and Redraw

I have implemented the logic to extract logical data from the chart and redraw it using D3.js.

## implemented Features

1.  **Data Extraction (`extractLogicalData`)**
    *   Iterates through legend items to identify series (`line` or `marker`).
    *   Finds associated paths using the `lc_legend_ref` attribute (established in previous steps).
    *   Parses path data (`d` attribute) to extract visual coordinates.
    *   Converts visual coordinates to logical coordinates using `xlm` and `ylm` attributes (parsed from JSON).
    *   Returns a structured array of series data: `[{ id, name, type, style, data: [{x, y}, ...] }]`.

2.  **Style Extraction (`extractTraceStyles`)**
    *   Captures `stroke`, `stroke-width`, `stroke-dasharray`, `stroke-opacity`, and `fill` from the original SVG elements.

3.  **Chart Redrawing (`redrawChart`)**
    *   Uses **D3.js** to render a new chart in a specified container (`#new-svg-container`).
    *   Sets up linear scales (`x`, `y`) based on the extent of the extracted logical data.
    *   Draws axes.
    *   Renders data series matching the extracted styles (lines and markers).

4.  **Integration**
    *   Updated `webview.html` to trigger extraction and redraw when the "Plot" button is clicked for line charts.

## Files Modified
*   `webview/SVGXLineChartRendering.js`: Added extraction and redraw methods.
*   `webview/webview.html`: Added integration code.

## Verification
*   **Code Review**: Verified method structure and integration points.
*   **Logic Check**: Confirmed that `xlm`/`ylm` mapping logic applies the linear transformation correctly.
