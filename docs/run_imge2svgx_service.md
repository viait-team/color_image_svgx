# Image2SVGX Backend Migration: Analysis & Implementation Plan

## 1. Critical Issues Analysis

The current `webview` implementation is deeply coupled to the Browser Runtime Environment. Migrating to a backend service requires solving three fundamental "Vision" and "Geometry" problems.

### 1.1 The "Blindness" of the Backend (DOM Dependency)
The most critical issue is the reliance on the Browser's **Layout Engine**.
*   **Current State**: The code uses `element.getBBox()` and `element.getBoundingClientRect()` to "see" where lines and text are.
*   **The Constraint**: Standard XML parsers (Python's `lxml`, Node's `xmldom`) are "blind". They see the *code* (`<path d="...">`) but do not know the *geometry* (width, height, intersection).
*   **The JSDOM Trap**: Libraries like `jsdom` implement the DOM API (tree structure) but **DO NOT** implement a layout engine. They return `0` for all geometry calls (`getBBox`, `getCTM`).
*   **Required Solution**: We must essentially build or use a **Geometry Engine**. We need a library that parses the `d` attribute and mathematically calculates the bounding box (Min/Max X,Y).

### 1.2 Coordinate Systems & Transforms
The second major issue is the **Stacking of Transformations**.
*   **Current State**: The code uses `getScreenCTM()` (Screen Coordinate Transform Matrix). This powerful browser API automatically multiplies all parent transforms:
    `Total Matrix = Viewport × Parent_Group2 × Parent_Group1 × Element_Matrix`
*   **The Challenge**: In the backend, we lose this automatic aggregation. If a path is inside `<g transform="translate(10,10)">`, its raw coordinates are wrong.
*   **Required Solution**: We must implement a **Transform Stack**.
    *   We must traverse the SVG tree recursively.
    *   Maintain a "Current Transformation Matrix" (CTM).
    *   Apply this matrix to every point in every path to normalize them into a single "Global User Space".

### 1.3 CSS Style Cascading
*   **Current State**: `window.getComputedStyle(element)` is used to find the final color (`fill/stroke`) of lines.
*   **The Challenge**: Styles can be inherited from parents or defined in `<style>` blocks.
*   **Required Solution**: A **Style Resolver** that implements basic CSS inheritance rules (parent -> child) and parses basic CSS strings.

---

## 2. Solution Evaluation & Prototyping

Before committing to a full rewrite, we must evaluate the feasibility of the "Node.js + JSDOM" path versus the "Python + Geometry" path.

### 2.1 Protocol: Evaluating `jsdom` (Node.js)
**Objective**: Determine if we can run existing JS code with minimal changes.
**Step-by-Step Evaluation**:
1.  **Setup**: Initialize a Node.js project with `jsdom`.
2.  **Load**: Load a Potrace-generated SVG string into the JSDOM `window`.
3.  **Test Geometry**:
    ```javascript
    const dom = new JSDOM(svgString);
    const path = dom.window.document.querySelector('path');
    console.log(path.getBBox()); // EXPECTED RESULT: {x:0, y:0, width:0, height:0}
    ```
4.  **Test Polyfills**: Attempt to use `svgdom` (a library that attempts to implement `getBBox` in Node).
    *   *Risk*: These libraries are often incomplete or unmaintained.

### 2.2 Protocol: Evaluating `svgpathtools` (Python)
**Objective**: Verify mathematical accuracy of Python geometry.
**Step-by-Step Evaluation**:
1.  **Parse**: Load SVG using `svgpathtools.svg2paths`.
2.  **BBox**: Call `.bbox()` on a complex curve.
3.  **Compare**: Output values and compare against the `getBBox()` values from the current Webview console.
    *   *Success Metric*: Deviation < 0.1 units.

---

## 3. Migration Execution Plan: File-by-File

Assuming the "Python + Geometry" approach (which is robust), here is the migration map.

### 3.1 `webview/SVGXLineChartAnalyzer.js` -> `backend/chart_analyzer.py`

| Current JS Method | Backend Logic Strategy | Complexity |
| :--- | :--- | :--- |
| `_getFullBBox` | **The Core Utility**. Use `svgpathtools` to get raw bbox. Apply `transform` attributes if present. | High |
| `_matchYAxisLeftLabels` | Calculate Euclidean distance between `GridLine.end_point` and `Label.bbox.center`. | Medium |
| `_findYAxisGridlines` | Filter implementation: `path.bbox.width > (viewbox.width * 0.7)`. | Low |
| `_extractPathColor` | Check `fill/stroke` attr. If None, check parent `g` attr. (Simplified Style Resolver). | Medium |

### 3.2 `webview/SVGXLineChartDataExtractor.js` -> `backend/data_extractor.py`

| Current JS Method | Backend Logic Strategy | Complexity |
| :--- | :--- | :--- |
| `_toLogicalX/Y` | Pure Math. `(val - min) / range`. Direct port. | Low |
| `extractLogicalData` | Iterate `svgpathtools` Path objects. Sample points using `path.point(t)`. | Medium |
| `_extractPointsFromPath` | **Thickness Logic**: In JS, we used line width. In Backend, we might need to rely on the `stroke-width` attribute or just sampling logic. | High |

### 3.3 `webview/SVGXTsvVisual.js` -> `backend/ocr_processor.py`

| Current JS Method | Backend Logic Strategy | Complexity |
| :--- | :--- | :--- |
| `getStructuredOcrData` | Map TSV rows to `TextNode` objects. Group by Y-coordinate (Row Detection). | Medium |
| `fixCsvTableData...` | Pure logic. Direct copy-paste of the algorithm. | Low |

---

## 4. Implementation Steps

1.  **Environment Setup**: Install `svgpathtools`, `numpy`.
2.  **Geometry Layer**: Implement the `SVGNode` wrapper class that handles:
    *   Parsing `d` path data.
    *   Applying `transform="matrix(...)"` to coordinates.
    *   Calculating BBox.
3.  **Porting Logic**: Systematically port `Analyzer` -> `Extractor` -> `Renderer` classes.
4.  **Integration**: Update `color_trace_multi.py` to call the new backend module instead of just outputting the raw SVG.
