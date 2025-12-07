# Line Chart Rendering Implementation Plan

## Overview
Create a D3.js-based line chart renderer (`SVGXLineChartRendering.js`) that transforms OCR-detected text and traced SVG paths into interactive, data-driven line charts with proper logical mappings.

## Goals
1. Extract logical data (x/y values, legends) from OCR TSV and SVG paths
2. Redraw line chart using D3.js with accurate data representation
3. Encode logical mappings (xlm, ylm) in SVG attributes
4. Support legend identification (lc_legend_id, lc_legend_ref)
5. Preserve visual styles (colors, line patterns) from original chart

---

## Architecture

### Input Sources
- **TSV File**: OCR text data (axis labels, ticks, legends, title)
- **SVG Paths**: Visual line traces from `potrace`
- **lctype Attribute**: Identifies chart as "line" vs "table"

### Output
- Redrawn SVG with:
  - D3-generated line charts with logical data
  - `xlm`/`ylm` attributes for axis mapping
  - `lc_legend_id` for legend items
  - `lc_legend_ref` linking lines to legends
  - Interactive hover/click behaviors

---

## Step-by-Step Implementation

### Phase 0: Extract Logical Mapping from Existing SVG ⭐ **CRITICAL PREREQUISITE**

**This is the fundamental difference between line charts and data tables.**

For line charts, we must **first establish the logical coordinate system** by analyzing the existing traced SVG and matching visual elements (axes, ticks, gridlines) with OCR text labels. This creates the transformation mapping needed to convert SVG path coordinates into logical data values.

---

#### Step 0.1: Gather Logical Mapping Code & Documentation
**Source**: Multiple existing projects
**Destination**: `/webview/logical_mapping/`

**Tasks**:
- [ ] Collect all logical mapping implementations from other projects
- [ ] Gather documentation on tick/gridline matching algorithms
- [ ] Create unified API for logical mapping extraction
- [ ] Document successful case studies

**Files to gather**:
```
webview/logical_mapping/
├── README.md                    # Overview and usage guide
├── LogicalMappingExtractor.js   # Main implementation
├── TickMatcher.js               # Tick/label matching algorithm
├── GridlineDetector.js          # Gridline detection
├── CoordinateTransformer.js     # SVG ↔ Logical conversion
└── examples/                    # Working examples
    ├── case1_simple_chart.md
    ├── case2_complex_chart.md
    └── ...
```

---

#### Step 0.2: Analyze Existing SVG Structure for Line Charts
**Methods**:
```javascript
analyzeAxisElements(svg)
detectGridlines(svg)
extractTickMarks(svg)
```

**Tasks**:
- [ ] Identify axis lines (horizontal/vertical lines at boundaries)
- [ ] Detect gridlines (parallel lines at regular intervals)
- [ ] Extract tick marks (short perpendicular lines on axes)
- [ ] Determine chart boundaries (min/max x, min/max y in SVG coordinates)

**Algorithm Overview**:
```javascript
// 1. Find axis lines
const axes = findAxisLines(svg); // { xAxis, yAxis }

// 2. Detect gridlines or tick marks
const gridlines = detectGridlines(svg); // { horizontal: [], vertical: [] }
const ticks = extractTickMarks(svg); // { xTicks: [], yTicks: [] }

// 3. Extract boundaries
const bounds = {
  xMin: Math.min(...gridlines.vertical.map(g => g.x)),
  xMax: Math.max(...gridlines.vertical.map(g => g.x)),
  yMin: Math.min(...gridlines.horizontal.map(g => g.y)),
  yMax: Math.max(...gridlines.horizontal.map(g => g.y))
};
```

---

#### Step 0.3: Match Visual Elements to OCR Tick Labels
**Methods**:
```javascript
matchTicksToLabels(visualTicks, ocrLabels, tolerance)
buildLogicalMapping(matchedPairs)
```

**Tasks**:
- [ ] Get OCR tick labels from TSV (xticks: ["0", "20", "40", ...])
- [ ] Match each visual tick/gridline position to nearest OCR label
- [ ] Use proximity threshold (e.g., within 20 pixels)
- [ ] Validate matches (monotonic ordering, reasonable spacing)

**Matching Algorithm**:
```javascript
// Input:
// - visualTicks: [{x: 100}, {x: 250}, {x: 400}, ...]  (SVG coordinates)
// - ocrLabels: ["0", "20", "40", "60", ...]            (logical values)

function matchTicksToLabels(visualTicks, ocrLabels, tolerance = 20) {
  const matches = [];
  
  for (let i = 0; i < visualTicks.length; i++) {
    const visualPos = visualTicks[i].x; // or .y for vertical
    
    // Find OCR label near this position
    const nearbyLabel = findNearbyOcrLabel(visualPos, ocrLabels, tolerance);
    
    if (nearbyLabel) {
      matches.push({
        svgCoord: visualPos,
        logicalValue: parseFloat(nearbyLabel.text),
        confidence: calculateConfidence(nearbyLabel)
      });
    }
  }
  
  // Validate: should be monotonic and evenly spaced
  validateMatches(matches);
  
  return matches;
}
```

**Output**: Matched pairs
```javascript
[
  { svgCoord: 100, logicalValue: 0 },
  { svgCoord: 250, logicalValue: 20 },
  { svgCoord: 400, logicalValue: 40 },
  ...
]
```

---

#### Step 0.4: Create Coordinate Transformation Functions
**Methods**:
```javascript
createScaleFunction(matchedPairs, axis)
svgToLogical(svgX, svgY)
logicalToSvg(logicalX, logicalY)
```

**Tasks**:
- [ ] Build linear interpolation from matched pairs
- [ ] Create `svgToLogical(x, y)` transformation function
- [ ] Create `logicalToSvg(x, y)` inverse transformation
- [ ] Handle edge cases (extrapolation beyond tick range)

**Scale Creation**:
```javascript
// From matched pairs, create linear scale
function createScaleFunction(matchedPairs, axis = 'x') {
  // Use D3 scale or custom linear interpolation
  const svgCoords = matchedPairs.map(p => p.svgCoord);
  const logicalValues = matchedPairs.map(p => p.logicalValue);
  
  // D3 scale approach
  const scale = d3.scaleLinear()
    .domain([Math.min(...logicalValues), Math.max(...logicalValues)])
    .range([Math.min(...svgCoords), Math.max(...svgCoords)]);
  
  return {
    toSvg: (logicalVal) => scale(logicalVal),
    toLogical: (svgCoord) => scale.invert(svgCoord)
  };
}

// Usage:
const xScale = createScaleFunction(xMatchedPairs, 'x');
const yScale = createScaleFunction(yMatchedPairs, 'y');

function svgToLogical(svgX, svgY) {
  return {
    x: xScale.toLogical(svgX),
    y: yScale.toLogical(svgY)
  };
}
```

---

#### Step 0.5: Validate Logical Mapping
**Methods**:
```javascript
validateMapping(xScale, yScale, testPoints)
```

**Tasks**:
- [ ] Test transformation with known tick positions
- [ ] Verify round-trip: `logicalToSvg(svgToLogical(x, y)) ≈ (x, y)`
- [ ] Check monotonicity and linearity
- [ ] Log mapping statistics (min, max, scale factor)

**Validation Example**:
```javascript
// Test with known tick at SVG x=250, logical value=20
const logical = xScale.toLogical(250);
console.assert(Math.abs(logical - 20) < 0.01, "Mapping error!");

// Test round-trip
const svgCoord = xScale.toSvg(20);
const backToLogical = xScale.toLogical(svgCoord);
console.assert(Math.abs(backToLogical - 20) < 0.01, "Round-trip error!");
```

---

#### Step 0.6: Store Logical Mapping Metadata
**Format**:
```javascript
const logicalMapping = {
  xAxis: {
    domain: [0, 100],           // Logical min/max
    range: [80, 720],           // SVG pixel min/max
    ticks: [0, 20, 40, 60, 80, 100],
    matchedPairs: [
      { svgCoord: 80, logicalValue: 0 },
      { svgCoord: 208, logicalValue: 20 },
      ...
    ]
  },
  yAxis: {
    domain: [0, 10],
    range: [540, 60],           // Note: SVG y increases downward
    ticks: [0, 2, 4, 6, 8, 10],
    matchedPairs: [...]
  },
  metadata: {
    chartType: 'line',
    extractionMethod: 'gridline_matching',
    confidence: 0.95,
    timestamp: Date.now()
  }
};
```

**Tasks**:
- [ ] Serialize mapping as JSON for debugging
- [ ] Add to SVG as `data-logical-mapping` attribute (optional)
- [ ] Pass to subsequent phases

---

### Phase 1: Data Extraction & Processing (AFTER Phase 0)

#### Step 1.1: Parse OCR TSV Data
**File**: `SVGXLineChartRendering.js`
**Methods**:
```javascript
constructor(svgElement)
getStructuredOcrData(tsvContent)
```

**Tasks**:
- [ ] Extract axis labels (xlabel, ylabel)
- [ ] Extract axis ticks (xticks, yticks) and convert to numeric arrays
- [ ] Extract chart title
- [ ] Extract legend text (use groupWordsByTop() pattern from DataTable)
- [ ] Identify vertical text (y-axis labels) using TSV hierarchy

**Pattern**: Follow `SVGXDataTableRendering.getStructuredOcrData()`

---

#### Step 1.2: Extract SVG Path Data
**Methods**:
```javascript
extractPathsFromSVG()
pathToDataPoints(pathElement)
```

**Tasks**:
- [ ] Query all `<path>` elements with `fill="none"` or `stroke!=none`
- [ ] Filter out axis lines (horizontal/vertical at boundaries)
- [ ] Extract path `d` attribute
- [ ] Parse SVG path commands (M, L, C, Q) into (x,y) coordinates
- [ ] Use `SVGXPathPoints.js` for curve-to-point conversion
- [ ] Group paths by color/stroke attributes (each = one line series)

**Dependencies**: 
- `SVGXPathPoints.js` - for path parsing
- `SVGXPathSegments.js` - for segment analysis

---

#### Step 1.3: Map Visual Styles to Line Series
**Methods**:
```javascript
getLineStyles()
matchPathsToLegends(paths, legends)
```

**Tasks**:
- [ ] Extract stroke color, width, dash-array for each path
- [ ] Create style objects: `{ color, width, pattern, opacity }`
- [ ] Match path positions/colors to legend text
- [ ] Generate `lc_legend_id` from legend text (use `_generateLcId()`)
- [ ] Assign `lc_legend_ref` to each path group

**Algorithm**:
```
For each legend item:
  1. Get text and position
  2. Find nearby path with matching color
  3. Link: path.lc_legend_ref = legend.lc_legend_id
```

---

### Phase 2: Coordinate Transformation

#### Step 2.1: Establish Logical Coordinate System
**Methods**:
```javascript
calculateLogicalDomains(xticks, yticks)
createScales(xDomain, yDomain, svgWidth, svgHeight)
```

**Tasks**:
- [ ] Determine x-domain from xticks: `[min, max]`
- [ ] Determine y-domain from yticks: `[min, max]`
- [ ] Create D3 scales:
  - `xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerWidth])`
  - `yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerHeight, 0])`
- [ ] Account for margins (top, right, bottom, left)

---

#### Step 2.2: Transform SVG Paths to Logical Data
**Methods**:
```javascript
transformPathToLogicalData(pathPoints, xScale, yScale)
```

**Tasks**:
- [ ] For each (x,y) in path points:
  - `logicalX = xScale.invert(x - marginLeft)`
  - `logicalY = yScale.invert(y - marginTop)`
- [ ] Store as data arrays: `[[x1,y1], [x2,y2], ...]`
- [ ] Validate data ranges match axis ticks

---

### Phase 3: D3 Chart Rendering

#### Step 3.1: Setup SVG Canvas
**Methods**:
```javascript
setupSVGCanvas(width, height)
drawAxes(g, xScale, yScale, xticks, yticks)
```

**Tasks**:
- [ ] Clear existing SVG content
- [ ] Set viewBox and dimensions
- [ ] Create chart group `<g>` with margins
- [ ] Draw X-axis with D3: `d3.axisBottom(xScale)`
- [ ] Draw Y-axis with D3: `d3.axisLeft(yScale)`
- [ ] Add axis labels (xlabel, ylabel)

**Pattern**: Similar to `SVGXDataTableRendering._setupSVGCanvas()`

---

#### Step 3.2: Draw Line Series
**Methods**:
```javascript
drawLines(g, lineData, xScale, yScale, styles)
```

**Tasks**:
- [ ] Use D3 line generator:
  ```javascript
  const line = d3.line()
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]))
    .curve(d3.curveMonotoneX); // or curveLinear
  ```
- [ ] For each line series:
  - Create `<path>` element
  - Set `d` attribute: `line(dataPoints)`
  - Apply styles: `stroke`, `stroke-width`, `stroke-dasharray`
  - Add `lc_legend_ref` attribute
  - Add `fill="none"`

---

#### Step 3.3: Add Logical Mapping Attributes
**Methods**:
```javascript
addGlobalLogicalMapping(svg, xDomain, yDomain, width, height, margin)
addLineLogicalMapping(pathElement, legendId, dataArray)
```

**Tasks**:
- [ ] Add to root `<svg>`:
  ```xml
  xlm="minX,maxX,marginLeft,innerWidth"
  ylm="minY,maxY,marginTop,innerHeight"
  ```
- [ ] Add to each `<path>`:
  ```xml
  lc_legend_ref="legend_id_80_ci"
  data-logical="[[x1,y1],[x2,y2],...]" (optional, for debugging)
  ```

**Pattern**: Follow `SVGXDataTableRendering._addGlobalLogicalMapping()`

---

#### Step 3.4: Draw Legends
**Methods**:
```javascript
drawLegends(svg, legends, styles, width)
```

**Tasks**:
- [ ] Position legends (top-right or custom)
- [ ] For each legend:
  - Draw line sample with style
  - Add text label
  - Add `lc_legend_id` attribute: `lc_legend_id="80_ci"`
  - Create clickable `<g>` group
- [ ] Add interactivity:
  - Hover: highlight corresponding line
  - Click: toggle line visibility

**Pattern**: Adapt from `SVGXDataTableRendering._drawLegends()`

---

### Phase 4: Interactivity & Polish

#### Step 4.1: Add Interactive Features
**Methods**:
```javascript
addLineInteractions(pathElements, legends)
```

**Tasks**:
- [ ] Hover over line → highlight + show tooltip with legend name
- [ ] Hover over legend → highlight corresponding line
- [ ] Click legend → toggle line visibility
- [ ] Use D3 events: `.on('mouseover', ...).on('click', ...)`

---

#### Step 4.2: Add Title & Comments
**Methods**:
```javascript
drawTitle(svg, title, width)
drawComments(svg, comments)
```

**Tasks**:
- [ ] Position title at top-center
- [ ] Add comments block (top-left)
- [ ] Format text with appropriate styles

**Pattern**: Use `SVGXDataTableRendering._drawTitle()` and `_drawComments()`

---

### Phase 5: Integration & Testing

#### Step 5.1: Integrate with wrbview.html
**File**: `webview/wrbview.html`

**Tasks**:
- [ ] Add `<script src="SVGXLineChartRendering.js"></script>` after SVGXDataTableRendering.js
- [ ] Modify plotBtn click handler to branch on `lctype` attribute
- [ ] Instantiate appropriate renderer based on lctype

**Implementation in wrbview.html** (lines ~295-350):

```javascript
plotBtn.addEventListener('click', (event) => {
  const svg = svgContainer.querySelector('svg');
  if (!svg) {
    console.warn('[WARN] No SVG loaded');
    return;
  }

  // Check lctype attribute to determine chart type
  const lctype = svg.getAttribute('lctype') || 'table'; // default to 'table'
  console.log(`[LOG] Chart type detected: ${lctype}`);

  if (lctype === 'table') {
    // ===== EXISTING TABLE RENDERING =====
    const rendering = new SVGXDataTableRendering(svg);
    const styles = rendering.getStyleFromTableSvg();
    
    const rowTolerance = 10;
    const exportComments = true;
    const exportTitle = true;
    const exportXAxisLabel = true;
    const exportXTicks = true;
    const exportYAxisLabel = true;
    const exportYTicks = true;
    const exportTableCsv = true;

    const data = rendering.getStructuredOcrData(correctedTsvContent,
      rowTolerance, exportComments, exportTitle, exportXAxisLabel,
      exportXTicks, exportYAxisLabel, exportYTicks, exportTableCsv
    );
    
    const tableCsv = data.tableCsv;
    const comments = data.comments;
    const title = data.title;
    const x_label = data.x_axis_label;
    const x_ticks = data.x_ticks;
    const y_label = data.y_axis_label;
    const y_ticks = data.y_ticks;

    rendering.drawSVGChartWithStyles(tableCsv, title, x_label, x_ticks, 
                                     y_label, y_ticks, styles, comments);

    if (event.ctrlKey) {
      downloadTableSvg();
    }
    if (event.shiftKey) {
      downloadCsv(tableCsv, 'plot_data.csv');
    }
    
  } else if (lctype === 'line') {
    // ===== NEW LINE CHART RENDERING =====
    const rendering = new SVGXLineChartRendering(svg);
    
    // Extract data from TSV and SVG paths
    const data = rendering.getStructuredOcrData(correctedTsvContent, {
      rowTolerance: 10,
      exportComments: true,
      exportTitle: true,
      exportXAxisLabel: true,
      exportXTicks: true,
      exportYAxisLabel: true,
      exportYTicks: true,
      exportLegends: true
    });
    
    // Render the line chart
    rendering.renderLineChart(data, {
      width: 800,
      height: 600,
      margin: { top: 60, right: 150, bottom: 60, left: 80 }
    });
    
    if (event.ctrlKey) {
      downloadTableSvg(); // Download redrawn SVG
    }
    if (event.shiftKey) {
      // Download line data as CSV
      const lineDataCsv = rendering.exportLineDataAsCsv();
      downloadCsv(lineDataCsv, 'line_data.csv');
    }
    
  } else {
    console.warn(`[WARN] Unknown lctype: ${lctype}`);
  }
});
```

**Script Loading Order** (in `<head>` section):
```html
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/paper.js/0.12.15/paper-full.min.js"></script>
<script src="SVGXPathPoints.js"></script>
<script src="SVGXPathPointsX.js"></script>
<script src="SVGXPathSegments.js"></script>
<script src="SVGXTsvProcessor.js"></script>
<script src="SVGXTsvVisual.js"></script>
<script src="SVGXCompositePathRendering.js"></script>
<script src="SVGXDataTableRendering.js"></script>
<script src="SVGXLineChartRendering.js"></script>  <!-- NEW -->
```

---

#### Step 5.2: Main Orchestration Method
**Methods**:
```javascript
renderLineChart(tsvContent, options)
```

**Tasks**:
- [ ] Parse TSV → extract metadata
- [ ] Extract paths → convert to logical data
- [ ] Match paths to legends
- [ ] Setup canvas and scales
- [ ] Draw axes, lines, legends
- [ ] Add logical mappings
- [ ] Enable interactivity

**Public API**:
```javascript
const renderer = new SVGXLineChartRendering(svgElement);
renderer.renderLineChart(tsvContent, {
  width: 800,
  height: 600,
  margin: { top: 60, right: 150, bottom: 60, left: 80 }
});
```

---

## Key Algorithms

### Legend Matching Algorithm
```
Input: paths[], legends[]
For each legend L:
  1. Get legend bounding box (left, top)
  2. Find paths within proximity (e.g., 50px)
  3. Among proximate paths, find color match
  4. Link: path.legendRef = legend.id
```

### Path Data Extraction
```
Input: <path d="M10,20 L30,40 C...">
1. Parse commands using SVGPathSegment API
2. Convert curves to points (sample at intervals)
3. Transform to logical coordinates using scales
4. Output: [[x1,y1], [x2,y2], ...]
```

---

## Data Structures

### Line Series Object
```javascript
{
  id: "line_0",
  legendId: "80_ci",
  legendText: "80% CI",
  data: [[x1,y1], [x2,y2], ...],
  style: {
    color: "#ff0000",
    width: 2,
    dashArray: "5,5",
    opacity: 1.0
  }
}
```

### Logical Mapping Format
```javascript
{
  xlm: "0,100,80,640",  // minX, maxX, marginLeft, innerWidth
  ylm: "0,10,60,480"    // minY, maxY, marginTop, innerHeight
}
```

---

## Testing Checklist

- [ ] Parse TSV with multiple legends
- [ ] Extract 5+ line paths correctly
- [ ] Match all paths to legends (100% accuracy)
- [ ] Transform coordinates accurately (test known points)
- [ ] Render lines with D3 (visual inspection)
- [ ] Verify xlm/ylm encoding
- [ ] Verify lc_legend_id/lc_legend_ref attributes
- [ ] Test hover interactions
- [ ] Test click toggle
- [ ] Test with different chart sizes
- [ ] Test with vertical y-axis labels

---

## Dependencies

### Existing Modules
- `SVGXPathPoints.js` - Path coordinate extraction
- `SVGXPathSegments.js` - SVG path parsing
- `SVGXTsvProcessor.js` - TSV parsing utilities
- `SVGXDataTableRendering.js` - Pattern reference

### External Libraries
- D3.js v7 (already included in wrbview.html)

---

## Performance Considerations

- Cache parsed TSV data
- Use D3 data binding for efficient updates
- Limit path point sampling (max 200 points per line)
- Debounce hover events

---

## Future Enhancements

1. Support for scatter plots (points + lines)
2. Multi-axis charts (dual y-axes)
3. Area charts (fill under line)
4. Export to JSON/CSV
5. Editable data points (drag to adjust)
6. Zoom/pan functionality

---

## File Structure
```
webview/
├── SVGXLineChartRendering.js       [NEW - main implementation]
├── SVGXDataTableRendering.js       [REFERENCE]
├── SVGXPathPoints.js               [DEPENDENCY]
├── SVGXPathSegments.js             [DEPENDENCY]
├── SVGXTsvProcessor.js             [DEPENDENCY]
├── wrbview.html                    [MODIFY - add script tag]
└── ...
```

---

## Implementation Order

1. ✅ Phase 1: Data Extraction (Steps 1.1-1.3)
2. ✅ Phase 2: Coordinate Transformation (Steps 2.1-2.2)
3. ✅ Phase 3: D3 Rendering (Steps 3.1-3.4)
4. ✅ Phase 4: Interactivity (Steps 4.1-4.2)
5. ✅ Phase 5: Integration & Testing (Steps 5.1-5.2)

**Estimated Complexity**: 800-1000 lines of code
**Estimated Time**: 12-16 hours

---

## Questions to Resolve

1. Should we support bezier curve smoothing or only linear interpolation?
2. What's the tolerance for legend-to-path matching (pixels)?
3. Should we preserve original SVG paths or only use D3-redrawn lines?
4. How to handle overlapping legends?
5. What format for data export (JSON/CSV)?
