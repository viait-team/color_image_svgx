# Line Chart Rendering - Prototype Tasks

**Approach**: Build incrementally, one small task at a time. Each task must have an immediate visible/verifiable result.

---

## Prototype Task 1: Basic Setup & Detection ✅ **START HERE**

**Goal**: Detect when a line chart is loaded and log basic information.

**Deliverable**: Console log message showing line chart detected with lctype attribute.

### Steps:

1. **Create skeleton `SVGXLineChartRendering.js`**
   - Empty class with constructor
   - Single method that logs "Line chart renderer initialized"

2. **Add script to `wrbview.html`**
   - Add `<script src="SVGXLineChartRendering.js"></script>` after SVGXDataTableRendering.js

3. **Modify plotBtn click handler**
   - Read `lctype` attribute from SVG
   - Branch: if `lctype === 'line'`, create new SVGXLineChartRendering instance
   - Log chart type detected

4. **Test**
   - Load a line chart SVG with `lctype="line"` attribute
   - Click plot button
   - **Verify**: Console shows "Line chart detected" message

### Expected Console Output:
```
[LOG] Chart type detected: line
[LOG] Line chart renderer initialized
```

### Success Criteria:
- [ ] SVGXLineChartRendering.js file exists
- [ ] Script loads without errors
- [ ] plotBtn handler branches correctly
- [ ] Console log appears when line chart loaded
- [ ] No errors when table chart loaded (still uses old code path)

---

## Prototype Task 2: Extract and Display All SVG Paths (NEXT)

**Goal**: Find all `<path>` elements in the SVG and display their count and basic info.

**Deliverable**: Console table showing all paths with their colors and approximate positions.

### Steps:

1. **Add method `getAllPaths()`**
   - Query all `<path>` elements
   - Filter out paths that are likely lines (stroke, no fill)
   - Return array of path elements

2. **Add method `analyzePathBasics(pathElement)`**
   - Extract stroke color
   - Get path bounding box
   - Count number of points/segments

3. **Display in console**
   - Create console.table() with path info
   - Show: index, color, x-range, y-range, segment count

4. **Test**
   - Load line chart SVG
   - Click plot button
   - **Verify**: Console table shows all line paths

### Expected Console Output:
```
┌─────┬────────┬──────────┬───────────┬────────────┬──────────┐
│ idx │ color  │ x-min    │ x-max     │ y-min      │ y-max    │
├─────┼────────┼──────────┼───────────┼────────────┼──────────┤
│ 0   │ #ff0000│ 100      │ 700       │ 200        │ 450      │
│ 1   │ #0000ff│ 100      │ 700       │ 150        │ 500      │
└─────┴────────┴──────────┴───────────┴────────────┴──────────┘
```

---

## Prototype Task 3: (TBD - will define after Task 2 works)

_Next task will be determined based on what we learn from Task 2_

Possible options:
- Extract tick labels from TSV
- Highlight paths on hover
- Extract one path's coordinates
- Find axis lines

---

## Notes

- Each task should take < 30 minutes
- Must be testable in browser immediately
- Build on previous task's code
- Only move to next task when current one is 100% working
