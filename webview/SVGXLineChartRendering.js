///
/// Line Chart Rendering using D3.js
///

class SVGXLineChartRendering {
    constructor(svgElement) {
        this.svg = svgElement;
        if (!this.svg) {
            throw new Error('SVG element is not defined.');
        }
        console.log('[LOG] Line chart renderer initialized');
    }

    // ==============================================================================
    // Logical Mapping Implementation (Y-Axis Only)
    // Granular approach with explicit Left/Right label detection and Highlighting
    // ==============================================================================

    addLogicalMapping() {
        console.log("[LOG] SVGXLineChartRendering: Starting granular Y-axis logical mapping...");

        try {
            const svgRect = this.svg.viewBox.baseVal;
            const svgWidth = svgRect.width || 1536;
            const svgHeight = svgRect.height || 864;

            // 0. Find left labels first
            console.log("[LOG] Finding left axis labels...");
            const leftLabels = this._findYAxisLeftLabels(svgWidth);
            if (leftLabels.length < 2) {
                console.warn("[WARN] Not enough left labels found.");
                return;
            }

            // 1. Method 1: Find gridlines, match with left labels
            console.log("[LOG] Method 1: Finding Y-axis grid lines (> 70% width)...");
            let gridLines = this._findYAxisGridlines(svgWidth, svgHeight, 0.70);
            let yPairs = this._matchYAxisLeftLabels(gridLines, leftLabels);

            // 2. If Method 1 failed, try Method 2: left ticks
            if (yPairs.length < 2) {
                console.log("[LOG] Method 1 failed. Trying Method 2: left ticks...");
                const leftTicks = this._findYAxisLeftTicks(svgWidth, svgHeight);
                yPairs = this._matchYAxisLeftLabels(leftTicks, leftLabels);
            }

            // 3. Y-axis Final Result
            if (yPairs.length < 2) {
                console.warn("[WARN] Could not find 2 valid Y-axis pairs.");
            } else {
                // Sort by LOGICAL value (min to max)
                yPairs.sort((a, b) => a.logical - b.logical);

                const y1 = yPairs[0];
                const y2 = yPairs[yPairs.length - 1];

                const ylm_string = `[${y1.logical.toFixed(2)}, ${y2.logical.toFixed(2)}, ${y1.visual.toFixed(2)}, ${y2.visual.toFixed(2)}]`;
                this.svg.setAttribute("ylm", ylm_string);
                console.log(`[LOG] Y-axis SUCCESS: (${y1.logical}, ${y1.visual.toFixed(1)}) and (${y2.logical}, ${y2.visual.toFixed(1)})`);

                // Highlight the matched pairs
                this._highlightMatches(yPairs, 'red');

                console.log(`[LOG] Final result: ylm=${ylm_string}`);
            }

            // ==================================================================
            // X-AXIS LOGICAL MAPPING
            // ==================================================================

            // 0. Find bottom labels first
            console.log("[LOG] Finding bottom axis labels...");
            const bottomLabels = this._findXAxisBottomLabels(svgHeight);
            if (bottomLabels.length < 2) {
                console.warn("[WARN] Not enough bottom labels found.");
            } else {
                // 1. Method 1: Find vertical gridlines, match with bottom labels
                console.log("[LOG] X-Axis Method 1: Finding vertical grid lines (> 70% height)...");
                let xGridLines = this._findXAxisGridlines(svgWidth, svgHeight, 0.70);
                let xPairs = this._matchXAxisBottomLabels(xGridLines, bottomLabels);

                // 2. If Method 1 failed, try Method 2: bottom ticks
                if (xPairs.length < 2) {
                    console.log("[LOG] X-Axis Method 1 failed. Trying Method 2: bottom ticks...");
                    const bottomTicks = this._findXAxisBottomTicks(svgWidth, svgHeight);
                    xPairs = this._matchXAxisBottomLabels(bottomTicks, bottomLabels);
                }

                // 3. X-axis Final Result
                if (xPairs.length < 2) {
                    console.warn("[WARN] Could not find 2 valid X-axis pairs.");
                } else {
                    // Sort by LOGICAL value (min to max)
                    xPairs.sort((a, b) => a.logical - b.logical);

                    const x1 = xPairs[0];
                    const x2 = xPairs[xPairs.length - 1];

                    const xlm_string = `[${x1.logical.toFixed(2)}, ${x2.logical.toFixed(2)}, ${x1.visual.toFixed(2)}, ${x2.visual.toFixed(2)}]`;
                    this.svg.setAttribute("xlm", xlm_string);
                    console.log(`[LOG] X-axis SUCCESS: (${x1.logical}, ${x1.visual.toFixed(1)}) and (${x2.logical}, ${x2.visual.toFixed(1)})`);

                    // Highlight the matched pairs
                    this._highlightMatches(xPairs, 'orange');

                    console.log(`[LOG] Final result: xlm=${xlm_string}`);
                }
            }

        } catch (error) {
            console.error("[ERROR] Failed to add logical mapping:", error);
        }
    }

    // ==============================================================================
    // Granular Helper Functions
    // ==============================================================================

    /**
     * Finds horizontal grid lines based on width threshold.
     * Returns objects with left and right endpoints.
     */
    _findYAxisGridlines(svgWidth, svgHeight, minLengthRatio) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const lines = [];
        const MAX_THICKNESS = 10;
        const MIN_WIDTH = svgWidth * minLengthRatio;

        const widthlist = [];
        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box || box.width < MIN_WIDTH) return;
            const boxdata = [box.x, box.y, box.cx, box.cy, box.width, box.height];
            widthlist.push(boxdata);
        });
        console.log(`[DEBUG] The total path count is ${paths.length}`);
        console.log(`[DEBUG] The width list condition: > ${MIN_WIDTH}`);
        console.log(`[DEBUG] The width list count is ${widthlist.length}`);
        widthlist.forEach((boxdata, i) => {
            console.log(`[DEBUG] Box ${i}: x=${boxdata[0].toFixed(1)}, y=${boxdata[1].toFixed(1)}, cx=${boxdata[2].toFixed(1)}, cy=${boxdata[3].toFixed(1)}, w=${boxdata[4].toFixed(1)}, h=${boxdata[5].toFixed(1)}`);
        });

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;

            // Check if it's a horizontal line (wide enough, thin enough)
            if (box.width > MIN_WIDTH && box.height < MAX_THICKNESS) {
                lines.push({
                    element: p,
                    x: box.x,
                    y: box.cy,
                    linePos: box.cy,
                    width: box.width
                });

                console.log(`[DEBUG] Found horizontal line: ${box.x}, ${box.y}, ${box.cx}, ${box.cy}, ${box.width}, ${box.height}`);

                // Highlight this path
                p.setAttribute('fill', 'red');
                p.setAttribute('stroke', 'red');
                p.setAttribute('stroke-width', '20');
                // Bring to front
                p.parentNode.appendChild(p);
            }
        });

        console.log(`[DEBUG] Found ${lines.length} horizontal lines (minRatio=${minLengthRatio})`);
        return lines;
    }

    /**
     * Method 2: Finds Y-axis left tick marks (short horizontal lines on the left edge).
     */
    _findYAxisLeftTicks(svgWidth, svgHeight) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const ticks = [];

        // Tick mark criteria
        const MAX_TICK_WIDTH = svgWidth * 0.05;  // Less than 5% of chart width
        const MIN_TICK_WIDTH = 3;                 // At least 3 pixels wide
        const MAX_TICK_HEIGHT = 5;                // Very thin
        const LEFT_EDGE_LIMIT = svgWidth * 0.20;  // Must start within left 20%

        console.log(`[DEBUG] _findYAxisTicks: Looking for ticks with width ${MIN_TICK_WIDTH}-${MAX_TICK_WIDTH.toFixed(1)}, height < ${MAX_TICK_HEIGHT}, x < ${LEFT_EDGE_LIMIT.toFixed(1)}`);

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;

            // Check if it's a tick mark
            const isShortEnough = box.width >= MIN_TICK_WIDTH && box.width <= MAX_TICK_WIDTH;
            const isThinEnough = box.height < MAX_TICK_HEIGHT;
            const isOnLeftEdge = box.x < LEFT_EDGE_LIMIT;

            if (isShortEnough && isThinEnough && isOnLeftEdge) {
                ticks.push({
                    element: p,
                    x: box.x,
                    y: box.cy,
                    linePos: box.cy,
                    width: box.width
                });

                console.log(`[DEBUG] Found tick: x=${box.x.toFixed(1)}, y=${box.cy.toFixed(1)}, w=${box.width.toFixed(1)}, h=${box.height.toFixed(1)}`);

                // Highlight this tick
                p.setAttribute('fill', 'blue');
                p.setAttribute('stroke', 'blue');
                p.setAttribute('stroke-width', '2');
                p.parentNode.appendChild(p);
            }
        });

        console.log(`[DEBUG] Found ${ticks.length} Y-axis ticks`);
        return ticks;
    }

    /**
     * Finds numeric labels on the LEFT side of the chart (0% - 15% width).
     */
    _findYAxisLeftLabels(svgWidth) {
        const THRESHOLD = svgWidth * 0.15;
        const allLabels = this._findNumericLabels();
        const leftLabels = allLabels.filter(l => l.x < THRESHOLD);
        console.log(`[DEBUG] Found ${leftLabels.length} Left Axis labels (x < ${THRESHOLD.toFixed(0)})`);
        return leftLabels;
    }

    // ==============================================================================
    // X-Axis Helper Functions
    // ==============================================================================

    /**
     * Finds vertical grid lines (for X-axis).
     * Criteria: tall (> minLengthRatio of height), thin (width < 10).
     */
    _findXAxisGridlines(svgWidth, svgHeight, minLengthRatio) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const lines = [];
        const MAX_THICKNESS = 10;
        const MIN_HEIGHT = svgHeight * minLengthRatio;

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;

            // Check if it's a vertical line (tall enough, thin enough)
            if (box.height > MIN_HEIGHT && box.width < MAX_THICKNESS) {
                lines.push({
                    element: p,
                    x: box.cx,
                    y: box.y,
                    linePos: box.cx,
                    height: box.height
                });

                console.log(`[DEBUG] Found vertical gridline: x=${box.cx.toFixed(1)}, y=${box.y.toFixed(1)}, h=${box.height.toFixed(1)}, w=${box.width.toFixed(1)}`);

                // Highlight this gridline
                p.setAttribute('fill', 'green');
                p.setAttribute('stroke', 'green');
                p.setAttribute('stroke-width', '2');
                p.parentNode.appendChild(p);
            }
        });

        console.log(`[DEBUG] Found ${lines.length} vertical gridlines (minRatio=${minLengthRatio})`);
        return lines;
    }

    /**
     * Finds X-axis bottom tick marks (short vertical lines on the bottom edge).
     */
    _findXAxisBottomTicks(svgWidth, svgHeight) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const ticks = [];

        const MAX_TICK_HEIGHT = svgHeight * 0.5;
        const MIN_TICK_HEIGHT = 3;
        const MAX_TICK_WIDTH = 5;
        const BOTTOM_EDGE_LIMIT = svgHeight * 0.70;

        console.log(`[DEBUG] _findXAxisBottomTicks: Looking for ticks with height ${MIN_TICK_HEIGHT}-${MAX_TICK_HEIGHT.toFixed(1)}, width < ${MAX_TICK_WIDTH}, y > ${BOTTOM_EDGE_LIMIT.toFixed(1)}`);

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;

            const isTallEnough = box.height >= MIN_TICK_HEIGHT && box.height <= MAX_TICK_HEIGHT;
            const isThinEnough = box.width < MAX_TICK_WIDTH;
            const isOnBottomEdge = box.y + box.height > BOTTOM_EDGE_LIMIT;

            if (isTallEnough && isThinEnough && isOnBottomEdge) {
                ticks.push({
                    element: p,
                    x: box.cx,
                    y: box.y + box.height,
                    linePos: box.cx,
                    height: box.height
                });

                console.log(`[DEBUG] Found X tick: x=${box.cx.toFixed(1)}, y=${(box.y + box.height).toFixed(1)}, h=${box.height.toFixed(1)}, w=${box.width.toFixed(1)}`);

                // Highlight this tick
                p.setAttribute('fill', 'cyan');
                p.setAttribute('stroke', 'cyan');
                p.setAttribute('stroke-width', '2');
                p.parentNode.appendChild(p);
            }
        });

        console.log(`[DEBUG] Found ${ticks.length} X-axis bottom ticks`);
        return ticks;
    }

    /**
     * Finds numeric labels on the BOTTOM of the chart (y > 85% height).
     */
    _findXAxisBottomLabels(svgHeight) {
        const THRESHOLD = svgHeight * 0.70;
        const allLabels = this._findNumericLabels();
        const bottomLabels = allLabels.filter(l => l.y > THRESHOLD);
        console.log(`[DEBUG] Found ${bottomLabels.length} Bottom Axis labels (y > ${THRESHOLD.toFixed(0)})`);
        return bottomLabels;
    }

    /**
     * Matches Bottom Labels to vertical gridlines/ticks.
     * Anchors on lines: for each line, find the best matching label.
     */
    _matchXAxisBottomLabels(gridLines, labels) {
        if (gridLines.length < 2 || labels.length < 2) return [];

        const DISTANCE_THRESHOLD = 100;
        const matched = [];

        for (let i = 0; i < gridLines.length; i++) {
            const line = gridLines[i];
            const lineX = line.x;
            const lineY = line.y;

            let bestLabel = null;
            let bestDistance = Infinity;

            for (let j = 0; j < labels.length; j++) {
                const label = labels[j];
                const labelX = label.x;
                const labelY = label.y;

                const dx = labelX - lineX;
                const dy = labelY - lineY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }

            if (bestDistance <= DISTANCE_THRESHOLD && bestLabel !== null) {
                matched.push({
                    logical: bestLabel.value,
                    visual: line.linePos,
                    distance: bestDistance,
                    labelText: bestLabel.text,
                    labelElement: bestLabel.element,
                    gridLineElement: line.element
                });
                console.log(`[DEBUG] XLine ${i} (x=${lineX.toFixed(1)}) matched to "${bestLabel.text}" (dist=${bestDistance.toFixed(1)})`);
            } else {
                console.log(`[DEBUG] XLine ${i} (x=${lineX.toFixed(1)}) has no match (min dist=${bestDistance.toFixed(1)} > ${DISTANCE_THRESHOLD})`);
            }
        }

        // Sort by distance
        for (let i = 0; i < matched.length - 1; i++) {
            for (let j = i + 1; j < matched.length; j++) {
                if (matched[j].distance < matched[i].distance) {
                    const temp = matched[i];
                    matched[i] = matched[j];
                    matched[j] = temp;
                }
            }
        }

        // Return only the 2 best pairs
        const result = [];
        for (let i = 0; i < matched.length && i < 2; i++) {
            result.push(matched[i]);
        }

        console.log(`[DEBUG] _matchXAxisBottomLabels: Returning ${result.length} best pairs`);
        return result;
    }

    /**
     * Matches Left Labels to the LEFT endpoint (leftX) of grid lines.
     * Anchors on lines: for each line, find the best matching label.
     * Threshold: if min distance > 100, no match for that line.
     */
    _matchYAxisLeftLabels(gridLines, labels) {
        if (gridLines.length < 2 || labels.length < 2) return [];

        const DISTANCE_THRESHOLD = 100;
        const matched = [];

        // For each gridline, find the best matching label
        for (let i = 0; i < gridLines.length; i++) {
            const line = gridLines[i];
            const lineX = line.x;
            const lineY = line.y;

            let bestLabel = null;
            let bestDistance = Infinity;

            // Find the closest label to this line
            for (let j = 0; j < labels.length; j++) {
                const label = labels[j];
                const labelX = label.x;
                const labelY = label.y;

                const dx = labelX - lineX;
                const dy = labelY - lineY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }

            // Check threshold
            if (bestDistance <= DISTANCE_THRESHOLD && bestLabel !== null) {
                matched.push({
                    logical: bestLabel.value,
                    visual: line.linePos,
                    distance: bestDistance,
                    labelText: bestLabel.text,
                    labelElement: bestLabel.element,
                    gridLineElement: line.element
                });
                console.log(`[DEBUG] Line ${i} (y=${lineY.toFixed(1)}) matched to "${bestLabel.text}" (dist=${bestDistance.toFixed(1)})`);
            } else {
                console.log(`[DEBUG] Line ${i} (y=${lineY.toFixed(1)}) has no match (min dist=${bestDistance.toFixed(1)} > ${DISTANCE_THRESHOLD})`);
            }
        }

        // Sort by distance (smallest first)
        for (let i = 0; i < matched.length - 1; i++) {
            for (let j = i + 1; j < matched.length; j++) {
                if (matched[j].distance < matched[i].distance) {
                    const temp = matched[i];
                    matched[i] = matched[j];
                    matched[j] = temp;
                }
            }
        }

        // Return only the 2 best pairs
        const result = [];
        for (let i = 0; i < matched.length && i < 2; i++) {
            result.push(matched[i]);
        }

        console.log(`[DEBUG] _matchYAxisLeftLabels: Returning ${result.length} best pairs`);
        return result;
    }



    _highlightMatches(pairs, color) {
        const strokeWidth = '4';
        const fontSize = '24px';

        pairs.forEach(pair => {
            // Highlight Grid Line
            if (pair.gridLineElement) {
                // Potrace paths are filled, so we use 'fill' to highlight them.
                pair.gridLineElement.setAttribute('fill', color);
                pair.gridLineElement.setAttribute('stroke', color);
                pair.gridLineElement.setAttribute('stroke-width', strokeWidth);

                pair.gridLineElement.style.fill = color;
                pair.gridLineElement.style.stroke = color;
                pair.gridLineElement.style.strokeWidth = strokeWidth;
            }
            // Highlight Label
            if (pair.labelElement) {
                pair.labelElement.setAttribute('fill', color);
                pair.labelElement.style.fill = color;
                pair.labelElement.setAttribute('font-weight', 'bold');
            }
        });
    }

    // ==============================================================================
    // Core Logic & Utilities
    // ==============================================================================

    _findNumericLabels() {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const labels = [];
        texts.forEach(t => {
            const box = this._getFullBBox(t);
            if (!box) return;
            const text = t.textContent.trim();
            const val = parseFloat(text.replace(/[$,%]/g, ''));
            if (!isNaN(val)) {
                labels.push({
                    element: t,
                    text,
                    x: box.cx,
                    y: box.cy,
                    value: val
                });
            }
        });
        return labels;
    }

    _getFullBBox(element) {
        try {
            const bbox = element.getBBox();
            const svgRect = this.svg.getBoundingClientRect();
            const svgViewBox = this.svg.viewBox.baseVal;

            // Scale factors from screen pixels to SVG user units
            const scaleX = svgViewBox.width / svgRect.width;
            const scaleY = svgViewBox.height / svgRect.height;

            // Get element's screen position
            const elemRect = element.getBoundingClientRect();

            // Convert to SVG user units relative to viewBox
            const x = (elemRect.left - svgRect.left) * scaleX;
            const y = (elemRect.top - svgRect.top) * scaleY;
            const width = elemRect.width * scaleX;
            const height = elemRect.height * scaleY;

            return {
                x: x,
                y: y,
                width: width,
                height: height,
                cx: x + width / 2,
                cy: y + height / 2
            };
        } catch (e) {
            return null;
        }
    }
}
