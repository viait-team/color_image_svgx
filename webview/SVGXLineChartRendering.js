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

        const MAX_TICK_HEIGHT = svgHeight * 0.4;
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

    // ==============================================================================
    // Legend Identification
    // ==============================================================================

    /**
     * Main entry point for legend identification and data line association.
     */
    addLegendInfo() {
        console.log("[LOG] SVGXLineChartRendering: Starting legend identification...");

        try {
            const svgRect = this.svg.viewBox.baseVal;
            const svgWidth = svgRect.width || 1536;
            const svgHeight = svgRect.height || 864;

            // Step 1: Detect the legend box region
            const legendBox = this._detectLegendBox(svgWidth, svgHeight);
            if (!legendBox) {
                console.warn("[WARN] Could not detect legend box.");
                return;
            }
            console.log(`[LOG] Legend box detected: y=${legendBox.y.toFixed(0)}, height=${legendBox.height.toFixed(0)}`);

            // Step 2: Find legend items (text + symbol pairs)
            const legendItems = this._findLegendItems(legendBox, svgWidth, svgHeight);
            if (legendItems.length === 0) {
                console.warn("[WARN] No legend items found.");
                return;
            }
            console.log(`[LOG] Found ${legendItems.length} legend items`);

            // Step 3: Apply lc_legend_id and lc_legend_instance attributes
            for (let i = 0; i < legendItems.length; i++) {
                const item = legendItems[i];
                if (item.textElement) {
                    item.textElement.setAttribute('lc_legend_id', item.id);
                }
                if (item.symbolElement) {
                    item.symbolElement.setAttribute('lc_legend_instance', item.id);
                }
                console.log(`[LOG] Legend: "${item.text}" -> id="${item.id}", color="${item.color}"`);
            }

            // Step 4: Find data lines and associate with legend
            const dataLines = this._findDataLines(svgWidth, svgHeight);
            console.log(`[LOG] Found ${dataLines.length} potential data lines`);

            this._associateLinesWithLegend(dataLines, legendItems);

        } catch (error) {
            console.error("[ERROR] Failed to add legend info:", error);
        }
    }

    /**
     * Detects the legend box region by finding horizontally-aligned text elements.
     */
    _detectLegendBox(svgWidth, svgHeight) {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const candidates = [];

        // Filter to texts in legend region (bottom 40% or right side)
        const BOTTOM_THRESHOLD = svgHeight * 0.60;

        for (let i = 0; i < texts.length; i++) {
            const t = texts[i];
            const box = this._getFullBBox(t);
            if (!box) continue;

            const text = t.textContent.trim();

            // Skip numeric texts (axis labels)
            if (/^[\d.,\-%$]+$/.test(text)) continue;

            // Skip short texts (single chars)
            if (text.length < 3) continue;

            // Skip axis titles and source text
            if (/^(years|percent|source|kamakura|trade date)/i.test(text)) continue;

            // Must be in bottom region
            if (box.y > BOTTOM_THRESHOLD) {
                candidates.push({ element: t, box: box, text: text });
            }
        }

        if (candidates.length < 2) return null;

        // Group by similar y position (within 30px)
        const groups = [];
        for (let i = 0; i < candidates.length; i++) {
            const c = candidates[i];
            let foundGroup = false;
            for (let g = 0; g < groups.length; g++) {
                if (Math.abs(groups[g].y - c.box.cy) < 30) {
                    groups[g].items.push(c);
                    foundGroup = true;
                    break;
                }
            }
            if (!foundGroup) {
                groups.push({ y: c.box.cy, items: [c] });
            }
        }

        // Find the group with most items - that's the legend row
        let bestGroup = null;
        for (let i = 0; i < groups.length; i++) {
            if (!bestGroup || groups[i].items.length > bestGroup.items.length) {
                bestGroup = groups[i];
            }
        }

        if (!bestGroup || bestGroup.items.length < 2) return null;

        // Calculate bounding box
        let minX = Infinity, maxX = 0, minY = Infinity, maxY = 0;
        for (let i = 0; i < bestGroup.items.length; i++) {
            const b = bestGroup.items[i].box;
            if (b.x < minX) minX = b.x;
            if (b.x + b.width > maxX) maxX = b.x + b.width;
            if (b.y < minY) minY = b.y;
            if (b.y + b.height > maxY) maxY = b.y + b.height;
        }

        return {
            x: minX - 100,  // Expand to include symbols to the left
            y: minY - 20,
            width: maxX - minX + 150,
            height: maxY - minY + 40,
            items: bestGroup.items
        };
    }

    /**
     * Finds legend items (text + symbol pairs) within the legend box.
     */
    _findLegendItems(legendBox, svgWidth, svgHeight) {
        const items = [];
        const paths = Array.from(this.svg.querySelectorAll('path'));

        // Use client rects and coordinate transformation as requested
        const ctm = this.svg.getScreenCTM().inverse();
        const pt = this.svg.createSVGPoint();

        for (let i = 0; i < legendBox.items.length; i++) {
            const textItem = legendBox.items[i];
            const textBox = textItem.box;

            let bestSymbol = null;
            let bestDist = Infinity;

            for (let j = 0; j < paths.length; j++) {
                const p = paths[j];

                // Get Client Rect
                const rect = p.getBoundingClientRect();

                // Check for valid rect
                if (rect.width === 0 || rect.height === 0) continue;

                // Transform 'left' and 'top' to user coordinates
                pt.x = rect.left;
                pt.y = rect.top;
                const userPos = pt.matrixTransform(ctm);

                // Transform width/height (approximation for size filtering)
                // We use the bottom-right corner to map the vector
                pt.x = rect.left + rect.width;
                pt.y = rect.top + rect.height;
                const userBottomRight = pt.matrixTransform(ctm);
                const userW = Math.abs(userBottomRight.x - userPos.x);
                const userH = Math.abs(userBottomRight.y - userPos.y);

                // Basic Size & Pos Filtering to avoid unrelated elements
                // Symbol must be small-ish (User limit: w<200, h<50)
                if (userW >= 200 || userH >= 50) continue;

                // Symbol must be to the left of text
                if (userPos.x >= textBox.x) continue;

                // Distance from Symbol Left/Top to Text Label
                const dx = textBox.x - userPos.x;
                const dy = Math.abs(textBox.y - userPos.y);

                // Relaxed vertical check 
                if (dy > 30) continue;

                // Calculate distance metric (Euclidean distance to the text anchor)
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestSymbol = p;
                }
            }

            // Extract color from symbol
            let color = '#000000';
            if (bestSymbol) {
                color = this._extractPathColor(bestSymbol);
            }

            // Generate legend ID from text
            const legendId = this._generateLegendId(textItem.text);

            items.push({
                text: textItem.text,
                textElement: textItem.element,
                symbolElement: bestSymbol,
                color: color,
                id: legendId,
                textBox: textBox
            });
        }

        return items;
    }

    /**
     * Generates a slug ID from legend text.
     */
    _generateLegendId(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '');
    }

    /**
     * Extracts the effective color from a path element.
     */
    _extractPathColor(element) {
        // Try fill attribute
        let color = element.getAttribute('fill');
        if (color && color !== 'none') return color;

        // Try style fill
        if (element.style.fill && element.style.fill !== 'none') {
            return element.style.fill;
        }

        // Try stroke
        color = element.getAttribute('stroke');
        if (color && color !== 'none') return color;

        // Try computed style
        try {
            const computed = window.getComputedStyle(element);
            if (computed.fill && computed.fill !== 'none') return computed.fill;
            if (computed.stroke && computed.stroke !== 'none') return computed.stroke;
        } catch (e) { }

        return '#000000';
    }

    /**
     * Finds ALL paths in the chart area to check for color matches.
     * We do NOT filter by size here, to catch dashed/dotted lines.
     */
    _findDataLines(svgWidth, svgHeight) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const dataLines = [];

        // Chart area bounds (exclude edges)
        const LEFT_MARGIN = svgWidth * 0.08;
        const RIGHT_MARGIN = svgWidth * 0.95;
        const TOP_MARGIN = svgHeight * 0.05;
        const BOTTOM_MARGIN = svgHeight * 0.85;

        for (let i = 0; i < paths.length; i++) {
            const p = paths[i];
            const box = this._getFullBBox(p);
            if (!box) continue;

            // Skip paths heavily outside chart area
            if (box.x + box.width < LEFT_MARGIN || box.x > RIGHT_MARGIN) continue;
            if (box.y + box.height < TOP_MARGIN || box.y > BOTTOM_MARGIN) continue;

            // Skip already-processed legend symbols
            if (p.hasAttribute('lc_legend_instance')) continue;

            // Skip huge background rects (e.g. > 90% of chart size)
            if (box.width > svgWidth * 0.9 && box.height > svgHeight * 0.9) continue;

            const color = this._extractPathColor(p);

            dataLines.push({
                element: p,
                box: box,
                color: color
            });
        }

        // DEBUG: Log color histogram
        const colorCounts = {};
        dataLines.forEach(l => {
            const c = this._normalizeColor(l.color);
            colorCounts[c] = (colorCounts[c] || 0) + 1;
        });
        console.log("[DEBUG] Chart Area Color Histogram:");
        Object.entries(colorCounts)
            .sort((a, b) => b[1] - a[1]) // Sort by count desc
            .slice(0, 10) // Top 10
            .forEach(([color, count]) => {
                console.log(`   ${color}: ${count} items`);
            });

        return dataLines;
    }

    /**
     * Associates data lines with legend items by matching colors.
     */
    /**
     * Associates data lines with legend items by matching colors.
     * Since we now check ALL paths, we MUST rely on strict color matching.
     */
    _associateLinesWithLegend(dataLines, legendItems) {
        let matchCount = 0;
        console.log(`[LOG] Associating from ${dataLines.length} candidate paths...`);

        // Check each data line against all legend items
        for (let i = 0; i < dataLines.length; i++) {
            const line = dataLines[i];
            const lineColor = this._normalizeColor(line.color);
            let bestMatch = null;
            let minDistance = Infinity;

            for (let j = 0; j < legendItems.length; j++) {
                const legendItem = legendItems[j];
                const legendColor = this._normalizeColor(legendItem.color);

                // Calculate distance
                const dist = this._getColorDistance(lineColor, legendColor);

                // Tolerance: 60 units (approx 20% in RGB space) to handle significant color shifts
                // We strictly enforce this tolerance.
                if (dist < 60 && dist < minDistance) {
                    minDistance = dist;
                    bestMatch = legendItem;
                }
            }

            if (bestMatch) {
                line.element.setAttribute('lc_legend_ref', bestMatch.id);
                matchCount++;
            }
        }

        console.log(`[LOG] Associated ${matchCount} data lines with legend items`);
    }

    /**
     * Calculates Euclidean distance between two colors in RGB space.
     * Colors must be hex strings (e.g. "#rrggbb").
     */
    _getColorDistance(c1, c2) {
        if (!c1 || !c2 || !c1.startsWith('#') || !c2.startsWith('#')) return Infinity;

        const r1 = parseInt(c1.substring(1, 3), 16);
        const g1 = parseInt(c1.substring(3, 5), 16);
        const b1 = parseInt(c1.substring(5, 7), 16);

        const r2 = parseInt(c2.substring(1, 3), 16);
        const g2 = parseInt(c2.substring(3, 5), 16);
        const b2 = parseInt(c2.substring(5, 7), 16);

        return Math.sqrt(
            Math.pow(r1 - r2, 2) +
            Math.pow(g1 - g2, 2) +
            Math.pow(b1 - b2, 2)
        );
    }

    /**
     * Normalizes a color to a comparable format (hex).
     */
    _normalizeColor(color) {
        if (!color) return '';

        color = color.trim().toLowerCase();

        // Already hex
        if (color.startsWith('#')) {
            if (color.length === 4) { // #RGB -> #RRGGBB
                return '#' + color[1] + color[1] + color[2] + color[2] + color[3] + color[3];
            }
            return color;
        }

        // RGB format
        const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (rgbMatch) {
            const r = parseInt(rgbMatch[1]).toString(16).padStart(2, '0');
            const g = parseInt(rgbMatch[2]).toString(16).padStart(2, '0');
            const b = parseInt(rgbMatch[3]).toString(16).padStart(2, '0');
            return `#${r}${g}${b}`;
        }

        return color; // Return as is if unknown format
    }

    /**
     * Enables interactivity: clicking legend label flashes corresponding data lines.
     */
    enableLegendInteractivity() {
        console.log("[LOG] SVGXLineChartRendering: Enabling legend interactivity...");

        const legendLabels = Array.from(this.svg.querySelectorAll('text[lc_legend_id]'));

        legendLabels.forEach(label => {
            // Style cursor to indicate clickable
            label.style.cursor = 'pointer';

            // Add click listener
            label.addEventListener('click', () => {
                const legendId = label.getAttribute('lc_legend_id');
                console.log(`[LOG] Legend clicked: ${legendId}`);
                this._flashDataLines(legendId);
            });
        });
    }

    /**
     * Flashes matching data lines for a given legend ID.
     */
    /**
     * Flashes matching data lines for a given legend ID.
     */
    _flashDataLines(legendId) {
        const lines = Array.from(this.svg.querySelectorAll(`path[lc_legend_ref="${legendId}"]`));
        if (lines.length === 0) return;

        console.log(`[LOG] Flashing ${lines.length} lines for legend ${legendId}`);

        lines.forEach(line => {
            // Get effective color to use for highlighting
            const color = this._extractPathColor(line);

            // Save original styles
            const originalStroke = line.style.stroke;
            const originalStrokeWidth = line.style.strokeWidth || line.getAttribute('stroke-width') || '1';
            const originalOpacity = line.style.opacity || '1';
            const originalFill = line.style.fill;

            // Apply flash effect
            // 1. Force stroke color (Red for high visibility)
            line.style.stroke = 'red';

            // 2. Make it thick
            line.style.transition = 'all 0.2s ease-in-out';
            line.style.strokeWidth = '40px'; // Very thick to be obvious

            // 3. Briefly dim it before brightening (flash effect)
            line.style.opacity = '0.5';

            // bring to front
            if (line.parentNode) {
                line.parentNode.appendChild(line);
            }

            // Timeline:
            // 0ms: Start (thick, dim)
            // 100ms: Brighten (full opacity)
            setTimeout(() => {
                line.style.opacity = '1';
            }, 100);

            // 600ms: Revert
            setTimeout(() => {
                line.style.strokeWidth = originalStrokeWidth;
                line.style.stroke = originalStroke; // Remove forced stroke
                line.style.opacity = originalOpacity;

                setTimeout(() => {
                    line.style.transition = '';
                    // Restore original fill if needed (though we didn't change it)
                }, 200);
            }, 600);
        });
    }
}
