/**
 * 1. SVGXLineChartAnalyzer
 * Responsible for reverse-engineering the SVG: finding axes, gridlines, labels, legends, and interactivity.
 */
class SVGXLineChartAnalyzer {
    constructor(svgElement, clrContent = null) {
        this.svg = svgElement;
        // Parse the .clr file content into a Set for efficient lookups
        this.officialPalette = clrContent ? new Set(clrContent.trim().split(/\r?\n/).map(c => this._normalizeColor(c))) : null;
        this.legendItems = []; // Store legend items here
        this.chartArea = null; // Store chart area here
    }

    // ==============================================================================
    // Logical Mapping Implementation (Y-Axis Only)
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



            // 1. Find Y gridlines (Method 1) and try matching
            console.log("[LOG] Method 1: Finding Y-axis grid lines (> 70% width)...");
            let yGridLines = this._findYAxisGridlines(svgWidth, svgHeight, 0.70);
            console.log(`[LOG] Found ${yGridLines.length} Y gridlines`);
            let yPairs = this._matchYAxisLeftLabels(yGridLines, leftLabels);

            // 2. If Method 1 matching failed, try Method 2: left ticks
            if (yPairs.length < 2) {
                console.log("[LOG] Y-axis Method 1 matching failed. Trying Method 2: left ticks...");
                const yTicks = this._findYAxisLeftTicks(svgWidth, svgHeight);
                console.log(`[LOG] Found ${yTicks.length} Y ticks`);
                yPairs = this._matchYAxisLeftLabels(yTicks, leftLabels);
                if (yPairs.length >= 2) {
                    yGridLines = yTicks; // Use ticks for chart area
                }
            }

            // 3. Find X gridlines/ticks
            console.log("[LOG] X-Axis Method 1: Finding vertical grid lines (> 70% height)...");
            let xGridLines = this._findXAxisGridlines(svgWidth, svgHeight, 0.70);
            console.log(`[LOG] Found ${xGridLines.length} X gridlines`);
            if (xGridLines.length < 2) {
                console.log("[LOG] X-Axis Method 1 failed. Trying Method 2: bottom ticks...");
                xGridLines = this._findXAxisBottomTicks(svgWidth, svgHeight);
                console.log(`[LOG] Found ${xGridLines.length} X ticks`);
            }

            // 4. LOG CHART AREA from matched Y gridlines/ticks and X gridlines/ticks
            if (xGridLines.length >= 2 && yGridLines.length >= 2) {
                const allXVisuals = xGridLines.map(g => g.x);
                const allYVisuals = yGridLines.map(g => g.y);
                const chartLeft = Math.min(...allXVisuals);
                const chartRight = Math.max(...allXVisuals);
                const chartTop = Math.min(...allYVisuals);
                const chartBottom = Math.max(...allYVisuals);
                console.log(`[LOG] Chart area: left=${chartLeft.toFixed(1)}, top=${chartTop.toFixed(1)}, right=${chartRight.toFixed(1)}, bottom=${chartBottom.toFixed(1)}`);

                // Store the calculated chart area for later use
                this.chartArea = {
                    visualX: [chartLeft, chartRight],
                    visualY: [chartTop, chartBottom]
                };
            }

            // 5. Y-axis Final Result
            if (yPairs.length < 2) {
                console.warn("[WARN] Could not find 2 valid Y-axis pairs.");
            } else {
                yPairs.sort((a, b) => a.logical - b.logical);
                const y1 = yPairs[0];
                const y2 = yPairs[yPairs.length - 1];
                const ylm_string = `[${y1.logical}, ${y2.logical}, ${y1.visual}, ${y2.visual}]`;
                this.svg.setAttribute("ylm", ylm_string);
                console.log(`[LOG] Final result: ylm=${ylm_string}`);
            }

            // ==================================================================
            // X-AXIS LOGICAL MAPPING (matching step)
            // ==================================================================

            console.log("[LOG] Finding bottom axis labels...");
            const bottomLabels = this._findXAxisBottomLabels(svgHeight);
            let xPairs = [];
            if (bottomLabels.length < 2) {
                console.warn("[WARN] Not enough bottom labels found.");
            } else {
                xPairs = this._matchXAxisBottomLabels(xGridLines, bottomLabels);
            }

            if (xPairs.length < 2) {
                console.log("[LOG] X-axis matching with gridlines failed or insufficient. Trying fallback to ticks...");
                // Fallback: Try ticks
                const xTicks = this._findXAxisBottomTicks(svgWidth, svgHeight);
                console.log(`[LOG] Found ${xTicks.length} X ticks (fallback)`);
                if (xTicks.length >= 2) {
                    xPairs = this._matchXAxisBottomLabels(xTicks, bottomLabels);
                }
            }

            if (xPairs.length < 2) {
                console.warn("[WARN] Could not find 2 valid X-axis pairs.");
            } else {
                xPairs.sort((a, b) => a.logical - b.logical);
                const x1 = xPairs[0];
                const x2 = xPairs[xPairs.length - 1];
                const xlm_string = `[${x1.logical}, ${x2.logical}, ${x1.visual}, ${x2.visual}]`;
                this.svg.setAttribute("xlm", xlm_string);
                // this._highlightMatches(xPairs, 'orange');
                console.log(`[LOG] Final result: xlm=${xlm_string}`);
            }

        } catch (error) {
            console.error("[ERROR] Failed to add logical mapping:", error);
        }
    }

    /**
     * Public getter for the calculated chart area.
     * @returns {object|null} The chart area object or null if not calculated.
     */
    getChartArea() {
        return this.chartArea;
    }

    // ==============================================================================
    // Helper Functions
    // ==============================================================================

    _findYAxisGridlines(svgWidth, svgHeight, minLengthRatio) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const lines = [];
        const MAX_THICKNESS = 10;
        const MIN_WIDTH = svgWidth * minLengthRatio;

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;
            if (box.width > MIN_WIDTH && box.height < MAX_THICKNESS) {
                lines.push({
                    element: p,
                    x: box.x,
                    y: box.cy,
                    linePos: box.cy,
                    width: box.width
                });
                // p.setAttribute('fill', 'red');
                // p.setAttribute('stroke', 'red');
                // p.setAttribute('stroke-width', '20');
                // p.parentNode.appendChild(p);
            }
        });
        return lines;
    }

    _findYAxisLeftTicks(svgWidth, svgHeight) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const ticks = [];
        const MAX_TICK_WIDTH = svgWidth * 0.01;
        const MIN_TICK_WIDTH = 3;
        const MAX_TICK_HEIGHT = 10;
        const LEFT_EDGE_LIMIT = svgWidth * 0.20;

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;
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
                // p.setAttribute('fill', 'blue');
                // p.setAttribute('stroke', 'blue');
                // p.setAttribute('stroke-width', '2');
                // p.parentNode.appendChild(p);
            }
        });
        return ticks;
    }

    _findYAxisLeftLabels(svgWidth) {
        const THRESHOLD = svgWidth * 0.15;
        const allLabels = this._findNumericLabels();
        return allLabels.filter(l => l.x < THRESHOLD);
    }

    _findXAxisGridlines(svgWidth, svgHeight, minLengthRatio) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const lines = [];
        const MAX_THICKNESS = 10;
        const MIN_HEIGHT = svgHeight * minLengthRatio;

        paths.forEach(p => {
            const box = this._getFullBBox(p);
            if (!box) return;
            if (box.height > MIN_HEIGHT && box.width < MAX_THICKNESS) {
                lines.push({
                    element: p,
                    x: box.cx,
                    y: box.y,
                    linePos: box.cx,
                    height: box.height
                });
                // p.setAttribute('fill', 'green');
                // p.setAttribute('stroke', 'green');
                // p.setAttribute('stroke-width', '2');
                // p.parentNode.appendChild(p);
            }
        });
        return lines;
    }

    _findXAxisBottomTicks(svgWidth, svgHeight) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const ticks = [];
        const MAX_TICK_HEIGHT = svgHeight * 0.4;
        const MIN_TICK_HEIGHT = 3;
        const MAX_TICK_WIDTH = 10;

        const BOTTOM_EDGE_LIMIT = svgHeight * 0.70;

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
                // p.setAttribute('fill', 'cyan');
                // p.setAttribute('stroke', 'cyan');
                // p.setAttribute('stroke-width', '2');
                // p.parentNode.appendChild(p);
            }
        });
        return ticks;
    }

    _findXAxisBottomLabels(svgHeight) {
        const THRESHOLD = svgHeight * 0.70;
        const allLabels = this._findNumericLabels();
        return allLabels.filter(l => l.y > THRESHOLD);
    }

    _matchXAxisBottomLabels(gridLines, labels) {
        if (gridLines.length < 2 || labels.length < 2) return [];

        // --- NEW ALIGNMENT FILTER: Filter out labels that are vertically misaligned ---
        // Calculate Median Y of all labels
        const ys = labels.map(l => l.y);
        ys.sort((a, b) => a - b);
        const medianY = ys[Math.floor(ys.length / 2)];

        // Filter out outliers (e.g. Y-axis labels caught in the bottom scan)
        const validLabels = labels.filter(l => Math.abs(l.y - medianY) < 30);

        const matched = [];

        for (let j = 0; j < validLabels.length; j++) {
            const label = validLabels[j];
            let bestLine = null;
            let bestDistance = Infinity;

            for (let i = 0; i < gridLines.length; i++) {
                const line = gridLines[i];
                const dx = label.x - line.x;
                const dy = label.y - line.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLine = line;
                }
            }

            if (bestDistance <= 100 && bestLine !== null) {
                matched.push({
                    logical: label.value,
                    visual: bestLine.linePos,
                    distance: bestDistance,
                    labelText: label.text,
                    labelElement: label.element,
                    gridLineElement: bestLine.element
                });
            }
        }

        matched.sort((a, b) => a.distance - b.distance);
        return matched;
    }

    _matchYAxisLeftLabels(gridLines, labels) {
        if (gridLines.length < 2 || labels.length < 2) return [];

        // --- NEW ALIGNMENT FILTER: Filter out labels that are horizontally misaligned ---
        // Calculate Median X of all labels
        const xs = labels.map(l => l.x);
        xs.sort((a, b) => a - b);
        const medianX = xs[Math.floor(xs.length / 2)];

        // Filter out outliers (e.g. X-axis labels caught in the left scan)
        const validLabels = labels.filter(l => Math.abs(l.x - medianX) < 30);

        const matched = [];

        for (let j = 0; j < validLabels.length; j++) {
            const label = validLabels[j];
            let bestLine = null;
            let bestDistance = Infinity;

            for (let i = 0; i < gridLines.length; i++) {
                const line = gridLines[i];
                const dx = label.x - line.x;
                const dy = label.y - line.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLine = line;
                }
            }

            if (bestDistance <= 100 && bestLine !== null) {
                matched.push({
                    logical: label.value,
                    visual: bestLine.linePos,
                    distance: bestDistance,
                    labelText: label.text,
                    labelElement: label.element,
                    gridLineElement: bestLine.element
                });
            }
        }

        if (matched.length < 2) return matched;

        // Find two matches with the largest separation in visual position
        let maxSep = -Infinity;
        let pair = [];
        for (let i = 0; i < matched.length; i++) {
            for (let j = i + 1; j < matched.length; j++) {
                const sep = Math.abs(matched[i].visual - matched[j].visual);
                if (sep > maxSep) {
                    maxSep = sep;
                    pair = [matched[i], matched[j]];
                }
            }
        }

        return pair;
    }

    _highlightMatches(pairs, color) {
        pairs.forEach(pair => {
            if (pair.gridLineElement) {
                pair.gridLineElement.setAttribute('fill', color);
                pair.gridLineElement.setAttribute('stroke', color);
                pair.gridLineElement.setAttribute('stroke-width', '4');
                pair.gridLineElement.style.fill = color;
                pair.gridLineElement.style.stroke = color;
            }
            if (pair.labelElement) {
                pair.labelElement.setAttribute('fill', color);
                pair.labelElement.style.fill = color;
                pair.labelElement.setAttribute('font-weight', 'bold');
            }
        });
    }

    _extractTickLabelNumber(text) {
        if (text == null) return null;

        const raw = text.trim();
        if (!raw) return null;

        // ------------------------------------------------------------
        // 1. strip
        // ------------------------------------------------------------
        const stripText = raw.replace(/[$€£¥,%]/g, '');

        // ------------------------------------------------------------
        // 2. validate raw structure (Option C)
        // ------------------------------------------------------------
        const pattern = new RegExp(
            '^' +
            '(?:[$€£¥]|[A-Z]{3})?\\s*' +
            '[+-]?' +
            '(?:' +
            '(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?' +
            '|' +
            '\\.\\d+' +
            ')' +
            '(?:[eE][+-]?\\d+)?' +
            '\\s*' +
            '(?:' +
            '\\(([a-zA-Zµ·*/^0-9%\\s-]+)\\)' +
            '|' +
            '(?<![eE])[a-df-zA-DF-Zµ%][a-zA-Z0-9µ·*/^%-]*' +
            ')?' +
            '([kKmMbBtT])?' +
            '\\s*$'
        );

        if (!pattern.test(raw)) return null;

        // ------------------------------------------------------------
        // 3. unit removal (Option C)
        // ------------------------------------------------------------
        let noUnit = stripText.replace(
            /\(([a-zA-Zµ·*/^0-9%\s-]+)\)$|(?<![eE])[a-df-zA-DF-Zµ%][a-zA-Z0-9µ·*/^%-]*$/g,
            ''
        ).trim();

        // ------------------------------------------------------------
        // 4. suffix factor (minimal fix)
        // ------------------------------------------------------------
        let factor = 1;

        const suffixMatch = raw.match(/([kKmMbBtT])\s*$/);
        if (suffixMatch) {
            const suffixChar = suffixMatch[1];
            const idx = raw.lastIndexOf(suffixChar);
            const prev = idx > 0 ? raw[idx - 1] : null;

            // ✅ Only apply suffix if previous char is NOT a letter
            if (!prev || !/[a-zA-Z]/.test(prev)) {
                const suffix = suffixChar.toLowerCase();
                if (suffix === 'k') factor = 1_000;
                if (suffix === 'm') factor = 1_000_000;
                if (suffix === 'b') factor = 1_000_000_000;
                if (suffix === 't') factor = 1_000_000_000_000;
            }
        }

        // ------------------------------------------------------------
        // 5. thousand separator
        // ------------------------------------------------------------
        const noThousands = noUnit.replace(/,/g, '');

        // ------------------------------------------------------------
        // 6. extract digits (scientific notation)
        // ------------------------------------------------------------
        const numberMatch = noThousands.match(
            /[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/
        );
        if (!numberMatch) return null;

        const baseValue = Number(numberMatch[0]);
        if (Number.isNaN(baseValue)) return null;

        // ------------------------------------------------------------
        // 7. apply factor
        // ------------------------------------------------------------
        return baseValue * factor;
    }


    _findNumericLabels() {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const labels = [];
        texts.forEach(t => {
            const box = this._getFullBBox(t);
            if (!box) return;
            const text = t.textContent.trim();
            const val = this._extractTickLabelNumber(text);
            if (val !== null && !Number.isNaN(val)) {
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

            // Get the Screen CTM of the element
            const ctm = element.getScreenCTM();
            // Get the Screen CTM of the root SVG
            const rootCTM = this.svg.getScreenCTM();

            // Transform element's CTM to be relative to the Root SVG's local coordinate system
            // This effectively removes the window/browser scaling/positioning
            const transformMatrix = rootCTM.inverse().multiply(ctm);

            // Create points for the corners of the bbox
            const pt = this.svg.createSVGPoint();
            const corners = [
                { x: bbox.x, y: bbox.y },
                { x: bbox.x + bbox.width, y: bbox.y },
                { x: bbox.x + bbox.width, y: bbox.y + bbox.height },
                { x: bbox.x, y: bbox.y + bbox.height }
            ];

            // Transform corners
            const transformedCorners = corners.map(c => {
                pt.x = c.x;
                pt.y = c.y;
                return pt.matrixTransform(transformMatrix);
            });

            // Calculate new axis-aligned bounding box in Root SVG Space
            const xCoords = transformedCorners.map(c => c.x);
            const yCoords = transformedCorners.map(c => c.y);
            const minX = Math.min(...xCoords);
            const maxX = Math.max(...xCoords);
            const minY = Math.min(...yCoords);
            const maxY = Math.max(...yCoords);

            return {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY - minY,
                cx: minX + (maxX - minX) / 2,
                cy: minY + (maxY - minY) / 2
            };
        } catch (e) {
            console.warn("Failed to get stable bbox:", e);
            return null;
        }
    }

    // ==============================================================================
    // Legend Identification
    // ==============================================================================

    addLegendInfo() {
        console.log("[LOG] SVGXLineChartRendering: Starting legend identification...");
        try {
            const svgRect = this.svg.viewBox.baseVal;
            const svgWidth = svgRect.width || 1536;
            const svgHeight = svgRect.height || 864;
            const legendBox = this._detectLegendBox(svgWidth, svgHeight);
            if (!legendBox) {
                console.warn("[WARN] Could not detect legend box.");
                return;
            }
            console.log(`[LOG] Legend box detected: y=${legendBox.y.toFixed(0)}, height=${legendBox.height.toFixed(0)}`);
            const legendItems = this._findLegendItems(legendBox, svgWidth, svgHeight);
            if (legendItems.length === 0) {
                console.warn("[WARN] No legend items found.");
                return;
            }
            console.log(`[LOG] Found ${legendItems.length} legend items`);
            this.legendItems = legendItems; // Save for later use in extraction

            for (let i = 0; i < legendItems.length; i++) {
                const item = legendItems[i];
                if (item.textElement) {
                    item.textElement.setAttribute('lc_legend_id', item.id);
                }
                if (item.symbolElement) {
                    item.symbolElement.setAttribute('lc_legend_instance', item.id);
                }
                console.log(`[LOG] Legend: "${item.text}" -> id="${item.id}", color="${item.color}", type="${item.lc_legend_type}"`);
            }
            // Pass the official palette to the data line finding function
            const dataLines = this._findDataLines(svgWidth, svgHeight, this.officialPalette);
            if (this.officialPalette) {
                console.log(`[LOG] Using an official palette of ${this.officialPalette.size} colors for matching.`);
            }

            console.log(`[LOG] Found ${dataLines.length} potential data lines`);

            // MARKER DETECTION: Scan all candidate paths and mark small ones as markers
            this._detectMarkers(dataLines);

            this._associateLinesWithLegend(dataLines, legendItems);
        } catch (error) {
            console.error("[ERROR] Failed to add legend info:", error);
        }
    }

    _detectLegendBox(svgWidth, svgHeight) { // No changes, just for context
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const candidates = [];
        const BOTTOM_THRESHOLD = svgHeight * 0.60;
        for (let i = 0; i < texts.length; i++) {
            const t = texts[i];
            const box = this._getFullBBox(t);
            if (!box) continue;
            const text = t.textContent.trim();
            if (/^[\d.,\-%$]+$/.test(text)) continue;
            if (text.length < 3) continue;
            if (/^(years|percent|source|kamakura|trade date)/i.test(text)) continue;
            if (box.y > BOTTOM_THRESHOLD) {
                candidates.push({ element: t, box: box, text: text });
            }
        }
        if (candidates.length < 2) return null;
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
        let bestGroup = null;
        for (let i = 0; i < groups.length; i++) {
            if (!bestGroup || groups[i].items.length > bestGroup.items.length) {
                bestGroup = groups[i];
            }
        }
        if (!bestGroup || bestGroup.items.length < 2) return null;
        let minX = Infinity, maxX = 0, minY = Infinity, maxY = 0;
        for (let i = 0; i < bestGroup.items.length; i++) {
            const b = bestGroup.items[i].box;
            if (b.x < minX) minX = b.x;
            if (b.x + b.width > maxX) maxX = b.x + b.width;
            if (b.y < minY) minY = b.y;
            if (b.y + b.height > maxY) maxY = b.y + b.height;
        }
        return {
            x: minX - 100,
            y: minY - 20,
            width: maxX - minX + 150,
            height: maxY - minY + 40,
            items: bestGroup.items
        };
    }

    _findLegendItems(legendBox, svgWidth, svgHeight) {
        const items = [];
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const ctm = this.svg.getScreenCTM().inverse();
        const pt = this.svg.createSVGPoint();

        for (let i = 0; i < legendBox.items.length; i++) {
            const textItem = legendBox.items[i];
            const textBox = textItem.box;
            let candidates = [];
            const textCy = textBox.y + (textBox.height / 2);

            for (let j = 0; j < paths.length; j++) {
                const p = paths[j];
                const rect = p.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) continue;

                pt.x = rect.left;
                pt.y = rect.top;
                const userPos = pt.matrixTransform(ctm);
                pt.x = rect.left + rect.width;
                pt.y = rect.top + rect.height;
                const userBottomRight = pt.matrixTransform(ctm);
                const userW = Math.abs(userBottomRight.x - userPos.x);
                const userH = Math.abs(userBottomRight.y - userPos.y);

                if (userW >= 200 || userH >= 50) continue;

                if (userW < 1 || userH < 1) continue;

                if (userPos.x >= textBox.x) continue;

                const symbolCy = userPos.y + (userH / 2);
                const dy = Math.abs(textCy - symbolCy);
                const dx = textBox.x - userPos.x;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dy <= 30 && dist < 150) {
                    candidates.push({
                        element: p,
                        dist: dist,
                        area: userW * userH,
                        width: userW,
                        height: userH
                    });
                }
            }

            let bestSymbol = null;
            let legendType = 'line'; // Default to line

            if (candidates.length > 0) {

                let bestCandidate = candidates[0];
                bestSymbol = bestCandidate.element;

                // Find the darkest candidate
                let darkestCandidate = null;
                let minLuminance = Infinity;

                // Iterate through candidates to find the one with the darkest color.
                // A lower luminance value means the color is darker.
                for (const candidate of candidates) {
                    const candidateColor = this._extractPathColor(candidate.element);
                    const luminance = this._getColorLuminance(candidateColor);

                    if (luminance < minLuminance) {
                        minLuminance = luminance;
                        darkestCandidate = candidate;
                    }
                }
                bestCandidate = darkestCandidate || bestCandidate;
                bestSymbol = bestCandidate.element;

                // Determine type based on aspect ratio and size
                const ratio = bestCandidate.width / bestCandidate.height;
                // Markers typically have aspect ratio close to 1 and are small?
                // Or maybe just based on size?
                // Let's use the logic from the plan:
                // Marker: small size (< 30px?), aspect ratio 0.5 < r < 2.0
                if (bestCandidate.width < 30 && bestCandidate.height < 30 && ratio > 0.5 && ratio < 2.0) {
                    legendType = 'marker';
                }
            }

            let color = '#000000';
            if (bestSymbol) {
                color = this._extractPathColor(bestSymbol);
                // Debug: Log the legend symbol details
                const fill = bestSymbol.getAttribute('fill');
                const stroke = bestSymbol.getAttribute('stroke');
                console.log(`[DEBUG] Legend "${textItem.text}": symbol fill="${fill}", stroke="${stroke}", extracted="${color}"`);
            }

            const legendId = this._generateLegendId(textItem.text);

            const legendItem = {
                text: textItem.text,
                textElement: textItem.element,
                symbolElement: bestSymbol,
                color: color,
                id: legendId,
                textBox: textBox,
                lc_legend_type: legendType
            };

            if (bestSymbol) {
                bestSymbol.setAttribute('lc_legend_type', legendType);
            }

            items.push(legendItem);
        }

        // Check if all legends are markers
        const allAreMarker = items.every(
            item => item.lc_legend_type === 'marker'
        );

        if (allAreMarker) {
            for (let i = 0; i < items.length; i++) {
                items[i].lc_legend_type = 'line';
            }
            console.log(`[LOG] All marker legends converted to line legends.`);
        }

        return items;
    }

    _generateLegendId(text) {
        return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    }

    _extractPathColor(element) {
        let color = element.getAttribute('fill');
        if (color && color !== 'none') return color;

        color = element.getAttribute('stroke');
        if (color && color !== 'none') return color;

        // Check parent element attributes only (no style)
        const parent = element.parentElement;
        if (parent) {
            let parentColor = parent.getAttribute('fill');
            if (parentColor && parentColor !== 'none') return parentColor;

            parentColor = parent.getAttribute('stroke');
            if (parentColor && parentColor !== 'none') return parentColor;
        }

        try {
            const computed = window.getComputedStyle(element);
            if (computed.fill && computed.fill !== 'none') return computed.fill;
            if (computed.stroke && computed.stroke !== 'none') return computed.stroke;
        } catch (e) { }

        return '#000000';
    }

    _findDataLines(svgWidth, svgHeight, officialPalette) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const dataLines = [];
        const LEFT_MARGIN = svgWidth * 0.08;
        const RIGHT_MARGIN = svgWidth * 0.95;
        const TOP_MARGIN = svgHeight * 0.05;
        const BOTTOM_MARGIN = svgHeight * 0.85;

        for (let i = 0; i < paths.length; i++) {
            const p = paths[i];
            const box = this._getFullBBox(p);
            if (!box) continue;

            // 1. General Margins (Backup)
            if (box.x + box.width < LEFT_MARGIN || box.x > RIGHT_MARGIN) continue;
            if (box.y + box.height < TOP_MARGIN || box.y > BOTTOM_MARGIN) continue;

            if (p.hasAttribute('lc_legend_instance')) continue;
            if (box.width > svgWidth * 0.9 && box.height > svgHeight * 0.9) continue;

            const color = this._extractPathColor(p);
            const normalizedColor = this._normalizeColor(color);

            // If an official palette is provided, only consider paths that use a palette color.
            // This dramatically reduces noise from anti-aliased or background shapes.
            if (officialPalette) {
                if (officialPalette.has(normalizedColor)) {
                    dataLines.push({ element: p, box: box, color: color });
                }
            } else {
                // Fallback to the old behavior if no palette is available.
                dataLines.push({ element: p, box: box, color: color });
            }
        }

        const colorCounts = {};
        dataLines.forEach(l => {
            const c = this._normalizeColor(l.color);
            colorCounts[c] = (colorCounts[c] || 0) + 1;
        });
        return dataLines;
    }

    /**
     * Detects small paths that are markers (not lines) based on bounding box size.
     * Adds an 'isMarker' property to each data line object.
     * @param {Array} dataLines - Array of {element, box, color} objects
     */
    _detectMarkers(dataLines) {
        const MARKER_THRESHOLD = 15; // Max dimension in pixels to be considered a marker
        let markerCount = 0;

        dataLines.forEach(line => {
            const box = line.box;
            const minDim = Math.min(box.width, box.height);
            const maxDim = Math.max(box.width, box.height);
            // Marker: min dimension > 4px AND max dimension < 15px
            if (box && minDim > 4 && maxDim < MARKER_THRESHOLD) {
                line.isMarker = true;
                line.element.setAttribute('lc_is_marker', 'true');
                markerCount++;
                console.log(`[LOG] Marker detected: pos=(${box.cx.toFixed(1)}, ${box.cy.toFixed(1)}), bbox=${box.width.toFixed(1)}x${box.height.toFixed(1)}, color=${line.color}`);
            } else {
                line.isMarker = false;
            }
        });

        console.log(`[LOG] Marker detection complete: ${markerCount} markers found out of ${dataLines.length} paths`);
    }

    _associateLinesWithLegend(dataLines, legendItems) {
        let matchCount = 0;
        console.log(`[LOG] Associating from ${dataLines.length} candidate paths...`);

        // Group legend items
        const legendMarkers = legendItems.filter(item => item.lc_legend_type === 'marker');
        const legendLines = legendItems.filter(item => item.lc_legend_type === 'line');
        const unassociatedPaths = new Set(dataLines);

        // Get chart area bounds from xlm/ylm if available
        let chartArea = null;
        const xlmStr = this.svg.getAttribute('xlm');
        const ylmStr = this.svg.getAttribute('ylm');
        if (xlmStr && ylmStr) {
            try {
                const xlm = xlmStr.split(',').map(Number);
                const ylm = ylmStr.split(',').map(Number);
                if (xlm.length === 4 && ylm.length === 4) {
                    chartArea = {
                        xMin: Math.min(xlm[2], xlm[3]),
                        xMax: Math.max(xlm[2], xlm[3]),
                        yMin: Math.min(ylm[2], ylm[3]),
                        yMax: Math.max(ylm[2], ylm[3])
                    };
                    // Add small tolerance
                    const marginX = (chartArea.xMax - chartArea.xMin) * 0.01;
                    const marginY = (chartArea.yMax - chartArea.yMin) * 0.01;
                    chartArea.xMin -= marginX;
                    chartArea.xMax += marginX;
                    chartArea.yMin -= marginY;
                    chartArea.yMax += marginY;
                }
            } catch (e) { console.warn("Failed to parse bounds for marker check", e); }
        }

        // --- Step 1: Marker Association ---
        if (legendMarkers.length > 0) {
            this._initializePaper();

            // Helper to check if marker is valid for a specific legend item
            /*   const isValidMarkerForLegend = (line, legendItem) => {
                  // 1. Size Check relative to legend symbol
                  if (legendItem.symbolElement) {
                      try {
                          const legBox = legendItem.symbolElement.getBBox();
                          const lineBox = line.box;
                          // Determine scaling factors or direct size comparison
                          // We use a simplified check: width and height must be at least 90% of legend symbol
                          // Or allow for some rotation/scaling differences by checking area or average dimension
                          const legDim = Math.max(legBox.width, legBox.height);
                          const lineDim = Math.max(lineBox.width, lineBox.height);
  
                          // Strict check: dimensions must be comparable
                          // User requirement: path size should not be less than 90% of legend symbol size
                          if (lineDim < legDim * 1.5) {
                              return false;
                          }
                      } catch (e) { }
                  }
                  return true;
              }; */

            // Case A: Single Marker Type
            if (legendMarkers.length === 1) {
                const legendMarker = legendMarkers[0];
                const legendColor = this._normalizeColor(legendMarker.color);

                for (const line of dataLines) {
                    if (this._isMarkerCandidate(line)) {
                        // Apply specific checks
                        // if (!isValidMarkerForLegend(line, legendMarker)) continue;

                        const lineColor = this._normalizeColor(line.color);
                        const dist = this._getColorDistance(lineColor, legendColor);
                        if (dist < 60) { // Color match
                            line.element.setAttribute('lc_legend_ref', legendMarker.id);
                            unassociatedPaths.delete(line);
                            matchCount++;
                        }
                    }
                }
            }
            // Case B: Multi-Marker Types
            else {
                for (const line of dataLines) {
                    if (this._isMarkerCandidate(line)) {
                        let bestMatch = null;
                        let bestScore = -1;

                        for (const legendMarker of legendMarkers) {
                            // Apply specific checks
                            // if (!isValidMarkerForLegend(line, legendMarker)) continue;

                            const score = this._calculateMarkerScore(line, legendMarker);
                            if (score > bestScore) {
                                bestScore = score;
                                bestMatch = legendMarker;
                            }
                        }

                        // Threshold for score?
                        if (bestMatch && bestScore > 0.6) { // Heuristic threshold
                            line.element.setAttribute('lc_legend_ref', bestMatch.id);
                            unassociatedPaths.delete(line);
                            matchCount++;
                        }
                    }
                }
            }
        }

        // --- Step 2: Line Association (Remaining unassociated paths) ---
        for (const line of unassociatedPaths) {
            const lineColor = this._normalizeColor(line.color);
            let bestMatch = null;
            let minDistance = Infinity;

            for (const legendLine of legendLines) {
                const legendColor = this._normalizeColor(legendLine.color);
                const dist = this._getColorDistance(lineColor, legendColor);
                if (dist < 60 && dist < minDistance) {
                    minDistance = dist;
                    bestMatch = legendLine;
                }
            }

            if (bestMatch) {
                line.element.setAttribute('lc_legend_ref', bestMatch.id);
                matchCount++;
            }
        }

        console.log(`[LOG] Associated ${matchCount} data lines with legend items`);
    }

    _initializePaper() {
        if (typeof paper !== 'undefined' && !paper.project) {
            const canvas = document.createElement('canvas');
            paper.setup(canvas);
        }
    }

    _isMarkerCandidate(line) {
        // Check if the path looks like a marker based on bounding box
        // Logic: small size, aspect ratio ~1
        const box = line.box; // Assuming line has {element, box, color}
        if (!box) return false;

        const MAX_MARKER_SIZE = 30; // 30px
        if (box.width > MAX_MARKER_SIZE || box.height > MAX_MARKER_SIZE) return false;

        const ratio = box.width / box.height;
        if (ratio < 0.5 || ratio > 2.0) return false;

        return true;
    }

    _calculateMarkerScore(line, legendMarker) {
        if (!legendMarker.symbolElement) return 0;

        // 1. Color Score
        const lineColor = this._normalizeColor(line.color);
        const legendColor = this._normalizeColor(legendMarker.color);
        const colorDist = this._getColorDistance(lineColor, legendColor);
        const colorScore = Math.max(0, 1 - (colorDist / 100)); // Normalize 0-100 diff to 1-0 score

        // 2. Shape Score using Paper.js
        let shapeScore = 0;
        try {
            const pathData1 = line.element.getAttribute('d');
            const pathData2 = legendMarker.symbolElement.getAttribute('d');

            if (pathData1 && pathData2) {
                const p1 = new paper.Path(pathData1);
                const p2 = new paper.Path(pathData2);

                // Normalize positions to (0,0)
                p1.position = new paper.Point(0, 0);
                p2.position = new paper.Point(0, 0);

                // Normalize size - Scale to fit in 20x20 box
                const bounds1 = p1.bounds;
                const bounds2 = p2.bounds;

                const scale1 = 20 / Math.max(bounds1.width, bounds1.height);
                const scale2 = 20 / Math.max(bounds2.width, bounds2.height);

                p1.scale(scale1);
                p2.scale(scale2);

                // Compare Area
                const area1 = Math.abs(p1.area);
                const area2 = Math.abs(p2.area);
                const areaRatio = Math.min(area1, area2) / (Math.max(area1, area2) + 0.0001);

                // Additional check: Intersection Area (IoU logic simplified)
                // Paper.js boolean operations might be heavy/unstable, let's stick to area + perimeter properties for now
                // Or try simple boolean intersection if possible
                // let intersectArea = 0;
                // try {
                //      const intersection = p1.intersect(p2);
                //      intersectArea = Math.abs(intersection.area);
                //      intersection.remove();
                // } catch(e) {}
                // const iou = intersectArea / (area1 + area2 - intersectArea);

                // Use simple property comparison for robustness
                // Compare perimeter
                const len1 = p1.length;
                const len2 = p2.length;
                const lenRatio = Math.min(len1, len2) / (Math.max(len1, len2) + 0.0001);

                shapeScore = (areaRatio * 0.5) + (lenRatio * 0.5);

                p1.remove();
                p2.remove();
            }
        } catch (e) {
            console.warn("Shape scoring failed", e);
        }

        // Weighted mix
        return (colorScore * 0.4) + (shapeScore * 0.6);
    }

    _getColorDistance(c1, c2) {
        if (!c1 || !c2 || !c1.startsWith('#') || !c2.startsWith('#')) return Infinity;
        const r1 = parseInt(c1.substring(1, 3), 16);
        const g1 = parseInt(c1.substring(3, 5), 16);
        const b1 = parseInt(c1.substring(5, 7), 16);
        const r2 = parseInt(c2.substring(1, 3), 16);
        const g2 = parseInt(c2.substring(3, 5), 16);
        const b2 = parseInt(c2.substring(5, 7), 16);
        return Math.sqrt(Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(b1 - b2, 2));
    }

    _normalizeColor(color) {
        if (!color) return '';
        color = color.trim().toLowerCase();
        if (color.startsWith('#')) {
            if (color.length === 4) {
                return '#' + color[1] + color[1] + color[2] + color[2] + color[3] + color[3];
            }
            return color;
        }
        const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (rgbMatch) {
            const r = parseInt(rgbMatch[1]).toString(16).padStart(2, '0');
            const g = parseInt(rgbMatch[2]).toString(16).padStart(2, '0');
            const b = parseInt(rgbMatch[3]).toString(16).padStart(2, '0');
            return `#${r}${g}${b}`;
        }
        return color;
    }

    /**
     * Calculates the perceived luminance of a color.
     * @param {string} color - The color in any valid CSS format (hex, rgb, name).
     * @returns {number} A luminance value from 0 (black) to 255 (white).
     * @private
     */
    _getColorLuminance(color) {
        const normalizedColor = this._normalizeColor(color);
        // Default to white (max luminance) for invalid or transparent colors
        if (!normalizedColor || normalizedColor === 'none' || !normalizedColor.startsWith('#') || normalizedColor.length !== 7) {
            return 255;
        }

        const r = parseInt(normalizedColor.substring(1, 3), 16);
        const g = parseInt(normalizedColor.substring(3, 5), 16);
        const b = parseInt(normalizedColor.substring(5, 7), 16);

        // Standard luminance formula (perceived brightness)
        // It weights green highest, then red, then blue, mimicking human vision.
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b);

        return luminance;
    }


    enableLegendInteractivity() {
        console.log("[LOG] SVGXLineChartRendering: Enabling legend interactivity...");
        const legendLabels = Array.from(this.svg.querySelectorAll('text[lc_legend_id]'));
        legendLabels.forEach(label => {
            label.style.cursor = 'pointer';
            label.addEventListener('click', () => {
                const legendId = label.getAttribute('lc_legend_id');
                console.log(`[LOG] Legend clicked: ${legendId}`);
                this._flashDataLines(legendId);
            });
        });
    }

    _flashDataLines(legendId) {
        const lines = Array.from(this.svg.querySelectorAll(`path[lc_legend_ref="${legendId}"]`));
        if (lines.length === 0) return;
        console.log(`[LOG] Flashing ${lines.length} lines for legend ${legendId}`);
        lines.forEach(line => {
            const color = this._extractPathColor(line);
            const originalStroke = line.style.stroke;
            const originalStrokeWidth = line.style.strokeWidth || line.getAttribute('stroke-width') || '1';
            const originalOpacity = line.style.opacity || '1';
            const originalFill = line.style.fill;
            line.style.stroke = 'red';
            line.style.transition = 'all 0.2s ease-in-out';
            line.style.strokeWidth = '40px';
            line.style.opacity = '0.5';
            if (line.parentNode) {
                line.parentNode.appendChild(line);
            }
            setTimeout(() => {
                line.style.opacity = '1';
            }, 100);
            setTimeout(() => {
                line.style.strokeWidth = originalStrokeWidth;
                line.style.stroke = originalStroke;
                line.style.opacity = originalOpacity;
                setTimeout(() => {
                    line.style.transition = '';
                }, 200);
            }, 600);
        });
    }
}