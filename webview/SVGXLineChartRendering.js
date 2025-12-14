///
/// Line Chart Rendering using D3.js
///

class SVGXLineChartRendering {
    constructor(svgElement, clrContent = null) {
        this.svg = svgElement;
        if (!this.svg) {
            throw new Error('SVG element is not defined.');
        }
        // Parse the .clr file content into a Set for efficient lookups
        this.officialPalette = clrContent ? new Set(clrContent.trim().split(/\r?\n/).map(c => this._normalizeColor(c))) : null;
        console.log('[LOG] Line chart renderer initialized');
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
            }

            // 5. Y-axis Final Result
            if (yPairs.length < 2) {
                console.warn("[WARN] Could not find 2 valid Y-axis pairs.");
            } else {
                yPairs.sort((a, b) => a.logical - b.logical);
                const y1 = yPairs[0];
                const y2 = yPairs[yPairs.length - 1];
                const ylm_string = `[${y1.logical.toFixed(2)}, ${y2.logical.toFixed(2)}, ${y1.visual.toFixed(2)}, ${y2.visual.toFixed(2)}]`;
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
                console.warn("[WARN] Could not find 2 valid X-axis pairs.");
            } else {
                xPairs.sort((a, b) => a.logical - b.logical);
                const x1 = xPairs[0];
                const x2 = xPairs[xPairs.length - 1];
                const xlm_string = `[${x1.logical.toFixed(2)}, ${x2.logical.toFixed(2)}, ${x1.visual.toFixed(2)}, ${x2.visual.toFixed(2)}]`;
                this.svg.setAttribute("xlm", xlm_string);
                // this._highlightMatches(xPairs, 'orange');
                console.log(`[LOG] Final result: xlm=${xlm_string}`);
            }

        } catch (error) {
            console.error("[ERROR] Failed to add logical mapping:", error);
        }
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
        const MAX_TICK_WIDTH = svgWidth * 0.05;
        const MIN_TICK_WIDTH = 3;
        const MAX_TICK_HEIGHT = 20;
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
        const matched = [];
        for (let i = 0; i < gridLines.length; i++) {
            const line = gridLines[i];
            let bestLabel = null;
            let bestDistance = Infinity;
            for (let j = 0; j < labels.length; j++) {
                const label = labels[j];
                const dx = label.x - line.x;
                const dy = label.y - line.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }
            if (bestDistance <= 100 && bestLabel !== null) {
                matched.push({
                    logical: bestLabel.value,
                    visual: line.linePos,
                    distance: bestDistance,
                    labelText: bestLabel.text,
                    labelElement: bestLabel.element,
                    gridLineElement: line.element
                });
            }
        }
        matched.sort((a, b) => a.distance - b.distance);
        const result = [];
        for (let i = 0; i < matched.length && i < 2; i++) {
            result.push(matched[i]);
        }
        return result;
    }

    _matchYAxisLeftLabels(gridLines, labels) {
        if (gridLines.length < 2 || labels.length < 2) return [];
        const matched = [];
        for (let i = 0; i < gridLines.length; i++) {
            const line = gridLines[i];
            let bestLabel = null;
            let bestDistance = Infinity;
            for (let j = 0; j < labels.length; j++) {
                const label = labels[j];
                const dx = label.x - line.x;
                const dy = label.y - line.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }
            if (bestDistance <= 100 && bestLabel !== null) {
                matched.push({
                    logical: bestLabel.value,
                    visual: line.linePos,
                    distance: bestDistance,
                    labelText: bestLabel.text,
                    labelElement: bestLabel.element,
                    gridLineElement: line.element
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
            const scaleX = svgViewBox.width / svgRect.width;
            const scaleY = svgViewBox.height / svgRect.height;
            const elemRect = element.getBoundingClientRect();
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
                // BUG 2 FIX Part B: Sort candidates by area to pick the inner symbol
                candidates.sort((a, b) => a.dist - b.dist);
                const minDist = candidates[0].dist;
                const closeCandidates = candidates.filter(c => c.dist <= minDist + 5);
                closeCandidates.sort((a, b) => a.area - b.area);
                let bestCandidate = closeCandidates[0];
                bestSymbol = bestCandidate.element;

                // Determine type based on aspect ratio and size
                const ratio = bestCandidate.width / bestCandidate.height;
                // Markers typically have aspect ratio close to 1 and are small?
                // Or maybe just based on size?
                // Let's use the logic from the plan:
                // Marker: small size (< 30px?), aspect ratio 0.5 < r < 2.0
                if (bestCandidate.width < 30 && bestCandidate.height < 30 && ratio > 0.5 && ratio < 2.0) {

                    bestCandidate = candidates[0];
                    bestSymbol = bestCandidate.element;

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

    // ==============================================================================
    // Data Extraction and Redrawing
    // ==============================================================================

    /**
     * Extracts logical data for each trace/legend item using xlm and ylm mappings.
     * Preserves individual path topology (main line vs. isolated markers/segments).
     * @returns {Array} Array of series objects with id, name, type, style, traces, and combined data points
     */
    extractLogicalData() {
        console.log("[LOG] SVGXLineChartRendering: Extracting logical data (Trace-Based)...");

        const xlmAttr = this.svg.getAttribute('xlm');
        const ylmAttr = this.svg.getAttribute('ylm');

        if (!xlmAttr || !ylmAttr) {
            console.warn("[WARN] Missing xlm or ylm attributes. Cannot extract logical data.");
            return [];
        }

        // Parse xlm/ylm using a helper to handle both JSON and CSV
        const parseMapping = (attr) => {
            if (!attr) return null;
            try {
                return JSON.parse(attr);
            } catch (e) {
                return attr.split(',').map(Number);
            }
        };

        const xMapping = parseMapping(xlmAttr);
        const yMapping = parseMapping(ylmAttr);

        if (!xMapping || xMapping.length !== 4 || !yMapping || yMapping.length !== 4) {
            console.warn("[WARN] Invalid xlm or ylm attributes.");
            return [];
        }

        console.log(`[LOG] X Mapping: logical=[${xMapping[0]}, ${xMapping[1]}], visual=[${xMapping[2]}, ${xMapping[3]}]`);
        console.log(`[LOG] Y Mapping: logical=[${yMapping[0]}, ${yMapping[1]}], visual=[${yMapping[2]}, ${yMapping[3]}]`);

        if (!this.legendItems || this.legendItems.length === 0) {
            console.warn("[WARN] Legend items not processed. Run addLegendInfo() first.");
            return [];
        }

        const extractedSeries = [];

        this.legendItems.forEach(legendItem => {
            const seriesId = legendItem.id;
            const seriesName = legendItem.text;
            const seriesType = legendItem.lc_legend_type;

            // Find all paths associated with this legend item
            const associatedPaths = Array.from(this.svg.querySelectorAll(`path[lc_legend_ref="${seriesId}"]`));

            if (associatedPaths.length === 0) {
                console.log(`[LOG] No paths found for series: ${seriesName}`);
                return;
            }

            console.log(`[LOG] Processing series: ${seriesName} (${associatedPaths.length} paths)`);

            // Extract style from the first path (as a baseline)
            const baseStyle = this._extractTraceStyle(associatedPaths[0]);

            const seriesTraces = [];
            const allLogicalPoints = [];

            associatedPaths.forEach(path => {
                const points = this._extractPointsFromPath(path);
                if (points.length === 0) return;

                // 1. Map points to Logical Coordinates
                // toLogicalX: dx_min + (vx - vx_min) * (dx_max - dx_min) / (vx_max - vx_min)
                // toLogicalY: dy_min + (vy - vy_min) * (dy_max - dy_min) / (vy_max - vy_min)
                const logicalPoints = points.map(pt => {
                    return {
                        x: this._toLogicalX(pt.x, xMapping[0], xMapping[1], xMapping[2], xMapping[3]),
                        y: this._toLogicalY(pt.y, yMapping[0], yMapping[1], yMapping[2], yMapping[3]),
                        isMarker: pt.isMarker // Preserve marker flag from extraction
                    };
                });

                // 2. Simplify/Clean points slightly to remove potrace micro-noise?
                // For now, let's keep raw resolution to preserve curve fidelity, 
                // but if points are indentical, d3 might complain? No, it's fine.
                // We'll perform a lightweight dedup.
                const cleanPoints = this._removeDuplicatePoints(logicalPoints, 0.000001);

                // Check if this path was marked as a marker during the detection phase
                const isMarker = path.hasAttribute('lc_is_marker') && path.getAttribute('lc_is_marker') === 'true';

                seriesTraces.push({
                    points: cleanPoints,
                    style: this._extractTraceStyle(path), // Specific style for this trace
                    isMarker: isMarker,
                    centroid: isMarker ? this._computeCentroid(cleanPoints) : null
                });

                allLogicalPoints.push(...cleanPoints);
            });

            // Override baseStyle stroke with legend color to ensure consistency with Legend Symbol
            if (legendItem.color) {
                baseStyle.stroke = legendItem.color;
            }

            extractedSeries.push({
                id: seriesId,
                name: seriesName,
                type: seriesType,
                style: baseStyle,
                traces: seriesTraces,
                data: allLogicalPoints // Flattened list for Axis Extents
            });

            console.log(`[LOG] Series "${seriesName}": ${seriesTraces.length} traces extracted.`);
        });

        console.log(`[LOG] Extracted ${extractedSeries.length} series total.`);

        // --- Correct Axis Range Calculation ---
        // 1. Determine Visual Chart Bounds (Plot Area)
        // We can infer this from the min/max coordinates of the data points we found,
        // OR better, from the gridlines if we can find them again.
        // Let's use the extent of the extracted data points + finding the main axis lines.

        // Helper to find range from Ticks (primary) or Gridlines (secondary)
        const findAxisDomains = () => {
            const svgBox = this.svg.viewBox.baseVal;
            const svgWidth = svgBox.width || 800;
            const svgHeight = svgBox.height || 600;

            // 1. Get X-Axis Ticks (Bottom Labels)
            // We can reuse the filter logic from _findXAxisBottomLabels but we need the VISUAL min/max
            const xLabels = this._findXAxisBottomLabels(svgHeight);

            let xDom = null;
            if (xLabels && xLabels.length >= 2) {
                // Sort by X position
                xLabels.sort((a, b) => a.x - b.x);
                const firstTick = xLabels[0];
                const lastTick = xLabels[xLabels.length - 1];

                // Map visual X -> Logical X
                const l1 = this._toLogicalX(firstTick.x, xMapping[0], xMapping[1], xMapping[2], xMapping[3]);
                const l2 = this._toLogicalX(lastTick.x, xMapping[0], xMapping[1], xMapping[2], xMapping[3]);
                xDom = [Math.min(l1, l2), Math.max(l1, l2)];
                console.log(`[LOG] Found X-Axis Ticks: First=[x:${firstTick.x.toFixed(1)}, val:${firstTick.value}], Last=[x:${lastTick.x.toFixed(1)}, val:${lastTick.value}] -> Dom: ${xDom}`);
            } else {
                // Fallback: Gridlines
                console.log("[LOG] No X-Ticks found, checking gridlines...");
                const xGrid = this._findXAxisGridlines(svgWidth, svgHeight, 0.4); // 40% height min
                if (xGrid && xGrid.length >= 2) {
                    const minX = Math.min(...xGrid.map(g => g.x));
                    const maxX = Math.max(...xGrid.map(g => g.x));
                    const l1 = this._toLogicalX(minX, xMapping[0], xMapping[1], xMapping[2], xMapping[3]);
                    const l2 = this._toLogicalX(maxX, xMapping[0], xMapping[1], xMapping[2], xMapping[3]);
                    xDom = [Math.min(l1, l2), Math.max(l1, l2)];
                }
            }

            // 2. Get Y-Axis Ticks (Left Labels)
            const yLabels = this._findYAxisLeftLabels(svgWidth);

            let yDom = null;
            if (yLabels && yLabels.length >= 2) {
                yLabels.sort((a, b) => a.y - b.y); // Top to Bottom
                const firstTick = yLabels[0]; // Topmost (lowest visual Y, highest Logical usually)
                const lastTick = yLabels[yLabels.length - 1]; // Bottommost

                const l1 = this._toLogicalY(firstTick.y, yMapping[0], yMapping[1], yMapping[2], yMapping[3]);
                const l2 = this._toLogicalY(lastTick.y, yMapping[0], yMapping[1], yMapping[2], yMapping[3]);
                yDom = [Math.min(l1, l2), Math.max(l1, l2)];
                console.log(`[LOG] Found Y-Axis Ticks: Top=[y:${firstTick.y.toFixed(1)}, val:${firstTick.value}], Bottom=[y:${lastTick.y.toFixed(1)}, val:${lastTick.value}] -> Dom: ${yDom}`);
            } else {
                // Fallback: Use Mapping anchors themselves if no ticks found?
                // Let's just use the tick locations from the MAPPING as the range.
                const l1 = yMapping[0];
                const l2 = yMapping[1];
                yDom = [Math.min(l1, l2), Math.max(l1, l2)];
            }

            // Final Safety Check
            if (!xDom) {
                // Fallback to the mapping's own ticks range
                xDom = [Math.min(xMapping[0], xMapping[1]), Math.max(xMapping[0], xMapping[1])];
            }





            // Calculate Tick Counts
            const xCount = xLabels ? xLabels.length : (xGrid ? xGrid.length : 0);
            const yCount = yLabels ? yLabels.length : 0;

            return {
                x: xDom,
                y: yDom,
                xTickCount: xCount,
                yTickCount: yCount,
                // Return Visual Bounds for Label Detection
                visualX: xLabels && xLabels.length >= 2
                    ? [Math.min(xLabels[0].x, xLabels[xLabels.length - 1].x), Math.max(xLabels[0].x, xLabels[xLabels.length - 1].x)]
                    : (xGrid && xGrid.length >= 2 ? [Math.min(...xGrid.map(g => g.x)), Math.max(...xGrid.map(g => g.x))] : null),
                visualY: yLabels && yLabels.length >= 2
                    ? [Math.min(yLabels[0].y, yLabels[yLabels.length - 1].y), Math.max(yLabels[0].y, yLabels[yLabels.length - 1].y)]
                    : null
            };
        };

        const domains = findAxisDomains();

        const axes = {
            x: {
                domain: domains.x,
                label: this._findAxisTitle('x', domains.visualX, domains.visualY),
                tickCount: domains.xTickCount
            },
            y: {
                domain: domains.y,
                label: this._findAxisTitle('y', domains.visualX, domains.visualY),
                tickCount: domains.yTickCount
            }
        };

        const titleAndFooter = this._findChartTitleAndFooter();


        // Requested Console Logs
        console.log("X info:", JSON.stringify(axes.x));
        console.log("Y info:", JSON.stringify(axes.y));

        return { series: extractedSeries, axes: axes, title: titleAndFooter.title, footer: titleAndFooter.footer };
    }

    _intersectsLegend(bbox) {
        if (!this.legendItems) return false;
        // Simple check if bbox overlaps with the legend detect area (if we stored it)
        // For now, assume false or check against a known legend box if we had one.
        return false;
    }



    /**
     * Heuristically finds the axis title.
     * Updated to look relative to the chart area.
     * @param {string} axis - 'x' or 'y'
     * @param {Array} xRange - [min, max] visual X of chart
     * @param {Array} yRange - [min, max] visual Y of chart
     * @returns {string} The text content of the found title or empty string.
     */
    _findAxisTitle(axis, xRange, yRange) {
        const texts = Array.from(this.svg.querySelectorAll('text'));

        // Default to viewbox if ranges undefined
        const vb = this.svg.viewBox.baseVal;
        const minX = xRange ? xRange[0] : 0;
        const maxX = xRange ? xRange[1] : vb.width;
        const minY = yRange ? yRange[0] : 0;
        const maxY = yRange ? yRange[1] : vb.height;

        let candidate = "";
        let maxScore = 0;

        texts.forEach(t => {
            const text = t.textContent.trim();
            if (!text || !isNaN(parseFloat(text.replace(/[$,%]/g, '')))) return;
            if (this.legendItems && this.legendItems.some(item => item.text === text)) return;

            const bbox = this._getFullBBox(t);
            if (!bbox) return;

            let score = 0;

            if (axis === 'x') {
                // X-Axis Title: Below the chart bottom (maxY), centered horizontally
                if (bbox.y > maxY - 10 && bbox.y < maxY + 100) score += 10;
                // Centered within the chart width?
                const chartCenter = minX + (maxX - minX) / 2;
                const dist = Math.abs(bbox.cx - chartCenter);
                if (dist < (maxX - minX) * 0.2) score += 5;
            } else {
                // Y-Axis Title: To the left of chart left (minX), vertically centered
                if (bbox.x < minX + 10) score += 10;
                // Centered val
                const chartCenterY = minY + (maxY - minY) / 2;
                const distY = Math.abs(bbox.cy - chartCenterY);
                if (distY < (maxY - minY) * 0.3) score += 5;

                const transform = t.getAttribute('transform');
                if (transform && transform.includes('rotate')) score += 5;
            }

            if (score > 8) {
                if (score > maxScore) {
                    maxScore = score;
                    candidate = text;
                }
            }
        });

        return candidate;
    }

    /**
     * Finds the chart title and footer, handling multiple lines.
     * @returns {{title: {text: string, x: number, y: number, width: number, height: number}[], footer: {text: string, x: number, y: number, width: number, height: number}[]}}
     */
    _findChartTitleAndFooter() {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const vb = this.svg.viewBox.baseVal;
        const svgHeight = vb.height || 864;
        const svgWidth = vb.width || 1536;

        const titleCandidates = [];
        const footerCandidates = [];

        // 1. Filter out numeric labels, legend items, and axis labels
        const numericLabels = this._findNumericLabels();
        const legendItemTexts = this.legendItems ? this.legendItems.map(item => item.text) : [];
        const allTexts = texts.map(t => {
            const bbox = this._getFullBBox(t);
            return {
                element: t,
                text: t.textContent.trim(),
                bbox: bbox,
            };
        }).filter(t => {
            if (!t.text || !t.bbox) return false;
            // Filter out purely numeric text (like axis ticks)
            if (/^[\d.,\-%$]+$/.test(t.text)) return false;
            // Filter out legend items
            if (legendItemTexts.includes(t.text)) return false;
            // Filter out previously identified axis ticks by checking against the numericLabels list
            if (numericLabels.some(nl => nl.text === t.text)) return false;

            return true;
        });


        // 2. Identify Title and Footer candidates by position
        const titleYThreshold = svgHeight * 0.20; // Top 20% - Increased
        const footerYThreshold = svgHeight * 0.85; // Bottom 15%

        allTexts.forEach(t => {
            if (t.bbox.y < titleYThreshold) {
                titleCandidates.push(t);
            } else if (t.bbox.y > footerYThreshold) {
                footerCandidates.push(t);
            }
        });

        // 3. Group and Sort Candidates
        const groupLines = (candidates) => {
            if (candidates.length === 0) return [];

            // Sort by vertical position first, then horizontal
            candidates.sort((a, b) => {
                if (Math.abs(a.bbox.y - b.bbox.y) > 5) {
                    return a.bbox.y - b.bbox.y;
                }
                return a.bbox.x - b.bbox.x;
            });

            const grouped = [];
            if (candidates.length === 0) {
                return [];
            }

            let currentLine = [candidates[0]];

            for (let i = 1; i < candidates.length; i++) {
                const prevInLine = currentLine[0];
                const curr = candidates[i];
                const avgHeight = (prevInLine.bbox.height + curr.bbox.height) / 2;

                // If the vertical distance is less than ~75% of the average height, consider it the same line.
                // This handles slight vertical drift within a line of text.
                if (Math.abs(curr.bbox.y - prevInLine.bbox.y) < avgHeight * 0.75) {
                    currentLine.push(curr);
                } else {
                    // New line found, push the completed line
                    grouped.push(currentLine);
                    currentLine = [curr]; // Start a new line
                }
            }
            // Push the very last line
            if (currentLine.length > 0) {
                grouped.push(currentLine);
            }

            // Join text fragments on the same line and return final structured lines
            return grouped.map(line => {
                // sort fragments by x before joining
                line.sort((a, b) => a.bbox.x - b.bbox.x);
                const text = line.map(l => l.text).join(' ');
                const x = Math.min(...line.map(l => l.bbox.x));
                const y = line[0].bbox.y;
                const width = Math.max(...line.map(l => l.bbox.x + l.bbox.width)) - x;
                const height = Math.max(...line.map(l => l.bbox.height));
                return { text, x, y, width, height };
            });
        };

        const titleLines = groupLines(titleCandidates);
        const footerLines = groupLines(footerCandidates);

        console.log(`[LOG] Found ${titleLines.length} title lines.`);
        console.log(`[LOG] Found ${footerLines.length} footer lines.`);

        return { title: titleLines, footer: footerLines };
    }

    _computeCentroid(points) {
        if (!points || points.length === 0) return { x: 0, y: 0 };
        let sx = 0, sy = 0;
        points.forEach(p => { sx += p.x; sy += p.y; });
        return { x: sx / points.length, y: sy / points.length };
    }

    /**
     * Convert visual X coordinate to logical X value.
     * Formula: dx_min + (vx - vx_min) * (dx_max - dx_min) / (vx_max - vx_min)
     */
    _toLogicalX(vx, dx_min, dx_max, vx_min, vx_max) {
        if (vx_max === vx_min) {
            return dx_min;
        }
        return dx_min + (vx - vx_min) * (dx_max - dx_min) / (vx_max - vx_min);
    }

    /**
     * Convert visual Y coordinate to logical Y value.
     * Formula: dy_min + (vy - vy_min) * (dy_max - dy_min) / (vy_max - vy_min)
     */
    _toLogicalY(vy, dy_min, dy_max, vy_min, vy_max) {
        if (vy_min === vy_max) {
            return dy_min;
        }
        return dy_min + (vy - vy_min) * (dy_max - dy_min) / (vy_max - vy_min);
    }

    /**
     * OPTIMIZED Vertical Scanline Decomposition.
     * Fixes performance by reusing objects and increasing scan step.
     * Fixes "missing markers" by tuning the threshold sensitivity.
     */
    _extractPointsFromPath(pathElement) {
        // 1. Setup Transforms
        let ctm = null;
        let inverseRootCTM = null;
        try {
            ctm = pathElement.getScreenCTM();
            const rootCTM = this.svg.getScreenCTM();
            if (rootCTM) {
                inverseRootCTM = rootCTM.inverse();
            }
        } catch (e) {
            return [];
        }

        const pt = this.svg.createSVGPoint();

        // --- NEW SECTION: Get Legend Marker Width ---
        let legendW = 15;
        try {
            const legendId = pathElement.getAttribute('lc_legend_ref');
            if (legendId && this.legendItems) {
                const item = this.legendItems.find(i => i.id === legendId);
                if (item && item.symbolElement) {
                    const bbox = item.symbolElement.getBBox();
                    if (bbox.width > 0) legendW = bbox.width;
                }
            }
        } catch (e) { }

        // LOGIC: Enforce split if marker group is wider than Legend OR 8px (whichever is smaller)
        const splitLimit = Math.min(legendW, 13.0);

        // 2. Sample the Path (The Tube Scan)
        const len = pathElement.getTotalLength();
        if (len === 0) return [];

        const buckets = new Map();

        // Keep Step = 4.0 as requested
        const step = 4.0;

        for (let i = 0; i < len; i += step) {
            const rawPt = pathElement.getPointAtLength(i);

            pt.x = rawPt.x;
            pt.y = rawPt.y;

            let transformedPt = pt;
            if (ctm && inverseRootCTM) {
                transformedPt = pt.matrixTransform(ctm).matrixTransform(inverseRootCTM);
            }

            const key = Math.round(transformedPt.x);

            if (!buckets.has(key)) {
                buckets.set(key, { min: transformedPt.y, max: transformedPt.y });
            } else {
                const b = buckets.get(key);
                if (transformedPt.y < b.min) b.min = transformedPt.y;
                if (transformedPt.y > b.max) b.max = transformedPt.y;
            }
        }

        // 3. Convert to Array (Raw Vertical Heights)
        let rawPoints = [];
        const sortedKeys = Array.from(buckets.keys()).sort((a, b) => a - b);

        sortedKeys.forEach(x => {
            const b = buckets.get(x);
            const vHeight = Math.abs(b.max - b.min);
            const midY = (b.min + b.max) / 2;
            if (vHeight > 0.1) {
                rawPoints.push({ x: x, y: midY, vHeight: vHeight });
            }
        });

        if (rawPoints.length === 0) return [];

        // 4. Perpendicular Thickness Calculation
        // Modified to use Trend Slope + Cosine Correction
        const processedPoints = [];

        for (let i = 0; i < rawPoints.length; i++) {
            const p = rawPoints[i];

            // Handle Start/End neighbors explicitly to avoid Index Out Of Bounds
            let idx1, idx2;
            if (i === 0) {
                idx1 = 0; idx2 = Math.min(rawPoints.length - 1, 1);
            } else if (i === rawPoints.length - 1) {
                idx1 = Math.max(0, i - 1); idx2 = i;
            } else {
                idx1 = i - 1; idx2 = i + 1;
            }

            const p1 = rawPoints[idx1];
            const p2 = rawPoints[idx2];
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;

            let slope = 0;
            if (dx !== 0) slope = dy / dx;

            // T = H * cos(theta)
            const cosTheta = 1 / Math.sqrt(1 + slope * slope);
            const trueThickness = p.vHeight * cosTheta;

            processedPoints.push({
                x: p.x,
                y: p.y,
                thickness: trueThickness
            });
        }

        // 5. Statistics & Safety Caps
        const widths = processedPoints.map(p => p.thickness).filter(w => w > 0.1);
        widths.sort((a, b) => a - b);
        const midIdx = Math.floor(widths.length / 2);

        let averageThickness = widths.length > 0 ? widths[midIdx] : 1.0;

        // CAP 1: Force Line Baseline (Lines are thin)
        if (averageThickness > 3.0) averageThickness = 3.0;

        // CAP 2: Dynamic Threshold
        let thresholdVal = 1.5;
        const maxThreshold = averageThickness * 1.5;
        if (thresholdVal > maxThreshold) thresholdVal = maxThreshold;

        const markerThreshold = averageThickness + thresholdVal;

        // 6. Separation Logic with SPLIT LIMIT
        const finalPoints = [];
        let inMarkerGroup = false;
        let markerGroup = [];

        for (let i = 0; i < processedPoints.length; i++) {
            const p = processedPoints[i];

            if (p.thickness > markerThreshold) {
                // Marker Candidate
                if (!inMarkerGroup) {
                    inMarkerGroup = true;
                    markerGroup = [p];
                } else {
                    // CHECK SPLIT LOGIC
                    const startX = markerGroup[0].x;
                    const currentWidth = p.x - startX;

                    if (currentWidth > splitLimit) {
                        // Force Split: Save current group, start new one
                        const avgX = markerGroup.reduce((s, m) => s + m.x, 0) / markerGroup.length;
                        const avgY = markerGroup.reduce((s, m) => s + m.y, 0) / markerGroup.length;
                        finalPoints.push({ x: avgX, y: avgY, isMarker: true });

                        markerGroup = [p];
                    } else {
                        markerGroup.push(p);
                    }
                }
            } else {
                // Line
                if (inMarkerGroup) {
                    if (markerGroup.length > 0) {
                        const avgX = markerGroup.reduce((s, m) => s + m.x, 0) / markerGroup.length;
                        const avgY = markerGroup.reduce((s, m) => s + m.y, 0) / markerGroup.length;
                        finalPoints.push({ x: avgX, y: avgY, isMarker: true });
                    }
                    inMarkerGroup = false;
                    markerGroup = [];
                }
                finalPoints.push({ x: p.x, y: p.y, isMarker: false });
            }
        }

        // Handle end of line
        if (inMarkerGroup && markerGroup.length > 0) {
            const avgX = markerGroup.reduce((s, m) => s + m.x, 0) / markerGroup.length;
            const avgY = markerGroup.reduce((s, m) => s + m.y, 0) / markerGroup.length;
            finalPoints.push({ x: avgX, y: avgY, isMarker: true });
        }

        return finalPoints;
    }

    /**
     * Removes duplicate points within a tolerance.
     * @param {Array<{x: number, y: number}>} points 
     * @param {number} tolerance 
     * @returns {Array<{x: number, y: number}>}
     */
    _removeDuplicatePoints(points, tolerance) {
        if (points.length === 0) return [];

        const unique = [points[0]];
        for (let i = 1; i < points.length; i++) {
            const prev = unique[unique.length - 1];
            const curr = points[i];
            const dx = Math.abs(curr.x - prev.x);
            const dy = Math.abs(curr.y - prev.y);
            if (dx > tolerance || dy > tolerance) {
                unique.push(curr);
            }
        }
        return unique;
    }

    /**
     * Extracts visual style properties from a path element.
     * @param {SVGPathElement} pathElement 
     * @returns {Object} Style object with stroke, strokeWidth, etc.
     */
    _extractTraceStyle(pathElement) {
        const computed = window.getComputedStyle(pathElement);

        let stroke = pathElement.getAttribute('stroke');
        if (!stroke || stroke === 'none') stroke = pathElement.style.stroke;
        if (!stroke || stroke === 'none') stroke = computed.stroke;

        let fill = pathElement.getAttribute('fill');
        if (!fill) fill = pathElement.style.fill;
        if (!fill) fill = computed.fill;

        let strokeWidth = pathElement.getAttribute('stroke-width');
        if (!strokeWidth) strokeWidth = pathElement.style.strokeWidth;
        if (!strokeWidth) strokeWidth = computed.strokeWidth;

        let strokeDasharray = pathElement.getAttribute('stroke-dasharray');
        if (!strokeDasharray) strokeDasharray = pathElement.style.strokeDasharray;
        if (!strokeDasharray) strokeDasharray = computed.strokeDasharray;

        let strokeOpacity = pathElement.getAttribute('stroke-opacity');
        if (!strokeOpacity) strokeOpacity = pathElement.style.strokeOpacity;
        if (!strokeOpacity) strokeOpacity = computed.strokeOpacity;

        return {
            stroke: stroke || '#000000',
            fill: fill || 'none',
            strokeWidth: strokeWidth || '1',
            strokeDasharray: strokeDasharray || 'none',
            strokeOpacity: strokeOpacity || '1'
        };
    }

    /**
     * Backwards compatible wrapper for extractTraceStyles
     */
    extractTraceStyles(pathElement) {
        return this._extractTraceStyle(pathElement);
    }

    /**
     * Redraws the chart using D3.js to match the original as closely as possible.
     * Use Trace-Based rendering to preserve disjoint paths and markers.
     * @param {Object|Array} input - Object { series, axes } from extractLogicalData() OR array of series.
     * @param {string} containerSelector - CSS selector for the container element
     */
    redrawChart(input, containerSelector) {
        console.log(`[LOG] Redrawing chart into ${containerSelector}...`);

        let chartData = [];
        let axesInfo = null;
        let title = [];
        let footer = [];

        // Handle both old format (array) and new format (object)
        if (Array.isArray(input)) {
            chartData = input;
        } else if (input && input.series) {
            chartData = input.series;
            axesInfo = input.axes;
            title = input.title || [];
            footer = input.footer || [];
        }

        const container = document.querySelector(containerSelector);
        if (!container) {
            console.warn(`[WARN] Container ${containerSelector} not found.`);
            return;
        }

        container.innerHTML = ''; // Clear

        if (chartData.length === 0) {
            console.warn("[WARN] No chart data to draw.");
            return;
        }

        // Get original SVG dimensions for reference
        const originalViewBox = this.svg.viewBox.baseVal;
        const originalWidth = originalViewBox.width || 800;
        const originalHeight = originalViewBox.height || 500;

        // Use similar aspect ratio but reasonable size
        const width = Math.min(originalWidth, 1200);
        const aspectRatio = originalHeight / originalWidth;
        const height = width * aspectRatio;

        const titleHeight = title.length * 20;
        const footerHeight = footer.length * 20;

        const margin = {
            top: 40 + titleHeight,
            right: 150,
            bottom: 75 + footerHeight, // Increased bottom margin for footer and legend
            left: 80
        };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select(container).append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", `0 0 ${width} ${height}`)
            .style("background", "#ffffff")
            .style("font-family", "sans-serif");

        // Draw Title
        if (title.length > 0) {
            const titleGroup = svg.append("g")
                .attr("transform", `translate(${width / 2}, 30)`);
            title.forEach((line, i) => {
                titleGroup.append("text")
                    .attr("y", i * 20)
                    .style("text-anchor", "middle")
                    .style("font-size", "16px")
                    .style("font-weight", "bold")
                    .text(line.text);
            });
        }


        // Create chart group with margins
        const chartGroup = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Collect all data points (flattened) to determine fallback scales
        let allPoints = [];
        chartData.forEach(series => {
            if (series.data) allPoints = allPoints.concat(series.data);
        });

        if (allPoints.length === 0) {
            console.warn("[WARN] No data points to plot.");
            return;
        }

        // Determine Axis Domains
        let xDomain, yDomain;

        if (axesInfo) {
            // Use explicit domains extracted from xlm/ylm
            xDomain = axesInfo.x.domain;
            yDomain = axesInfo.y.domain;
            console.log(`[LOG] Using explicit axes: X[${xDomain}], Y[${yDomain}]`);
        } else {
            // Fallback to data extent
            const xExtent = d3.extent(allPoints, d => d.x);
            const yExtent = d3.extent(allPoints, d => d.y);
            // Add some padding to the domain
            const xPadding = (xExtent[1] - xExtent[0]) * 0.02 || (xExtent[0] * 0.1);
            const yPadding = (yExtent[1] - yExtent[0]) * 0.05 || (yExtent[0] * 0.1);
            xDomain = [xExtent[0] - xPadding, xExtent[1] + xPadding];
            yDomain = [yExtent[0] - yPadding, yExtent[1] + yPadding];
        }

        const xScale = d3.scaleLinear()
            .domain(xDomain)
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain(yDomain)
            .range([plotHeight, 0]);

        // Set logical mapping attributes on the new SVG
        const xlm = [xDomain[0], xDomain[1], margin.left, margin.left + plotWidth];
        const ylm = [yDomain[0], yDomain[1], margin.top + plotHeight, margin.top];
        svg.attr("xlm", `[${xlm.join(", ")}]`);
        svg.attr("ylm", `[${ylm.join(", ")}]`);

        // Generate explicitly calculated ticks if interval is provided
        // Get tick counts from extracted data
        const xTickCount = (axesInfo && axesInfo.x.tickCount) ? axesInfo.x.tickCount : null;
        const yTickCount = (axesInfo && axesInfo.y.tickCount) ? axesInfo.y.tickCount : null;

        // Draw grid lines
        const yGridAxis = d3.axisLeft(yScale)
            .tickSize(-plotWidth)
            .tickFormat("");
        if (yTickCount) yGridAxis.ticks(yTickCount);

        chartGroup.append("g")
            .attr("class", "grid")
            .attr("opacity", 0.3)
            .call(yGridAxis)
            .selectAll("line")
            .attr("stroke", "#ccc");

        const xGridAxis = d3.axisBottom(xScale)
            .tickSize(-plotHeight)
            .tickFormat("");
        if (xTickCount) xGridAxis.ticks(xTickCount);

        chartGroup.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0,${plotHeight})`)
            .attr("opacity", 0.3)
            .call(xGridAxis)
            .selectAll("line")
            .attr("stroke", "#ccc");

        // Draw X axis
        const xAxisGen = d3.axisBottom(xScale);
        if (xTickCount) xAxisGen.ticks(xTickCount);

        const xAxis = chartGroup.append("g")
            .attr("transform", `translate(0,${plotHeight})`)
            .call(xAxisGen);
        xAxis.selectAll("line").attr("stroke", "black");
        xAxis.selectAll("path").attr("stroke", "black");
        xAxis.selectAll("text").style("font-size", "12px").style("fill", "black");

        // Draw Y axis
        const yAxisGen = d3.axisLeft(yScale);
        if (yTickCount) yAxisGen.ticks(yTickCount);

        const yAxis = chartGroup.append("g")
            .call(yAxisGen);
        yAxis.selectAll("line").attr("stroke", "black");
        yAxis.selectAll("path").attr("stroke", "black");
        yAxis.selectAll("text").style("font-size", "12px").style("fill", "black");

        // Draw X-Axis Label if present
        if (axesInfo && axesInfo.x.label) {
            console.log(`[LOG] Drawing X-Axis label: ${axesInfo.x.label}`);
            svg.append("text")
                .attr("transform", `translate(${margin.left + plotWidth / 2}, ${height - margin.bottom + 30})`)
                .style("text-anchor", "middle")
                .style("font-size", "14px")
                .text(axesInfo.x.label);
        }

        // Draw Y-Axis Label if present
        if (axesInfo && axesInfo.y.label) {
            console.log(`[LOG] Drawing Y-Axis label: ${axesInfo.y.label}`);
            svg.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", margin.left / 3)
                .attr("x", 0 - (margin.top + plotHeight / 2))
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .style("font-size", "14px")
                .text(axesInfo.y.label);
        }

        // Draw Footer
        if (footer.length > 0) {
            const footerGroup = svg.append("g")
                .attr("transform", `translate(${margin.left}, ${height - margin.bottom + 75})`);
            footer.forEach((line, i) => {
                footerGroup.append("text")
                    .attr("y", i * 18)
                    .style("text-anchor", "start")
                    .style("font-size", "11px")
                    .style("fill", "#555")
                    .text(line.text);
            });
        }


        // Line generator with Smoothing (MonotoneX)
        const lineGenerator = d3.line()
            .x(d => xScale(d.x))
            .y(d => yScale(d.y))
            .curve(d3.curveMonotoneX); // Smooth curves to approximate potrace

        // Draw each series
        chartData.forEach((series, index) => {
            if (!series.traces || series.traces.length === 0) return;

            console.log(`[LOG] Drawing series "${series.name}": ${series.traces.length} traces`);

            series.traces.forEach(trace => {
                // Use the Series color (from Legend) as the primary source of truth
                // trace.style might be black/null if the path attribute was missing or complex
                const seriesColor = series.style.stroke !== 'none' ? series.style.stroke : '#000000';

                // If trace is a Marker OR has only 1 point: Draw Circle
                if (trace.isMarker || trace.points.length < 2) {
                    const pt = trace.centroid || trace.points[0];
                    if (pt) {
                        chartGroup.append("circle")
                            .attr("lc_legend_ref", series.id)
                            .attr("cx", xScale(pt.x))
                            .attr("cy", yScale(pt.y))
                            .attr("r", 4) // "Big Dot" size
                            .attr("fill", seriesColor)
                            .attr("stroke", "none");
                    }
                }
                // Otherwise: Draw Line Path
                else {
                    chartGroup.append("path")
                        .datum(trace.points)
                        .attr("lc_legend_ref", series.id)
                        .attr("d", lineGenerator)
                        .attr("fill", "none")
                        .attr("stroke", seriesColor)
                        .attr("stroke-width", series.style.strokeWidth || 2)
                        .attr("stroke-dasharray", series.style.strokeDasharray || 'none')
                        .attr("stroke-opacity", series.style.strokeOpacity || '1');

                    // OPTIONAL: Also draw markers embedded in the path that were identified by scanline
                    const embeddedMarkers = trace.points.filter(p => p.isMarker);
                    if (embeddedMarkers.length > 0) {
                        chartGroup.selectAll(null)
                            .data(embeddedMarkers)
                            .enter()
                            .append("circle")
                            .attr("lc_legend_ref", series.id)
                            .attr("cx", d => xScale(d.x))
                            .attr("cy", d => yScale(d.y))
                            .attr("r", 4)
                            .attr("fill", seriesColor)
                            .attr("stroke", "white")
                            .attr("stroke-width", 1);
                    }
                }
            });
        });

        // Draw Legend - Horizontal at bottom
        const legend = svg.append("g")
            .attr("transform", `translate(${margin.left}, ${height - margin.bottom + 50})`);

        let xOffset = 0;
        // Hardcoded Palette just for legend logic if color missing? No, use extracted style.

        chartData.forEach((series, index) => {
            const legendItem = legend.append("g")
                .attr("transform", `translate(${xOffset}, 0)`);

            const legendColor = series.style.stroke !== 'none' ? series.style.stroke : '#000';

            // Legend line or marker
            if (series.type === 'marker') {
                legendItem.append("circle")
                    .attr("cx", 10)
                    .attr("cy", 0)
                    .attr("r", 5)
                    .attr("fill", legendColor);
            } else {
                legendItem.append("line")
                    .attr("x1", 0)
                    .attr("y1", 0)
                    .attr("x2", 20)
                    .attr("y2", 0)
                    .attr("stroke", legendColor)
                    .attr("stroke-width", 2);
            }

            // Legend text
            const textElement = legendItem.append("text")
                .attr("x", 25)
                .attr("y", 4)
                .attr("lc_legend_id", series.id)
                .text(series.name)
                .style("font-size", "12px")
                .style("fill", "#333");

            // Calculate width for next item placement
            // Estimate width since we can't measure text easily in this context without rendering
            const textWidth = series.name.length * 7;
            xOffset += 35 + textWidth;
        });

        console.log(`[LOG] Chart redrawn with ${chartData.length} series.`);

        // Enable interactivity on the new chart
        this._enableNewChartInteractivity(containerSelector);
    }

    _enableNewChartInteractivity(containerSelector) {
        const container = document.querySelector(containerSelector);
        if (!container) return;
        const svg = container.querySelector('svg');
        if (!svg) return;

        console.log("[LOG] Enabling interactivity for the new chart...");

        const legendLabels = svg.querySelectorAll('text[lc_legend_id]');

        legendLabels.forEach(label => {
            label.style.cursor = 'pointer';
            label.addEventListener('click', () => {
                const legendId = label.getAttribute('lc_legend_id');
                if (!legendId) return;

                console.log(`[LOG] New chart legend clicked: ${legendId}`);

                const dataElements = svg.querySelectorAll(`[lc_legend_ref="${legendId}"]`);
                if (dataElements.length === 0) return;

                console.log(`[LOG] Flashing ${dataElements.length} elements for legend ${legendId}`);

                dataElements.forEach(el => {
                    const originalStroke = el.getAttribute('stroke') || 'none';
                    const originalStrokeWidth = el.getAttribute('stroke-width') || '1';
                    const originalFill = el.getAttribute('fill') || 'none';
                    const originalRadius = el.getAttribute('r') || '4';

                    const isPath = el.tagName.toLowerCase() === 'path';

                    // Bring to front
                    if (el.parentNode) {
                        el.parentNode.appendChild(el);
                    }

                    // Apply flash effect
                    el.style.transition = 'all 0.1s ease-in-out';

                    if (isPath) {
                        el.style.stroke = 'red';
                        el.style.strokeWidth = `${parseFloat(originalStrokeWidth) + 3}px`;
                    } else { // It's a circle
                        el.style.fill = 'red';
                        el.setAttribute('r', `${parseFloat(originalRadius) * 1.5}`);
                    }

                    // Revert after a delay
                    setTimeout(() => {
                        el.style.transition = 'all 0.4s ease-out';
                        if (isPath) {
                            el.style.stroke = originalStroke;
                            el.style.strokeWidth = originalStrokeWidth;
                        } else {
                            el.style.fill = originalFill;
                            el.setAttribute('r', originalRadius);
                        }
                    }, 150);

                    // Clean up transition property after it's done
                    setTimeout(() => {
                        el.style.transition = '';
                    }, 550);
                });
            });
        });
    }
}