/**
 * 2. SVGXLineChartDataExtractor
 * Responsible for transforming visual SVG data into logical data points.
 */
class SVGXLineChartDataExtractor {
    constructor(svgElement, legendItems, chartArea) {
        this.svg = svgElement;
        this.legendItems = legendItems;
        this.chartArea = chartArea; // Store the pre-calculated chart area
    }

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

                // Capture source color for re-association
                const pathColor = this._extractTraceStyle(path).stroke;

                // 1. Map points to Logical Coordinates
                // toLogicalX: dx_min + (vx - vx_min) * (dx_max - dx_min) / (vx_max - vx_min)
                // toLogicalY: dy_min + (vy - vy_min) * (dy_max - dy_min) / (vy_max - vy_min)
                const logicalPoints = points.map(pt => {
                    return {
                        x: this._toLogicalX(pt.x, xMapping[0], xMapping[1], xMapping[2], xMapping[3]),
                        y: this._toLogicalY(pt.y, yMapping[0], yMapping[1], yMapping[2], yMapping[3]),
                        isMarker: pt.isMarker,
                        // Pass markerType ('general') and srcColor to allow re-association in Rendering
                        markerType: pt.markerType,
                        srcColor: pathColor
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
        
        // --- CRITICAL FIX ---
        // Use the accurate chartArea passed from the Analyzer, not the one calculated internally by findAxisDomains().
        // The internal one (domains.visualX/Y) can be slightly off, causing text to be missed.
        const visualXForText = this.chartArea && this.chartArea.visualX ? this.chartArea.visualX : domains.visualX;
        const visualYForText = this.chartArea && this.chartArea.visualY ? this.chartArea.visualY : domains.visualY;
        const chartTexts = this._findChartAreaTexts(visualXForText, visualYForText, xMapping, yMapping);
        // --- END FIX ---


        // Requested Console Logs
        console.log("X info:", JSON.stringify(axes.x));
        console.log("Y info:", JSON.stringify(axes.y));

        return { series: extractedSeries, axes: axes, title: titleAndFooter.title, footer: titleAndFooter.footer, chartTexts: chartTexts };
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
     * Finds text elements located within the main chart plot area.
     * @param {Array} xRange - [min, max] visual X of chart plot area.
     * @param {Array} yRange - [min, max] visual Y of chart plot area.
     * @returns {Array<{text: string, x: number, y: number, style: object}>}
     * @private
     */
    _findChartAreaTexts(xRange, yRange, xMapping, yMapping) {
        if (!xRange || !yRange) {
            console.warn("[WARN] Cannot find chart area texts without a valid visual range.");
            return [];
        }
    
        const allTexts = Array.from(this.svg.querySelectorAll('text'));
        const chartTexts = [];
    
        // --- Build an exclusion list of all known non-chart text elements ---
        const exclusionSet = new Set();
        // 1. Exclude legend text elements
        const legendItemTexts = this.legendItems ? this.legendItems.map(item => item.text) : [];
        legendItemTexts.forEach(text => exclusionSet.add(text));
        
        const [minX, maxX] = xRange;
        const [minY, maxY] = yRange;
    
        allTexts.forEach(t => {
            const text = t.textContent.trim();
            if (!text || exclusionSet.has(text)) {
                return; // Skip empty or explicitly excluded text
            }
    
            const bbox = this._getFullBBox(t);
            if (!bbox) return;
    
            // Check if the center of the text is inside the plot area
            if (bbox.cx >= minX && bbox.cx <= maxX && bbox.cy >= minY && bbox.cy <= maxY) {
                const lx = this._toLogicalX(bbox.x, ...xMapping);
                const ly = this._toLogicalY(bbox.y, ...yMapping);
                const style = window.getComputedStyle(t);
                chartTexts.push({
                    text: text,
                    x: bbox.x,
                    y: bbox.y,
                    lx: lx,
                    ly: ly,
                    style: {
                        fill: style.fill,
                        fontSize: style.fontSize,
                        fontFamily: style.fontFamily,
                        fontWeight: style.fontWeight,
                    }
                });
            }
        });
        console.log(`[LOG] Found ${chartTexts.length} text elements inside the chart area.`);
        return chartTexts;
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

        // Keep Step = 4.0 as requested, but ensure minimal density for short paths
        const step = Math.max(0.5, Math.min(4.0, len / 100));

        let debugCount = 0; // Debug limit

        for (let i = 0; i < len; i += step) {
            const rawPt = pathElement.getPointAtLength(i);

            pt.x = rawPt.x;
            pt.y = rawPt.y;

            let transformedPt = pt;
            if (ctm && inverseRootCTM) {
                transformedPt = pt.matrixTransform(ctm).matrixTransform(inverseRootCTM);

                // DATA DROP DEBUGGING
                if (debugCount < 3) {
                    console.log(`[DEBUG] Point ${debugCount}: Raw(${rawPt.x.toFixed(1)}, ${rawPt.y.toFixed(1)}) -> Transformed(${transformedPt.x.toFixed(1)}, ${transformedPt.y.toFixed(1)})`);
                    debugCount++;
                }
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
                        finalPoints.push({ x: avgX, y: avgY, isMarker: true, markerType: 'general' });

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
                        finalPoints.push({ x: avgX, y: avgY, isMarker: true, markerType: 'general' });
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
            finalPoints.push({ x: avgX, y: avgY, isMarker: true, markerType: 'general' });
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
}