///
/// Line Chart Rendering using D3.js
///

class SVGXLineChartRendering {
    /**
     * @param {SVGElement} svgElement The SVG element containing the line chart
     */
    constructor(svgElement) {
        this.svg = svgElement;
        if (!this.svg) {
            throw new Error('SVG element is not defined.');
        }
        console.log('[LOG] Line chart renderer initialized');
    }

    // ==============================================================================
    // Logical Mapping Implementation
    // Grid lines are ground truth; match labels to closest grid lines
    // ==============================================================================

    addLogicalMapping() {
        console.log("[LOG] SVGXLineChartRendering: Starting logical mapping extraction...");
        this._debugLogSVGStructure();

        try {
            // 1. Identify grid lines based on geometry
            const horizontalGridLines = this._findCandidateGridLines('horizontal');
            const verticalGridLines = this._findCandidateGridLines('vertical');

            if (horizontalGridLines.length < 2 && verticalGridLines.length < 2) {
                console.warn("[WARN] Could not find sufficient grid lines (H: " + horizontalGridLines.length + ", V: " + verticalGridLines.length + ")");
                return;
            }

            // 2. Find all text labels with their positions
            const allLabels = this._findAllLabels();

            if (allLabels.length < 2) {
                console.warn("[WARN] Found insufficient text labels.");
                return;
            }

            // 3. Match labels to grid lines
            // Y-axis: match labels to horizontal grid lines (by Y proximity)
            // X-axis: match labels to vertical grid lines (by X proximity)
            const yRefPoints = this._matchLabelsToGridByProximity(horizontalGridLines, allLabels, 'y');
            const xRefPoints = this._matchLabelsToGridByProximity(verticalGridLines, allLabels, 'x');

            let xlm_string = "";
            let ylm_string = "";

            // 4. Calculate Y mapping
            if (yRefPoints && yRefPoints.length >= 2) {
                // Sort by visual position
                const sortedY = [...yRefPoints].sort((a, b) => a.visual - b.visual);
                const d_y_1 = sortedY[0].logical;
                const v_y_1 = sortedY[0].visual;
                const d_y_2 = sortedY[sortedY.length - 1].logical;
                const v_y_2 = sortedY[sortedY.length - 1].visual;

                ylm_string = `[${d_y_1.toFixed(2)}, ${d_y_2.toFixed(2)}, ${v_y_1.toFixed(2)}, ${v_y_2.toFixed(2)}]`;
                this.svg.setAttribute("ylm", ylm_string);
            } else {
                console.warn("[WARN] Could not extract robust Y-axis mapping.");
            }

            // 5. Calculate X mapping
            if (xRefPoints && xRefPoints.length >= 2) {
                const sortedX = [...xRefPoints].sort((a, b) => a.visual - b.visual);
                const d_x_1 = sortedX[0].logical;
                const v_x_1 = sortedX[0].visual;
                const d_x_2 = sortedX[sortedX.length - 1].logical;
                const v_x_2 = sortedX[sortedX.length - 1].visual;

                const x1Str = typeof d_x_1 === 'bigint' ? d_x_1.toString() : d_x_1.toFixed(2);
                const x2Str = typeof d_x_2 === 'bigint' ? d_x_2.toString() : d_x_2.toFixed(2);
                xlm_string = `[${x1Str}, ${x2Str}, ${v_x_1.toFixed(2)}, ${v_x_2.toFixed(2)}]`;
                this.svg.setAttribute("xlm", xlm_string);
            } else {
                console.warn("[WARN] Could not extract robust X-axis mapping.");
            }

            if (!xlm_string && !ylm_string) {
                console.warn("[WARN] Failed to extract any logical mapping.");
                return;
            }

            console.log(`[LOG] Logical mapping added:`);
            console.log(`  xlm: ${xlm_string}`);
            console.log(`  ylm: ${ylm_string}`);

        } catch (error) {
            console.error("[ERROR] Failed to add logical mapping:", error);
        }
    }

    // -- Grid Line Detection --

    _findCandidateGridLines(orientation) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const distinctLines = [];

        const svgRect = this.svg.viewBox.baseVal;
        const svgWidth = svgRect.width || this.svg.clientWidth;
        const svgHeight = svgRect.height || this.svg.clientHeight;

        const MIN_LENGTH_RATIO = 0.04;
        const MAX_THICKNESS = 8;

        paths.forEach(p => {
            const box = this._getBBoxInRoot(p);
            if (!box) return;

            let isCandidate = false;
            let position = 0;

            if (orientation === 'horizontal') {
                if (box.width > svgWidth * MIN_LENGTH_RATIO && box.height < MAX_THICKNESS) {
                    isCandidate = true;
                    position = box.cy;
                }
            } else {
                if (box.height > svgHeight * MIN_LENGTH_RATIO && box.width < MAX_THICKNESS) {
                    isCandidate = true;
                    position = box.cx;
                }
            }

            if (isCandidate) {
                distinctLines.push({ pos: position, bbox: box });
            }
        });

        // Cluster nearby lines
        const CLUSTER_TOLERANCE = 3.0;
        const sorted = distinctLines.sort((a, b) => a.pos - b.pos);
        const uniquePositions = [];

        if (sorted.length > 0) {
            let currentCluster = [sorted[0].pos];
            for (let i = 1; i < sorted.length; i++) {
                if (sorted[i].pos - currentCluster[currentCluster.length - 1] < CLUSTER_TOLERANCE) {
                    currentCluster.push(sorted[i].pos);
                } else {
                    const avg = currentCluster.reduce((a, b) => a + b, 0) / currentCluster.length;
                    uniquePositions.push(avg);
                    currentCluster = [sorted[i].pos];
                }
            }
            const avg = currentCluster.reduce((a, b) => a + b, 0) / currentCluster.length;
            uniquePositions.push(avg);
        }

        console.log(`[LOG] Found ${uniquePositions.length} candidate ${orientation} grid lines.`);
        return uniquePositions;
    }

    // -- Label Extraction --

    _findAllLabels() {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const labels = texts.map(t => {
            const box = this._getBBoxInRoot(t);
            if (!box) return null;
            const val = this._parseLabelValue(t.textContent.trim());
            return {
                text: t.textContent.trim(),
                x: box.cx,
                y: box.cy,
                value: val,
                isParsable: !isNaN(val)
            };
        }).filter(item => item !== null && item.text.length > 0);

        console.log(`[DEBUG] Found ${labels.length} labels, ${labels.filter(l => l.isParsable).length} parsable as numbers`);
        return labels;
    }

    _parseLabelValue(text) {
        const clean = text.replace(/[$,%]/g, '').trim();
        return parseFloat(clean);
    }

    // -- Label to Grid Line Matching --

    /**
     * Match labels to grid lines using proximity.
     * For each parsable label, find the closest grid line.
     * Return pairs where the match distance is reasonable.
     */
    _matchLabelsToGridByProximity(gridPositions, allLabels, axis) {
        if (gridPositions.length < 2) return null;

        console.log(`[DEBUG] Matching ${axis}-axis. Grid positions: ${gridPositions.map(p => p.toFixed(1)).join(', ')}`);

        // Filter to parsable numeric labels
        const numericLabels = allLabels.filter(l => l.isParsable);

        const pairs = [];

        numericLabels.forEach(label => {
            const labelPos = (axis === 'y') ? label.y : label.x;

            // Find closest grid line
            let closestGrid = null;
            let closestDist = Infinity;

            gridPositions.forEach(gridPos => {
                const dist = Math.abs(labelPos - gridPos);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestGrid = gridPos;
                }
            });

            if (closestGrid !== null) {
                pairs.push({
                    logical: label.value,
                    visual: closestGrid,  // Grid line position is the ground truth
                    labelPos: labelPos,
                    distance: closestDist,
                    text: label.text
                });
            }
        });

        // Sort by distance (best matches first)
        pairs.sort((a, b) => a.distance - b.distance);

        console.log(`[DEBUG] ${axis}-axis label-to-grid matches (sorted by distance):`);
        pairs.slice(0, 10).forEach(p => {
            console.log(`  "${p.text}" (${p.logical}) @ label=${p.labelPos.toFixed(1)} -> grid=${p.visual.toFixed(1)}, dist=${p.distance.toFixed(1)}`);
        });

        // Keep only pairs with reasonable distance (adaptive threshold)
        // Use median distance as a guide if we have enough pairs
        let threshold = 100; // Default max threshold
        if (pairs.length >= 3) {
            const sortedDists = pairs.map(p => p.distance).sort((a, b) => a - b);
            const medianDist = sortedDists[Math.floor(sortedDists.length / 2)];
            threshold = Math.max(medianDist * 2, 50); // At least 50, or 2x median
        }

        const goodPairs = pairs.filter(p => p.distance < threshold);
        console.log(`[DEBUG] Keeping ${goodPairs.length} pairs with distance < ${threshold.toFixed(1)}`);

        // Remove duplicates by logical value (keep best match)
        const uniqueMap = new Map();
        goodPairs.forEach(p => {
            if (!uniqueMap.has(p.logical)) {
                uniqueMap.set(p.logical, p);
            }
        });

        // Also ensure we use different grid lines for different logical values
        const usedGrids = new Set();
        const finalPairs = [];
        Array.from(uniqueMap.values()).forEach(p => {
            const gridKey = p.visual.toFixed(1);
            if (!usedGrids.has(gridKey)) {
                usedGrids.add(gridKey);
                finalPairs.push(p);
            }
        });

        console.log(`[DEBUG] Final ${axis}-axis pairs: ${finalPairs.length}`);
        finalPairs.forEach(p => console.log(`  logical=${p.logical}, visual=${p.visual.toFixed(1)}`));

        return finalPairs.length >= 2 ? finalPairs : null;
    }

    // -- Geometric Helpers --

    _getBBoxInRoot(element) {
        try {
            const bbox = element.getBBox();
            const ctm = element.getCTM();
            if (!ctm || !bbox) return null;

            const pts = [
                this._transformPoint(bbox.x, bbox.y, ctm),
                this._transformPoint(bbox.x + bbox.width, bbox.y, ctm),
                this._transformPoint(bbox.x + bbox.width, bbox.y + bbox.height, ctm),
                this._transformPoint(bbox.x, bbox.y + bbox.height, ctm)
            ];

            const xs = pts.map(p => p.x);
            const ys = pts.map(p => p.y);

            return {
                x: Math.min(...xs),
                y: Math.min(...ys),
                width: Math.max(...xs) - Math.min(...xs),
                height: Math.max(...ys) - Math.min(...ys),
                cx: (Math.min(...xs) + Math.max(...xs)) / 2,
                cy: (Math.min(...ys) + Math.max(...ys)) / 2
            };
        } catch (e) {
            return null;
        }
    }

    _transformPoint(x, y, matrix) {
        return {
            x: x * matrix.a + y * matrix.c + matrix.e,
            y: x * matrix.b + y * matrix.d + matrix.f
        };
    }

    _debugLogSVGStructure() {
        console.log("--- SVG Structure Analysis ---");
        const allElements = this.svg.querySelectorAll('*');
        let pathCount = 0, textCount = 0, gCount = 0;
        allElements.forEach(el => {
            if (el.tagName === 'path') pathCount++;
            if (el.tagName === 'text') textCount++;
            if (el.tagName === 'g') gCount++;
        });
        console.log(`Counts: paths=${pathCount}, texts=${textCount}, groups=${gCount}`);
        if (this.svg.viewBox && this.svg.viewBox.baseVal) {
            console.log(`ViewBox: ${this.svg.viewBox.baseVal.x}, ${this.svg.viewBox.baseVal.y}, ${this.svg.viewBox.baseVal.width}, ${this.svg.viewBox.baseVal.height}`);
        }
        console.log("----------------------------");
    }
}
