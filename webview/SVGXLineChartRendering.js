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
    // Logical Mapping Implementation (Heuristic / Robust)
    // ==============================================================================

    /**
     * Main method to add logical mapping attributes (xlm, ylm) to the SVG.
     * @returns {void}
     */
    addLogicalMapping() {
        console.log("[LOG] SVGXLineChartRendering: Starting heuristic logical mapping extraction...");
        this._debugLogSVGStructure();

        try {
            // 1. Identify grid lines based on geometry
            const horizontalGridLines = this._findCandidateGridLines('horizontal');
            const verticalGridLines = this._findCandidateGridLines('vertical');

            if (horizontalGridLines.length < 2 && verticalGridLines.length < 2) {
                console.warn("[WARN] Could not find sufficient grid lines (H: " + horizontalGridLines.length + ", V: " + verticalGridLines.length + ")");
                // We don't return here immediately if one axis is found but not the other, 
                // we try to proceed for partial mapping if possible, though usually both are needed.
            }

            // 2. Find and parse all text labels
            const labels = this._findAllLabels();

            if (labels.length < 2) {
                console.warn("[WARN] Found insufficient text labels.");
                return;
            }

            // 3. Match labels to grid lines
            const yRefPoints = this._matchLabelsToGrid(horizontalGridLines, labels, 'y');
            const xRefPoints = this._matchLabelsToGrid(verticalGridLines, labels, 'x');

            if (!yRefPoints || yRefPoints.length < 2) {
                console.warn("[WARN] Could not extract robust Y-axis mapping.");
            }
            if (!xRefPoints || xRefPoints.length < 2) {
                console.warn("[WARN] Could not extract robust X-axis mapping.");
            }

            if ((!yRefPoints || yRefPoints.length < 2) && (!xRefPoints || xRefPoints.length < 2)) {
                return;
            }

            let xlm_string = "";
            let ylm_string = "";

            // 4. Calculate X mapping
            if (xRefPoints && xRefPoints.length >= 2) {
                // Sort by logical value
                const sortedX = [...xRefPoints].sort((a, b) => {
                    if (typeof a.logical === 'bigint' && typeof b.logical === 'bigint') {
                        return (a.logical < b.logical) ? -1 : ((a.logical > b.logical) ? 1 : 0);
                    }
                    return a.logical - b.logical;
                });

                const d_x_1 = sortedX[0].logical;
                const v_x_1 = sortedX[0].visual;
                const d_x_2 = sortedX[sortedX.length - 1].logical;
                const v_x_2 = sortedX[sortedX.length - 1].visual;

                // Format
                const x1Str = typeof d_x_1 === 'bigint' ? d_x_1.toString() : d_x_1.toFixed(2);
                const x2Str = typeof d_x_2 === 'bigint' ? d_x_2.toString() : d_x_2.toFixed(2);
                xlm_string = `[${x1Str}, ${x2Str}, ${v_x_1.toFixed(2)}, ${v_x_2.toFixed(2)}]`;
                this.svg.setAttribute("xlm", xlm_string);
            }

            // 5. Calculate Y mapping
            if (yRefPoints && yRefPoints.length >= 2) {
                const sortedY = [...yRefPoints].sort((a, b) => a.logical - b.logical);
                const d_y_1 = sortedY[0].logical;
                const v_y_1 = sortedY[0].visual;
                const d_y_2 = sortedY[sortedY.length - 1].logical;
                const v_y_2 = sortedY[sortedY.length - 1].visual;

                ylm_string = `[${d_y_1.toFixed(2)}, ${d_y_2.toFixed(2)}, ${v_y_1.toFixed(2)}, ${v_y_2.toFixed(2)}]`;
                this.svg.setAttribute("ylm", ylm_string);
            }

            console.log(`[LOG] Logical mapping added:`);
            console.log(`  xlm: ${xlm_string}`);
            console.log(`  ylm: ${ylm_string}`);

        } catch (error) {
            console.error("[ERROR] Failed to add logical mapping:", error);
        }
    }

    // -- Geometric Helpers --

    /**
     * Calculates the bounding box of an element in the root SVG user coordinate system.
     * Handles nested transforms.
     */
    _getBBoxInRoot(element) {
        try {
            const bbox = element.getBBox();
            const ctm = element.getCTM();
            if (!ctm || !bbox) return null;

            // Transform all 4 corners to handle rotation/skew correctly (though typically axis-aligned)
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

    // -- Extraction Logic --

    _findCandidateGridLines(orientation) {
        const paths = Array.from(this.svg.querySelectorAll('path'));
        const distinctLines = [];
        // Thresholds (in root user units)
        // We assume the SVG viewBox is reasonably standard or we use relative checks.
        // Let's get the SVG size.
        const svgRect = this.svg.viewBox.baseVal;
        const svgWidth = svgRect.width || this.svg.width.baseVal.value;
        const svgHeight = svgRect.height || this.svg.height.baseVal.value;

        // Heuristics regarding "long enough" to be a grid line
        // Lowered thresholds to capture axis tick marks and traced grid lines
        const MIN_LENGTH_RATIO = 0.04; // Must be at least 4% of the dimension
        const MAX_THICKNESS = 8; // Max visual thickness (traced lines can be thicker)

        let debugCount = { h: 0, v: 0 };
        paths.forEach(p => {
            // Skip if it is filling a tiny area or huge area (background)
            const box = this._getBBoxInRoot(p);
            if (!box) return;

            let isCandidate = false;
            let position = 0;

            if (orientation === 'horizontal') {
                // Wide and thin
                if (box.width > svgWidth * MIN_LENGTH_RATIO && box.height < MAX_THICKNESS) {
                    isCandidate = true;
                    position = box.cy; // Use center Y
                } else if (debugCount.h < 3 && box.width > svgWidth * 0.1 && box.height < 20) {
                    console.log(`[DEBUG] Near-miss H: W=${box.width.toFixed(1)}, H=${box.height.toFixed(1)} (need W>${(svgWidth * MIN_LENGTH_RATIO).toFixed(1)}, H<${MAX_THICKNESS})`);
                    debugCount.h++;
                }
            } else {
                // Tall and thin
                if (box.height > svgHeight * MIN_LENGTH_RATIO && box.width < MAX_THICKNESS) {
                    isCandidate = true;
                    position = box.cx; // Use center X
                } else if (debugCount.v < 3 && box.height > svgHeight * 0.1 && box.width < 20) {
                    console.log(`[DEBUG] Near-miss V: W=${box.width.toFixed(1)}, H=${box.height.toFixed(1)} (need H>${(svgHeight * MIN_LENGTH_RATIO).toFixed(1)}, W<${MAX_THICKNESS})`);
                    debugCount.v++;
                }
            }

            if (isCandidate) {
                distinctLines.push({ element: p, pos: position, bbox: box });
            }
        });

        // Dedup / Cluster lines that are very close (e.g. drawn with double stroke or dashed)
        const CLUSTER_TOLERANCE = 2.0;
        const sorted = distinctLines.sort((a, b) => a.pos - b.pos);
        const uniquePositions = [];

        if (sorted.length > 0) {
            let currentCluster = [sorted[0].pos];
            for (let i = 1; i < sorted.length; i++) {
                if (Math.abs(sorted[i].pos - currentCluster[0]) < CLUSTER_TOLERANCE) {
                    currentCluster.push(sorted[i].pos);
                } else {
                    // Average the cluster
                    const avg = currentCluster.reduce((a, b) => a + b, 0) / currentCluster.length;
                    uniquePositions.push(avg);
                    currentCluster = [sorted[i].pos];
                }
            }
            // Add last cluster
            const avg = currentCluster.reduce((a, b) => a + b, 0) / currentCluster.length;
            uniquePositions.push(avg);
        }

        console.log(`[LOG] Found ${uniquePositions.length} candidate ${orientation} grid lines.`);
        return uniquePositions;
    }

    _findAllLabels() {
        const texts = Array.from(this.svg.querySelectorAll('text'));
        const labels = texts.map(t => {
            const box = this._getBBoxInRoot(t);
            if (!box) return null;
            return {
                element: t,
                text: t.textContent.trim(),
                x: box.cx, // Use center for alignment logic
                y: box.cy,
                box: box
            };
        }).filter(item => item !== null && item.text.length > 0);

        // DEBUG: Log label positions
        console.log(`[DEBUG] Found ${labels.length} text labels:`);
        labels.slice(0, 10).forEach(l => {
            const val = this._parseLabelValue(l.text);
            console.log(`  "${l.text}" @ (${l.x.toFixed(1)}, ${l.y.toFixed(1)}) parsable=${!isNaN(val)}`);
        });

        return labels;
    }

    _parseLabelValue(text, isDate = false) {
        // Simple numeric parse first
        // Remove common currency symbols or grouping characters
        const numericClean = text.replace(/[$,]/g, '');
        if (!isNaN(parseFloat(numericClean))) {
            return parseFloat(numericClean);
        }
        // Date parsing could go here if needed...
        return NaN;
    }

    _matchLabelsToGrid(gridPositions, labels, axis) {
        if (gridPositions.length < 2) return null;

        // Increase tolerance since traced SVGs may have positioning variance
        const MATCH_TOLERANCE = 30; // User units tolerance for alignment

        // DEBUG: Log grid positions
        console.log(`[DEBUG] Matching ${axis}-axis. Grid positions (${gridPositions.length}):`);
        gridPositions.slice(0, 10).forEach(pos => console.log(`  ${axis}=${pos.toFixed(1)}`));

        const pairs = [];

        gridPositions.forEach(gridPos => {
            // Find labels that align with this grid line
            // If axis is 'y' (horizontal lines), we match Label Y to Grid Y
            // If axis is 'x' (vertical lines), we match Label X to Grid X

            let bestLabel = null;
            let minDist = Infinity;

            labels.forEach(lbl => {
                const dist = Math.abs((axis === 'y' ? lbl.y : lbl.x) - gridPos);
                if (dist < MATCH_TOLERANCE && dist < minDist) {
                    // Check if it's parsable
                    const val = this._parseLabelValue(lbl.text);
                    if (!isNaN(val)) {
                        minDist = dist;
                        bestLabel = { label: lbl, val: val };
                    }
                }
            });

            if (bestLabel) {
                pairs.push({ logical: bestLabel.val, visual: gridPos });
            }
        });

        // Filter duplicates by logical value
        const distinctMap = new Map();
        pairs.forEach(p => distinctMap.set(p.logical, p));
        return Array.from(distinctMap.values());
    }

    _debugLogSVGStructure() {
        console.log("--- SVG Structure Analysis ---");
        const allElements = this.svg.querySelectorAll('*');
        const classes = new Set();
        const tags = new Set();
        let pathCount = 0;
        let textCount = 0;
        let gCount = 0;

        allElements.forEach(el => {
            if (el.classList && el.classList.length > 0) {
                el.classList.forEach(c => classes.add(c));
            }
            tags.add(el.tagName);
            if (el.tagName === 'path') pathCount++;
            if (el.tagName === 'text') textCount++;
            if (el.tagName === 'g') gCount++;
        });

        console.log(`Unique Tags: ${Array.from(tags).join(', ')}`);
        console.log(`Unique Classes: ${Array.from(classes).join(', ')}`);
        console.log(`Counts: paths=${pathCount}, texts=${textCount}, groups=${gCount}`);

        // Debug SVG dimensions
        if (this.svg.viewBox && this.svg.viewBox.baseVal) {
            console.log(`ViewBox: ${this.svg.viewBox.baseVal.x}, ${this.svg.viewBox.baseVal.y}, ${this.svg.viewBox.baseVal.width}, ${this.svg.viewBox.baseVal.height}`);
        } else {
            console.log(`ViewBox: Not defined?`);
        }

        console.log("----------------------------");
    }
}
