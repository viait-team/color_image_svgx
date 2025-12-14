/**
 * 3. SVGXLineChartNewRendering
 * Responsible for using D3.js to render axes, lines, markers, and visual elements.
 */
class SVGXLineChartNewRendering {
    constructor(svgElement) {
        this.svg = svgElement;
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
                            .attr("stroke-width", 1)
                            .attr("lc_marker_type", "general"); // Tag for re-association
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

        // --- NEW: Perform Re-association of embedded markers ---
        this._associationCorrection(svg, chartData);

        // Enable interactivity on the new chart
        this._enableNewChartInteractivity(containerSelector);
    }

    _associationCorrection(svg, chartData) {
        console.log("[LOG] Performing association correction for embedded markers...");
        
        // 1. Filter chartData to find legend definitions that are Markers
        const markerLegends = chartData.filter(s => s.type === 'marker');
        if (markerLegends.length === 0) return;

        // 2. Select all embedded markers tagged as 'general'
        const embeddedMarkers = svg.selectAll('circle[lc_marker_type="general"]');

        if (embeddedMarkers.empty()) return;

        const self = this;

        embeddedMarkers.each(function() {
            const circle = d3.select(this);
            const circleColor = circle.attr("fill");

            let bestMatch = null;
            let minDist = Infinity;

            // 3. Compare color with Marker Legends
            markerLegends.forEach(leg => {
                // leg.style.stroke usually holds the legend color in the extracted data
                const legColor = leg.style.stroke;
                const dist = self._getColorDistance(
                    self._normalizeColor(circleColor), 
                    self._normalizeColor(legColor)
                );

                if (dist < 60 && dist < minDist) {
                    minDist = dist;
                    bestMatch = leg;
                }
            });

            // 4. Update Reference if match found
            if (bestMatch) {
                circle.attr("lc_legend_ref", bestMatch.id);
                // Optional: Update visual style to match legend if needed (e.g. if legend had different fill)
            }
        });
    }

    _getColorDistance(c1, c2) {
        if (!c1 || !c2 || !c1.startsWith('#') || !c2.startsWith('#')) return Infinity;
        const r1 = parseInt(c1.substr(1, 2), 16), g1 = parseInt(c1.substr(3, 2), 16), b1 = parseInt(c1.substr(5, 2), 16);
        const r2 = parseInt(c2.substr(1, 2), 16), g2 = parseInt(c2.substr(3, 2), 16), b2 = parseInt(c2.substr(5, 2), 16);
        return Math.sqrt(Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(b1 - b2, 2));
    }

    _normalizeColor(color) {
        if (!color) return '';
        color = color.trim().toLowerCase();
        if (color.startsWith('#')) return color.length === 4 ? `#${color[1]}${color[1]}${color[2]}${color[2]}${color[3]}${color[3]}` : color;
        const rgb = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (rgb) return `#${(+rgb[1]).toString(16).padStart(2,'0')}${(+rgb[2]).toString(16).padStart(2,'0')}${(+rgb[3]).toString(16).padStart(2,'0')}`;
        return color;
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