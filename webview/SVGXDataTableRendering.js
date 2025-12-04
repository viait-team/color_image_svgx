///
/// OCR TSV to structured data and SVG chart rendering
///
class SVGXDataTableRendering {
    /**
     * @param {string} svgElementId The ID of the SVG element containing original svg chart .
     */
    constructor(svgElement) {
        this.svg = svgElement
        if (!this.svg) {
            throw new Error(`SVG element is not defined.`);
        }
    }

    // New helper method to generate a machine-friendly ID from a text label.
    /**
     * Generates a machine-friendly ID from a human-readable text label.
     * @param {string} text - The input text (e.g., "80% CI").
     * @returns {string} A formatted ID (e.g., "80_ci").
     * @private
     */
    _generateLcId(text) {
        if (!text) return '';
        return text.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
    }

    groupWordsByTop(words, rowTolerance = 10) {
        const rowDict = {};

        for (let i = 0; i < words.length; i++) {
            const word = words[i];

            if (
                !word ||
                !word.text ||
                word.text.length === 0
            ) {
                continue;
            }

            const key = Math.round(word.top / rowTolerance) * rowTolerance;

            if (!rowDict[key]) {
                rowDict[key] = [];
            }
            rowDict[key].push(word);
        }

        const rows = Object.entries(rowDict).map(([top, words]) => ({
            top: parseInt(top),
            words
        }));

        return rows;
    }

    hasTwoMoreZeros(arr) {
        let count = 0;
        let found = false;
        for (const val of arr) {
            if (val === undefined || val === "") continue;

            if (parseFloat(val) === 0) {
                count++;
                if (count >= 2) {
                    found = true;
                    break;
                }
            }
        }

        return found;
    }

    isNumberRow(arr) {
        if (!Array.isArray(arr) || arr.length === 0) return false;

        const count = arr.filter(val => {
            const num = typeof val === 'number' ? val : parseFloat(val);
            return !isNaN(num);
        }).length;

        return count / arr.length >= 0.9;
    }

    isTickRow(arr) {
        let isTickRow = this.isNumberRow(arr);
        if (isTickRow) {
            isTickRow = !this.hasTwoMoreZeros(arr);
        }
        return isTickRow
    }

    getDominantDiff(arr) {
        const nums = arr.map(parseFloat);
        const freq = {};

        for (let i = 1; i < nums.length; i++) {
            const diff = nums[i] - nums[i - 1];
            if (diff === 0) continue;

            const key = diff.toFixed(10);
            freq[key] = (freq[key] || 0) + 1;
        }

        let dominant = null;
        let maxCount = 0;

        for (const [key, count] of Object.entries(freq)) {
            const value = parseFloat(key);
            if (
                count > maxCount ||
                (count === maxCount && value < dominant)
            ) {
                maxCount = count;
                dominant = value;
            }
        }

        return dominant;
    }

    validateTicks(ticks, colCount) {
        if (!Array.isArray(ticks) || ticks.length < 2) return ticks;

        const start_val = parseFloat(ticks[0]);
        const end_val = parseFloat(ticks[ticks.length - 1]);
        if ((start_val < end_val) && (ticks.length === colCount)) {
            const count = ticks.length;
            const interval = (end_val - start_val) / (count - 1);

            const corrected = [...ticks];

            for (let i = 1; i < count; i++) {
                const expected = start_val + i * interval;
                const expectedStr = expected.toFixed(1);
                corrected[i] = expectedStr;
            }

            return corrected;
        } else {
            const diff = this.getDominantDiff(ticks);
            const result = [];
            for (let i = 0; i < colCount; i++) {
                const value = diff + i * diff;
                result.push(value.toString());
            }
            return result;
        }

    }

    identifyAndFixCsvTableData(finalRows) {
        if (!Array.isArray(finalRows)) {
            console.error("Error: Input data is not an array.");
            return finalRows;
        }

        let tempRows1 = this.identifyOutliersAndFixCsvTableData(finalRows);
        let tempRows2 = this.fixCsvTableDataByColumnSummation(tempRows1);

        finalRows = tempRows2;
        return finalRows;
    }

    identifyOutliersAndFixCsvTableData(finalRows) {
        for (let i = 0; i < finalRows.length; i++) {
            if (!Array.isArray(finalRows[i])) continue;

            for (let j = 0; j < finalRows[i].length; j++) {
                const originalString = finalRows[i][j];
                if (typeof originalString !== 'string') continue;

                const currentValue = Number(originalString);
                const leftStr = j > 0 ? finalRows[i][j - 1] : null;
                const rightStr = j < finalRows[i].length - 1 ? finalRows[i][j + 1] : null;
                const left = typeof leftStr === 'string' ? Number(leftStr) : null;
                const right = typeof rightStr === 'string' ? Number(rightStr) : null;
                if (left === null && right === null) continue;

                const neighborMax = Math.max(left, right);
                const outlierLimit = neighborMax * 3;
                if (currentValue > outlierLimit) {
                    const noDecimalString = originalString.replace('.', '');
                    if (noDecimalString.includes('41')) {
                        let decimalPlaces = 0;
                        if (originalString.includes('.')) {
                            decimalPlaces = originalString.split('.')[1].length;
                        }
                        const correctedNoDecimalString = noDecimalString.replace('41', '1');
                        let finalString = correctedNoDecimalString;
                        if (decimalPlaces > 0) {
                            const insertPosition = correctedNoDecimalString.length - decimalPlaces;
                            finalString = correctedNoDecimalString.slice(0, insertPosition) + '.' + correctedNoDecimalString.slice(insertPosition);
                        }
                        console.log(`Fix at [${i},${j}]: "${originalString}" -> "${finalString}"`);
                        finalRows[i][j] = finalString;
                    }
                }
            }
        }
        return finalRows;
    }

    fixCsvTableDataByColumnSummation(finalRows) {
        const numRows = finalRows.length;
        if (numRows === 0) return finalRows;

        const numCols = finalRows[0].length;
        const columnSums = Array(numCols).fill(0);

        for (let col = 0; col < numCols; col++) {
            for (let row = 0; row < numRows; row++) {
                const val = parseFloat(finalRows[row][col]);
                if (!isNaN(val)) columnSums[col] += val;
            }
        }

        for (let col = 0; col < numCols; col++) {
            const sum = columnSums[col];
            const diff = +(Math.abs(sum - 100)).toFixed(10);

            if (diff != 0) {
                if (diff === 3) {
                    for (let row = 0; row < numRows; row++) {
                        let cell = finalRows[row][col];
                        const dotIndex = cell.indexOf('.');
                        if (dotIndex > 0) {
                            const digitBeforeDot = cell[dotIndex - 1];
                            if (digitBeforeDot === '8') {
                                finalRows[row][col] = cell.slice(0, dotIndex - 1) + '5' + cell.slice(dotIndex);
                                break;
                            }
                        } else if (cell === '8') {
                            finalRows[row][col] = '5.0';
                            break;
                        }
                    }
                } else if (diff === 0.3) {
                    for (let row = 0; row < numRows; row++) {
                        let cell = finalRows[row][col];
                        const dotIndex = cell.indexOf('.');
                        if (dotIndex !== -1 && cell[dotIndex + 1] === '8') {
                            finalRows[row][col] = cell.slice(0, dotIndex + 1) + '5' + cell.slice(dotIndex + 2);
                            break;
                        }
                    }
                } else if (diff === 0.03) {
                    for (let row = 0; row < numRows; row++) {
                        if (finalRows[row][col] === '0.08') {
                            finalRows[row][col] = '0.05';
                            break;
                        }
                    }
                }
            }
        }
        return finalRows;
    }

    getStructuredOcrData(
        tsvContent,
        rowTolerance = 10,
        exportComments = true,
        exportTitle = true,
        exportXAxisLabel = true,
        exportXTicks = true,
        exportYAxisLabel = true,
        exportYTicks = true,
        exportTableCsv = true
    ) {
        let ycount = exportYTicks ? 1 : 0;
        let xcount = exportXTicks ? 1 : 0;

        const lines = tsvContent.trim().split('\n');
        const headers = lines[0].split('\t');
        const words = [];

        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split('\t');
            const obj = {};
            for (let j = 0; j < headers.length; j++) {
                obj[headers[j]] = parts[j];
            }
            if (obj.text && obj.text.trim() !== '') {
                words.push({
                    text: obj.text.trim(),
                    left: parseInt(obj.left),
                    top: parseInt(obj.top),
                    width: parseInt(obj.width),
                    height: parseInt(obj.height),
                    right: parseInt(obj.left) + parseInt(obj.width),
                    bottom: parseInt(obj.top) + parseInt(obj.height)
                });
            }
        }

        const rows = this.groupWordsByTop(words, rowTolerance);

        const countMap = new Map();
        for (let i = 0; i < rows.length; i++) {
            const count = rows[i].words.length;
            countMap.set(count, (countMap.get(count) || 0) + 1);
        }

        let mostCount = 0;
        let maxFreq = 0;
        for (let [count, freq] of countMap.entries()) {
            if (freq > maxFreq) {
                mostCount = count;
                maxFreq = freq;
            }
        }

        const tableCount = mostCount - ycount;

        const candidateRows = rows.filter(row => row.words.length >= tableCount);
        const sortedRows = candidateRows.sort((a, b) => a.top - b.top);
        const tableRows = [];

        for (let i = 0; i < sortedRows.length; i++) {
            const row = sortedRows[i];
            const sortedWords = [...row.words].sort((a, b) => b.left - a.left);
            const sliced = sortedWords.slice(0, tableCount);
            if (sliced.length !== tableCount) continue;

            const ordered = sliced.sort((a, b) => a.left - b.left);
            tableRows.push(ordered.map(w => w.text));
        }

        let tableStartIndex = -1;
        for (let i = 0; i < rows.length; i++) {
            if (rows[i].words.length >= tableCount) {
                tableStartIndex = i;
                break;
            }
        }

        let x_axis_label_shift = 1;
        let title_shift = 2;

        let x_ticks = [];
        if (exportXTicks) {
            let isTickRows = this.isTickRow(tableRows[0]);
            if (isTickRows) {
                x_ticks = tableRows[0];
            } else {
                let tempRow = rows[tableStartIndex - 1];
                x_ticks = tempRow.words.map(word => word.text);
                xcount = 0;
                x_axis_label_shift = 2;
                title_shift = 3;
                // title OCR create a row with less than 3 characters, then we need to shift up
                const titleTemp = exportTitle && tableStartIndex >= title_shift ?
                        rows[tableStartIndex - title_shift].words.map(w => w.text).join(' ') :
                        "";
                if (titleTemp.length <3 ){
                    x_axis_label_shift += 1;
                    title_shift += 1;
                }               
            }
        }
        x_ticks = this.validateTicks(x_ticks, tableCount);

        let finalRows = tableRows;
        if (xcount > 0) {
            finalRows = tableRows.slice(xcount);
        }

        const x_axis_label = exportXAxisLabel && tableStartIndex >= x_axis_label_shift ?
            rows[tableStartIndex - x_axis_label_shift].words.map(w => w.text).join(' ') :
            "";

        const title = exportTitle && tableStartIndex >= title_shift ?
            rows[tableStartIndex - title_shift].words.map(w => w.text).join(' ') :
            "";

        const comments = exportComments && tableStartIndex >= (title_shift + 1) ?
            rows.slice(0, tableStartIndex - title_shift).map(row =>
                row.words.map(w => w.text).join(' ')
            ) :
            [];

        const y_ticks = exportYTicks ?
            sortedRows.map(row => {
                const index = row.words.length - (tableCount + ycount);
                const word = row.words[index];
                return word && word.text && word.text.trim() !== "" ? word.text.trim() : null;
            }).filter(text => text !== null) :
            [];

        let y_axis_label = "";
        if (exportYAxisLabel) {
            const parts = [];
            for (let i = tableStartIndex; i < rows.length; i++) {
                const row = rows[i];
                let overflow = row.words.length - (tableCount + ycount);
                if (overflow > 0) {
                    for (let j = 0; j < overflow; j++) {
                        const word = row.words[j];
                        if (word && word.text && word.text.trim() !== "") {
                            parts.push(word.text.trim());
                        }
                    }
                } else if (overflow < -ycount) {
                    for (let j = 0; j < row.words.length; j++) {
                        const word = row.words[j];
                        if (word && word.text && word.text.trim() !== "") {
                            parts.push(word.text.trim());
                        }
                    }
                }
            }
            y_axis_label = parts.join(' ');
        }

        const finalRowsFixed = this.identifyAndFixCsvTableData(finalRows);

        const tableCsv = exportTableCsv ?
            finalRowsFixed.map(row =>
                row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')
            ).join('\n') :
            "";

        return {
            comments,
            title,
            x_axis_label,
            x_ticks,
            y_axis_label,
            y_ticks,
            tableCsv
        };
    }

    getCsvTableData(tsvContent, rowTolerance = 10) {
        let ycount = 1;
        let xcount = 1;

        const lines = tsvContent.trim().split('\n');
        const headers = lines[0].split('\t');
        const words = [];

        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split('\t');
            const obj = {};
            for (let j = 0; j < headers.length; j++) {
                obj[headers[j]] = parts[j];
            }
            if (obj.text && obj.text.trim() !== '') {
                words.push({
                    text: obj.text.trim(),
                    left: parseInt(obj.left),
                    top: parseInt(obj.top),
                    width: parseInt(obj.width),
                    height: parseInt(obj.height),
                    right: parseInt(obj.left) + parseInt(obj.width),
                    bottom: parseInt(obj.top) + parseInt(obj.height)
                });
            }
        }

        const rows = this.groupWordsByTop(words, rowTolerance);

        const countMap = new Map();
        for (let i = 0; i < rows.length; i++) {
            const count = rows[i].words.length;
            countMap.set(count, (countMap.get(count) || 0) + 1);
        }

        let mostCount = 0;
        let maxFreq = 0;
        for (let [count, freq] of countMap.entries()) {
            if (freq > maxFreq) {
                mostCount = count;
                maxFreq = freq;
            }
        }

        const tableCount = mostCount - ycount;

        const candidateRows = rows.filter(row => row.words.length >= tableCount);

        const sortedRows = candidateRows.sort((a, b) => a.top - b.top);
        const tableRows = [];

        for (let i = 0; i < sortedRows.length; i++) {
            const row = sortedRows[i];
            const sortedWords = [...row.words].sort((a, b) => b.left - a.left);
            const sliced = sortedWords.slice(0, tableCount);
            if (sliced.length !== tableCount) continue;

            const ordered = sliced.sort((a, b) => a.left - b.left);
            tableRows.push(ordered.map(w => w.text));
        }

        let finalRows = tableRows;
        if (xcount > 0) {
            finalRows = tableRows.slice(xcount);
        }

        const csvString = finalRows.map(row =>
            row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(',')
        ).join('\n');

        return csvString;
    }

    getStyleFromTableSvg() {
        const svgNode = this.svg;

        const width = svgNode.getAttribute('width');
        const height = svgNode.getAttribute('height');

        const firstG = svgNode.querySelector('g[fill]');
        const bgColor = firstG ? firstG.getAttribute('fill') : '#ffffff';

        const fills = new Set();
        const gElements = svgNode.querySelectorAll('g[fill]');
        gElements.forEach(g => {
            const fill = g.getAttribute('fill').toLowerCase();
            if (fill !== bgColor) {
                fills.add(fill);
            }
        });

        const hexToHsl = (hex) => {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            let r = parseInt(result[1], 16) / 255;
            let g = parseInt(result[2], 16) / 255;
            let b = parseInt(result[3], 16) / 255;
            const max = Math.max(r, g, b),
                min = Math.min(r, g, b);
            let h, s, l = (max + min) / 2;
            if (max === min) {
                h = s = 0;
            } else {
                const d = max - min;
                s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                switch (max) {
                    case r:
                        h = (g - b) / d + (g < b ? 6 : 0);
                        break;
                    case g:
                        h = (b - r) / d + 2;
                        break;
                    case b:
                        h = (r - g) / d + 4;
                        break;
                }
                h /= 6;
            }
            return { h, s, l };
        };

        const sortedColors = [...fills].sort((a, b) => {
            const hslA = hexToHsl(a);
            const hslB = hexToHsl(b);
            if (hslA.h < hslB.h) return 1;
            if (hslA.h > hslB.h) return -1;
            return 0;
        });

        const colorRange = [bgColor, ...sortedColors];

        return {
            width,
            height,
            bgColor,
            colorRange
        };
    }

    ///
    /// heatmap chart with mean values, 80% confidence interval,95% confidence interval
    ///
    /**
     * Draws an SVG chart with heatmap, mean, and confidence intervals, including legends.
     * This is the main public method that orchestrates the entire chart drawing process.
     * @param {string} csvTableData - The CSV data as a string.
     * @param {string} title - The title of the chart.
     * @param {string} xlabel - The label for the X-axis.
     * @param {Array<number>} xticks - The tick values for the X-axis.
     * @param {string} ylabel - The label for the Y-axis.
     * @param {Array<number>} yticks - The tick values for the Y-axis.
     * @param {object} allStyles - An object containing styling information like width, height, colors, etc.
     * @param {Array<string>} comments - An array of strings to be displayed as comments at the top.
     */
    drawSVGChartWithStyles(csvTableData, title, xlabel, xticks, ylabel, yticks, allStyles, comments) {
        // Section 1: Setup and Data Processing
        const width = parseFloat(allStyles.width);
        const height = parseFloat(allStyles.height);
        const svg = this._setupSVGCanvas(width, height, allStyles.bgColor);
        const rows = csvTableData.trim().split("\n").map(r => r.split(",").map(c => parseFloat(c.replace(/[^0-g.\\-]/g, "")) || 0));
        if (!rows.length || !rows[0].length) return;

        // d3.js use domain and range to map data to screen coordinates  
        svg.attr("is_logical_mapping_local", "true");

        // Section 2: Draw Header Content
        const commentsBottomY = this._drawComments(svg, comments);
        this._drawTitle(svg, title, width, commentsBottomY);

        // Section 3: Define Geometry, Domains, and Scales
        const margin = { top: commentsBottomY + 50, right: 40, bottom: 80, left: 80 };
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;
        const { xDomain, yDomain } = this._calculateDomains(xticks, yticks);
        const xScale = d3.scaleLinear().domain(xDomain).range([0, innerW]);
        const yScale = d3.scaleLinear().domain(yDomain).range([0, innerH]);

        // Section 4: Create Color Scale and Draw Legends
        const flatData = this._flattenData(rows);
        const maxProb = d3.max(flatData, d => d.value);
        const { colorScale, sampledDomain } = this._createColorScale(allStyles.colorRange, maxProb);
        this._drawLegends(svg, width, colorScale, maxProb);

        // Section 5: Create Main Group and Apply Logical Mappings
        const g = this._createMainGroup(svg, margin, xDomain, yDomain, innerW, innerH);
        this._addGlobalLogicalMapping(svg, xDomain, yDomain, width, height, margin, innerW, innerH);

        // Section 6: Draw Chart Elements
        const cellW = innerW / xticks.length;
        const cellH = innerH / yticks.length;
        this._drawHeatmap(g, flatData, cellW, cellH, colorScale, sampledDomain);
        const lineData = this._processLineData(rows, yticks, xticks);
        const lineGen = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
        this._drawIntervalLines(g, lineData, lineGen);
        this._drawMeanMarkers(g, lineData.meanPts, xScale, yScale, cellW);

        // Section 7: Draw Axes and Final Labels
        this._drawAxes(g, xScale, yScale, xticks, yticks, innerH);
        this._drawAxisLabels(svg, xlabel, ylabel, width, height);
    }

    /**
     * Initializes the SVG canvas, clearing any previous content.
     * @private
     */
    _setupSVGCanvas(width, height, bgColor) {
        const container = d3.select("#new-svg-container");
        container.html("");
        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", `0 0 ${width} ${height}`)
            .style("background-color", bgColor);
        return svg;
    }

    /**
     * Draws the comment block at the top-left of the SVG.
     * @private
     */
    _drawComments(svg, comments) {
        const commentLineHeight = 14;
        let commentsBottomY = 15;

        if (comments && comments.length > 0) {
            // Extract date from comments
            const dateRegex = /Simulation Start Date:\s*([A-Za-z]+ \d{1,2}, \d{2}(?:\.\d+)?)/;
            const match = comments.find(line => dateRegex.test(line));
            const rawdate = match ? match.match(dateRegex)[1] : null;
           
            // Normalize date by removing any periods
            if (rawdate) {
                const normalizedDate = rawdate.replace('.', '');
                console.log(normalizedDate); // Output: "September 12, 2025"
                svg.attr("x_start_date", normalizedDate)
                .attr("x_scale_days", 365);
            }

            svg.append("g")
                .attr("transform", `translate(15, 25)`)
                .selectAll("text")
                .data(comments)
                .enter()
                .append("text")
                .attr("y", (d, i) => i * commentLineHeight)
                .text(d => d)
                .style("font-size", "11px");

            commentsBottomY = (comments.length * commentLineHeight) + 30;
        }

        return commentsBottomY;
    }

    /**
     * Draws the main chart title.
     * @private
     */
    _drawTitle(svg, title, width, commentsBottomY) {
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", commentsBottomY + 20)
            .attr("text-anchor", "middle")
            .style("font-size", "18px")
            .text(title);
    }

    /**
     * Draws both the interactive line-style and the interactive heatmap color legends.
     * @private
     */
    _drawLegends(svg, width, colorScale, maxProb) {
        // BUG FIX: Add prefix to make labels globally meaningful and add a stable 'id' for logic.
        const prefix = "UST 10Y HJM Simulation ";
        const legendData = [
            { id: "mean",   label: prefix + "Mean",   color: "darkred", strokeWidth: 2, strokeDasharray: "none" },
            { id: "ci_80",  label: prefix + "80% CI", color: "orange",  strokeWidth: 2, strokeDasharray: "5,3"  },
            { id: "ci_95",  label: prefix + "95% CI", color: "red",     strokeWidth: 2, strokeDasharray: "2,2"  },
        ];

        const legend = svg.append("g")
            .attr("class", "legend")
            // BUG FIX: Shift legend left to make space for longer text.
            .attr("transform", `translate(${width - 450}, 25)`);
        const legendItems = legend.selectAll(".legend-item")
            .data(legendData)
            .enter()
            .append("g")
            .attr("class", "legend-item")
            .attr("transform", (d, i) => `translate(0, ${i * 20})`)
            // SVGX MODIFICATION START
            .attr("lc_legend_instance", d => this._generateLcId(d.label))
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", d => `${d.id}-curve`) // Use stable id for logic
            .style("cursor", "pointer");
        legendItems.append("line")
            .attr("x1", 0)
            .attr("x2", 30)
            .attr("y1", 10)
            .attr("y2", 10)
            .style("stroke", d => d.color)
            .style("stroke-width", d => d.strokeWidth)
            .style("stroke-dasharray", d => d.strokeDasharray);
        legendItems.append("text")
            .attr("x", 40)
            .attr("y", 10)
            .attr("dy", "0.35em")
            .style("font-size", "12px")
            // SVGX MODIFICATION START
            .attr("lc_legend_id", d => this._generateLcId(d.label))
            // SVGX MODIFICATION END
            .text(d => d.label);

        // BUG FIX: Update flashPath to use the stable 'id' property instead of the changing 'label'.
        function flashPath(event, d) {
            const originalStrokeWidth = 2;
            const flashStrokeWidth = 6;
            let selection;
            switch (d.id) { // <-- Use stable d.id for logic
                case "mean":
                    selection = d3.select("#mean-path");
                    break;
                case "ci_80":
                    selection = d3.selectAll(".ci-80-path");
                    break;
                case "ci_95":
                    selection = d3.selectAll(".ci-95-path");
                    break;
            }
            if (selection) {
                selection.transition()
                    .duration(150)
                    .attr("stroke-width", flashStrokeWidth)
                    .transition()
                    .duration(150)
                    .attr("stroke-width", originalStrokeWidth)
                    .transition()
                    .duration(150)
                    .attr("stroke-width", flashStrokeWidth)
                    .transition()
                    .duration(150)
                    .attr("stroke-width", originalStrokeWidth);
            }
        }
        legendItems.on("click", flashPath);

        const colorLegend = svg.append("g")
            .attr("class", "color-legend")
            .attr("transform", `translate(${width - 150}, 45)`);
        colorLegend.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .style("font-size", "12px")
            .text("Probability");
        
        const colorLegendWidth = 120;
        
        const gradient = svg.append("defs")
            .append("linearGradient")
            .attr("id", "color-gradient")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");
        gradient.selectAll("stop")
            .data(colorScale.range())
            .enter()
            .append("stop")
            .attr("offset", (d, i) => i / (colorScale.range().length - 1))
            .attr("stop-color", d => d);
            
        colorLegend.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", colorLegendWidth)
            .attr("height", 10)
            .style("fill", "url(#color-gradient)");

        const numSegments = 4;
        const segmentWidth = colorLegendWidth / numSegments;
        const interactiveSegments = d3.range(numSegments);

        colorLegend.selectAll(".hitbox")
            .data(interactiveSegments)
            .enter()
            .append("rect")
            .attr("class", "hitbox")
            .attr("x", (d, i) => i * segmentWidth)
            .attr("y", 0)
            .attr("width", segmentWidth)
            .attr("height", 10)
            .attr("fill", "transparent")
            .attr("data-logical-group-id", (d, i) => `heatmap_${i}`)
            .style("cursor", "pointer")
            .on("click", function(event, d) {
                const index = d;
                const selection = d3.selectAll(`.heatmap-group-${index}`);
                if (selection) {
                    selection.selectAll('rect')
                        .transition()
                        .duration(150)
                        .attr("stroke", "yellow")
                        .attr("stroke-width", 2)
                        .transition()
                        .duration(150)
                        .attr("stroke", "#ccc")
                        .attr("stroke-width", 1)
                        .transition()
                        .duration(150)
                        .attr("stroke", "yellow")
                        .attr("stroke-width", 2)
                        .transition()
                        .duration(150)
                        .attr("stroke", "#ccc")
                        .attr("stroke-width", 1);
                }
            });

        colorLegend.append("text")
            .attr("x", 0)
            .attr("y", 25)
            .style("font-size", "11px")
            .text("0");
        colorLegend.append("text")
            .attr("x", colorLegendWidth)
            .attr("y", 25)
            .attr("text-anchor", "end")
            .style("font-size", "11px")
            .text(maxProb.toFixed(2));
    }

    /**
     * Calculates the X and Y data domains for the scales.
     * @private
     */
    _calculateDomains(xticks, yticks) {
        const xVals = xticks.map(v => +v);
        const yVals = yticks.map(v => +v);
        const xInterval = (xVals[xVals.length - 1] - xVals[0]) / (xVals.length - 1);
        const xDomain = [xVals[0] - xInterval / 2, xVals[xVals.length - 1] + xInterval / 2];
        const yMin = d3.min(yVals);
        const yMax = d3.max(yVals);
        const yInterval = (yMax - yMin) / (yVals.length - 1);
        const yDomain = [yMax + yInterval / 2, yMin - yInterval / 2];
        return { xDomain, yDomain };
    }

    /**
     * Creates the D3 color scale and its domain boundaries for the heatmap.
     * @private
     */
    _createColorScale(colorRange, maxProb) {
        let finalColorRange = colorRange;
        if (colorRange.length > 5) {
            finalColorRange = [];
            for (let i = 0; i < 5; i++) {
                const index = Math.round(i * (colorRange.length - 1) / 4);
                finalColorRange.push(colorRange[index]);
            }
        }
        const sampledDomain = d3.range(finalColorRange.length)
            .map(i => d3.interpolate(0, maxProb)(i / (finalColorRange.length - 1)));
        const colorScale = d3.scaleLinear()
            .domain(sampledDomain)
            .range(finalColorRange);
        return { colorScale, sampledDomain };
    }

    /**
     * Flattens the 2D row data into a 1D array for D3 binding.
     * @private
     */
    _flattenData(rows) {
        const flat = [];
        rows.forEach((row, yi) =>
            row.forEach((v, xi) => flat.push({ x: xi, y: yi, value: v }))
        );
        return flat;
    }

    /**
     * Gets the color index (0-3) for a given value based on the domain breaks.
     * @private
     */
    _getQuantizeIndex(value, domain) {
        // domain has 5 points, creating 4 intervals [d0-d1, d1-d2, d2-d3, d3-d4]
        // We map these to indices 0, 1, 2, 3 respectively.
        if (value <= domain[1]) {
            return 0; // First interactive group
        }
        if (value <= domain[2]) {
            return 1; // Second interactive group
        }
        if (value <= domain[3]) {
            return 2; // Third interactive group
        }
        // Anything greater than domain[3] falls in the last group.
        return 3; // Fourth interactive group
    }

    /**
     * Adds the Global Logical Mapping attribute (xlm, ylm) to the root SVG.
     * @private
     */
    _addGlobalLogicalMapping(svg, xDomain, yDomain, width, height, margin, innerW, innerH) {
        svg.attr("xlm", `[${xDomain[0]}, ${xDomain[1]}, ${margin.left}, ${margin.left + innerW}]`)
            .attr("ylm", `[${yDomain[1]}, ${yDomain[0]}, ${margin.top}, ${margin.top + innerH}]`);
    }

    /**
     * Creates the main <g> element for the chart and adds Local Logical Mappings.
     * @private
     */
    _createMainGroup(svg, margin, xDomain, yDomain, innerW, innerH) {
        return svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`)
            .attr("xlm", `[${xDomain[0]}, ${xDomain[1]}, 0, ${innerW}]`)
            .attr("ylm", `[${yDomain[1]}, ${yDomain[0]}, 0, ${innerH}]`);
    }

    /**
     * Draws the heatmap grid and assigns discrete logical group IDs to cells.
     * @private
     */
    _drawHeatmap(g, flatData, cellW, cellH, colorScale, sampledDomain) {
        const cells = g.selectAll('.cell')
            .data(flatData)
            .enter()
            .append('g')
            .attr('class', d => `cell heatmap-group-${this._getQuantizeIndex(d.value, sampledDomain)}`)
            .attr("data-logical-group-id", d => `heatmap_${this._getQuantizeIndex(d.value, sampledDomain)}`)
            .attr('transform', d => `translate(${d.x * cellW}, ${d.y * cellH})`);
        cells.append('rect')
            .attr('width', cellW)
            .attr('height', cellH)
            .attr('fill', d => colorScale(d.value))
            .attr('stroke', '#ccc')
            .attr('stroke-width', 1);
        cells.append('foreignObject')
            .attr('width', cellW)
            .attr('height', cellH)
            .style('pointer-events', 'none')
            .append('xhtml:div')
            .style('width', '100%')
            .style('height', '100%')
            .style('box-sizing', 'border-box')
            .style('display', 'flex')
            .style('align-items', 'center')
            .style('justify-content', 'flex-end')
            .style('padding-right', '5px')
            .style('font-size', '12px')
            .style('color', 'black')
            .html(d => d.value.toFixed(2));
    }

    /**
     * Processes raw data to calculate mean and confidence interval points.
     * @private
     */
    _processLineData(rows, yticks, xticks) {
        const yVals = yticks.map(v => +v);
        const xVals = xticks.map(v => +v);
        const meanPts = [],
            lower80Pts = [],
            upper80Pts = [],
            lower95Pts = [],
            upper95Pts = [];
        for (let xi = 0; xi < xticks.length; xi++) {
            const col = rows.map((row, yi) => ({ y: yVals[yi], pRaw: row[xi] }));
            const total = d3.sum(col, d => d.pRaw);
            col.forEach(d => d.p = total > 0 ? d.pRaw / total : 0);
            const mu = d3.sum(col, d => d.y * d.p);
            const asc = col.slice().sort((a, b) => a.y - b.y);
            let cum = 0,
                l80 = asc[0].y,
                u80 = asc[asc.length - 1].y,
                l95 = asc[0].y,
                u95 = asc[asc.length - 1].y;
            for (const d of asc) {
                cum += d.p;
                if (cum >= 0.10 && l80 === asc[0].y) l80 = d.y;
                if (cum >= 0.90 && u80 === asc[asc.length - 1].y) u80 = d.y;
                if (cum >= 0.025 && l95 === asc[0].y) l95 = d.y;
                if (cum >= 0.975 && u95 === asc[asc.length - 1].y) u95 = d.y;
            }
            meanPts.push({ x: xVals[xi], y: mu });
            lower80Pts.push({ x: xVals[xi], y: l80 });
            upper80Pts.push({ x: xVals[xi], y: u80 });
            lower95Pts.push({ x: xVals[xi], y: l95 });
            upper95Pts.push({ x: xVals[xi], y: u95 });
        }
        return { meanPts, lower80Pts, upper80Pts, lower95Pts, upper95Pts };
    }

    /**
     * Draws the mean and confidence interval paths.
     * @private
     */
    _drawIntervalLines(g, lineData, lineGen) {
        g.append("path")
            .datum(lineData.meanPts)
            .attr("d", lineGen)
            .attr("fill", "none")
            .attr("stroke", "darkred")
            .attr("stroke-width", 2)
            .attr("id", "mean-path")
            // SVGX MODIFICATION START
            .attr("lc_legend_ref", `["${this._generateLcId('UST 10Y HJM Simulation Mean')}"]`)
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", "mean-curve");
        g.append("path")
            .datum(lineData.lower80Pts)
            .attr("d", lineGen)
            .attr("fill", "none")
            .attr("stroke", "orange")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,3")
            .attr("class", "ci-80-path")
            // SVGX MODIFICATION START
            .attr("lc_legend_ref", `["${this._generateLcId('UST 10Y HJM Simulation 80% CI')}"]`)
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", "ci-80-curve");
        g.append("path")
            .datum(lineData.upper80Pts)
            .attr("d", lineGen)
            .attr("fill", "none")
            .attr("stroke", "orange")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,3")
            .attr("class", "ci-80-path")
            // SVGX MODIFICATION START
            .attr("lc_legend_ref", `["${this._generateLcId('UST 10Y HJM Simulation 80% CI')}"]`)
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", "ci-80-curve");
        g.append("path")
            .datum(lineData.lower95Pts)
            .attr("d", lineGen)
            .attr("fill", "none")
            .attr("stroke", "red")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "2,2")
            .attr("class", "ci-95-path")
            // SVGX MODIFICATION START
            .attr("lc_legend_ref", `["${this._generateLcId('UST 10Y HJM Simulation 95% CI')}"]`)
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", "ci-95-curve");
        g.append("path")
            .datum(lineData.upper95Pts)
            .attr("d", lineGen)
            .attr("fill", "none")
            .attr("stroke", "red")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "2,2")
            .attr("class", "ci-95-path")
            // SVGX MODIFICATION START
            .attr("lc_legend_ref", `["${this._generateLcId('UST 10Y HJM Simulation 95% CI')}"]`)
            // SVGX MODIFICATION END
            .attr("data-logical-group-id", "ci-95-curve");
    }

    /**
     * Draws the markers and text labels on the mean line.
     * @private
     */
    _drawMeanMarkers(g, meanPts, xScale, yScale, cellW) {
        g.selectAll(".mean-point")
            .data(meanPts)
            .enter()
            .append("circle")
            .attr("class", "mean-point")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 3)
            .attr("fill", "darkred")
            .append("title")
            .text(d => `x: ${d.x.toFixed(2)}, y: ${d.y.toFixed(2)}`);
        g.selectAll(".mean-text")
            .data(meanPts)
            .enter()
            .append("text")
            .attr("class", "mean-text")
            .attr("x", d => xScale(d.x) - (cellW / 4))
            .attr("y", d => yScale(d.y) - 5)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .style("fill", "darkred")
            .text(d => d.y.toFixed(2));
    }

    /**
     * Draws the X and Y axes onto the chart group.
     * @private
     */
    _drawAxes(g, xScale, yScale, xVals, yVals, innerH) {
        const xAxisG = g.append("g")
            .attr("transform", `translate(0,${innerH})`);
        const xAxis = d3.axisBottom(xScale)
            .tickValues(xVals)
            .tickFormat(d3.format(".1f"));
        xAxisG.call(xAxis)
            .selectAll("text")
            .style("fill", "gray");
        const yAxisG = g.append("g");
        const yAxis = d3.axisLeft(yScale)
            .tickValues(yVals)
            .tickFormat(d3.format(".1f"));
        yAxisG.call(yAxis)
            .selectAll("text")
            .style("fill", "gray");
    }

    /**
     * Draws the X and Y axis labels onto the main SVG.
     * @private
     */
    _drawAxisLabels(svg, xlabel, ylabel, width, height) {
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height - 10)
            .attr("text-anchor", "middle")
            .style("font-family", "sans-serif")
            .style("font-size", "12px")
            .text(xlabel);
        svg.append("text")
            .attr("transform", `translate(20,${height / 2}) rotate(-90)`)
            .attr("text-anchor", "middle")
            .style("font-family", "sans-serif")
            .style("font-size", "12px")
            .text(ylabel);
    }
}