/**
 * 4. SVGXLineChartRendering (Facade)
 * Acts as a facade.
 * Manages TWO crosshair instances: one for the Original SVG, one for the New D3 SVG.
 */
class SVGXLineChartRendering {
    constructor(svgElement, clrContent = null) {
        this.svg = svgElement; // The Original SVG

        // State for crosshairs
        this._crosshairOriginal = null;
        this._crosshairNew = null;

        if (!this.svg) {
            throw new Error('SVG element is not defined.');
        }

        this.analyzer = new SVGXLineChartAnalyzer(this.svg, clrContent);
        console.log('[LOG] Line chart renderer initialized.');
    }

    addLogicalMapping() {
        // 1. Analyze Original SVG
        const result = this.analyzer.addLogicalMapping();

        // 2. Enable Crosshair on Original SVG
        this.enableOriginalCrosshair();

        return result;
    }

    addLegendInfo() { return this.analyzer.addLegendInfo(); }
    enableLegendInteractivity() { return this.analyzer.enableLegendInteractivity(); }

    extractLogicalData() {
        const chartArea = this.analyzer.getChartArea ? this.analyzer.getChartArea() : null;
        const extractor = new SVGXLineChartDataExtractor(this.svg, this.analyzer.legendItems, chartArea);

        extractor._getFullBBox = this.analyzer._getFullBBox.bind(this.analyzer);
        extractor._findNumericLabels = this.analyzer._findNumericLabels.bind(this.analyzer);
        extractor._findXAxisBottomLabels = this.analyzer._findXAxisBottomLabels.bind(this.analyzer);
        extractor._findYAxisLeftLabels = this.analyzer._findYAxisLeftLabels.bind(this.analyzer);
        extractor._findXAxisGridlines = this.analyzer._findXAxisGridlines.bind(this.analyzer);

        return extractor.extractLogicalData();
    }

    /**
     * Enables crosshair ONLY for the original Potrace SVG
     */
    enableOriginalCrosshair() {
        if (this._crosshairOriginal) {
            this._crosshairOriginal.dispose();
            this._crosshairOriginal = null;
        }

        const chartArea = this.analyzer.getChartArea ? this.analyzer.getChartArea() : null;

        if (typeof SVGXCrosshair !== 'undefined') {
            console.log("[LOG] Enabling Crosshair for Original SVG");
            this._crosshairOriginal = new SVGXCrosshair(this.svg, chartArea);
        }
    }

    redrawChart(input, containerSelector) {
        // 1. Draw the New Chart
        const renderer = new SVGXLineChartNewRendering(this.svg);
        renderer.redrawChart(input, containerSelector);

        // 2. Enable Crosshair on the New Chart
        this._enableNewCrosshair(containerSelector);
    }

    /**
     * Internal: Finds the newly created SVG and attaches a crosshair to it.
     */
    _enableNewCrosshair(containerSelector) {
        // Cleanup previous new-chart crosshair
        if (this._crosshairNew) {
            this._crosshairNew.dispose();
            this._crosshairNew = null;
        }

        const container = document.querySelector(containerSelector);
        if (!container) return;

        const newSvg = container.querySelector('svg');
        if (!newSvg) return;

        if (typeof SVGXCrosshair === 'undefined') return;

        console.log("[LOG] Enabling Crosshair for New D3 SVG");

        // 3. Derive Chart Area for the New SVG
        // FIX: Calculate Min/Max specifically to handle inverted Y axes (D3 standard)
        let newChartArea = null;

        try {
            const xlm = JSON.parse(newSvg.getAttribute("xlm") || "[]");
            const ylm = JSON.parse(newSvg.getAttribute("ylm") || "[]");

            if (xlm.length === 4 && ylm.length === 4) {
                newChartArea = {
                    // xlm/ylm format: [logMin, logMax, vis1, vis2]
                    // We normalize vis1 and vis2 to ensure [SmallestPixel, LargestPixel]
                    visualX: [Math.min(xlm[2], xlm[3]), Math.max(xlm[2], xlm[3])],
                    visualY: [Math.min(ylm[2], ylm[3]), Math.max(ylm[2], ylm[3])]
                };
                console.log(`[LOG] New Chart Area Boundaries: X[${newChartArea.visualX}], Y[${newChartArea.visualY}]`);
            }
        } catch (e) {
            console.warn("Could not parse XLM/YLM from new chart for crosshair boundaries.", e);
        }

        // 4. Create the instance
        this._crosshairNew = new SVGXCrosshair(newSvg, newChartArea);
    }

    dispose() {
        if (this._crosshairOriginal) this._crosshairOriginal.dispose();
        if (this._crosshairNew) this._crosshairNew.dispose();
    }
}