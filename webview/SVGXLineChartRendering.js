///
/// Line Chart Rendering using D3.js
///

/**
 * 4. SVGXLineChartRendering (Facade)
 * Acts as a facade, orchestrating the calls to the other three new classes
 * to ensure the external behavior remains absolutely identical.
 */
class SVGXLineChartRendering {
    constructor(svgElement, clrContent = null) {
        this.svg = svgElement;
        if (!this.svg) {
            throw new Error('SVG element is not defined.');
        }
        // Initialize Analyzer
        this.analyzer = new SVGXLineChartAnalyzer(this.svg, clrContent);
        console.log('[LOG] Line chart renderer initialized (Refactored)');
    }

    addLogicalMapping() {
        return this.analyzer.addLogicalMapping();
    }

    addLegendInfo() {
        return this.analyzer.addLegendInfo();
    }

    enableLegendInteractivity() {
        return this.analyzer.enableLegendInteractivity();
    }

    extractLogicalData() {
        // Get the accurate chart area from the analyzer
        const chartArea = this.analyzer.getChartArea ? this.analyzer.getChartArea() : null;

        const extractor = new SVGXLineChartDataExtractor(
            this.svg, 
            this.analyzer.legendItems,
            chartArea // Pass the chart area to the extractor
        );
        
        // Dependency Injection: Helper methods required by Extractor that live in Analyzer
        extractor._getFullBBox = this.analyzer._getFullBBox.bind(this.analyzer);
        extractor._findNumericLabels = this.analyzer._findNumericLabels.bind(this.analyzer);
        
        // These are needed by extractLogicalData to determine fallback bounds
        extractor._findXAxisBottomLabels = this.analyzer._findXAxisBottomLabels.bind(this.analyzer);
        extractor._findYAxisLeftLabels = this.analyzer._findYAxisLeftLabels.bind(this.analyzer);
        extractor._findXAxisGridlines = this.analyzer._findXAxisGridlines.bind(this.analyzer);

        return extractor.extractLogicalData();
    }

    redrawChart(input, containerSelector) {
        const renderer = new SVGXLineChartNewRendering(this.svg);
        return renderer.redrawChart(input, containerSelector);
    }
}