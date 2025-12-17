/**
 * 3. SVGXCrosshair
 * Responsible for rendering a visual crosshair and displaying logical values.
 * UPDATED: Uses getScreenCTM() for both cursor logic AND visual overlay positioning.
 */
class SVGXCrosshair {
    constructor(svgElement, chartArea = null) {
        this.svg = svgElement;
        this.chartArea = chartArea;
        this.settings = {
            enabled: true,
            lineColor: '#808080',
            labelBg: '#ffff00',
            labelColor: '#000000',
            numberPrecision: 2
        };

        this.overlay = null;
        this.rafId = null;
        // Reuse an SVG point object for performance
        this._pt = this.svg.createSVGPoint();

        this._boundMouseMove = this._handleMouseMove.bind(this);
        this._boundMouseLeave = this._handleMouseLeave.bind(this);

        this._createDOM();
        this._bindEvents();
    }

    dispose() {
        if (this.rafId) cancelAnimationFrame(this.rafId);
        window.removeEventListener('mousemove', this._boundMouseMove);
        if (this.overlay && this.overlay.parentNode) {
            this.overlay.parentNode.removeChild(this.overlay);
        }
    }

    _createDOM() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'svgx-crosshair-overlay';
        Object.assign(this.overlay.style, {
            pointerEvents: 'none',
            position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
            zIndex: '9999', display: 'none'
        });

        this.xLine = this._createLine('vertical');
        this.yLine = this._createLine('horizontal');
        this.xLabel = this._createLabel();
        this.yLabel = this._createLabel();

        this.overlay.append(this.xLine, this.yLine, this.xLabel, this.yLabel);
        document.body.appendChild(this.overlay);
    }

    _createLine(type) {
        const div = document.createElement('div');
        Object.assign(div.style, {
            position: 'absolute', backgroundColor: 'transparent',
            borderStyle: 'dashed', borderWidth: '0',
            borderColor: this.settings.lineColor, opacity: '0.8',
            top: '0', left: '0'
        });
        if (type === 'vertical') { div.style.borderLeftWidth = '1px'; div.style.height = '100%'; }
        else { div.style.borderTopWidth = '1px'; div.style.width = '100%'; }
        return div;
    }

    _createLabel() {
        const div = document.createElement('div');
        Object.assign(div.style, {
            position: 'absolute', padding: '2px 5px',
            backgroundColor: this.settings.labelBg, color: this.settings.labelColor,
            border: '1px solid #454545', borderRadius: '3px',
            fontSize: '12px', fontFamily: 'monospace', zIndex: '10000',
            whiteSpace: 'nowrap', pointerEvents: 'none'
        });
        return div;
    }

    _bindEvents() {
        window.addEventListener('mousemove', this._boundMouseMove, { passive: true });
        document.addEventListener('mouseleave', this._boundMouseLeave);
    }

    _handleMouseLeave() { if (this.overlay) this.overlay.style.display = 'none'; }

    _handleMouseMove(e) {
        if (!this.settings.enabled || !this.svg.isConnected) return;
        if (this.rafId) cancelAnimationFrame(this.rafId);
        this.rafId = requestAnimationFrame(() => this._render(e.clientX, e.clientY));
    }

    _render(clientX, clientY) {
        // --- STEP 1: Calculate Logical SVG Coordinate ---
        // Transform screen pixel -> SVG User Units
        this._pt.x = clientX;
        this._pt.y = clientY;

        let cursor;
        let ctm;
        try {
            ctm = this.svg.getScreenCTM();
            cursor = this._pt.matrixTransform(ctm.inverse());
        } catch (e) {
            return;
        }

        // --- STEP 2: Bounds Check (Logical) ---
        // Determine the chart area in SVG User Units
        let bounds = {
            minX: 0, maxX: 0, minY: 0, maxY: 0
        };

        const vb = this.svg.viewBox.baseVal;

        if (this.chartArea && this.chartArea.visualX && this.chartArea.visualY) {
            bounds.minX = Math.min(this.chartArea.visualX[0], this.chartArea.visualX[1]);
            bounds.maxX = Math.max(this.chartArea.visualX[0], this.chartArea.visualX[1]);
            bounds.minY = Math.min(this.chartArea.visualY[0], this.chartArea.visualY[1]);
            bounds.maxY = Math.max(this.chartArea.visualY[0], this.chartArea.visualY[1]);
        } else {
            bounds.minX = vb.x;
            bounds.maxX = vb.x + vb.width;
            bounds.minY = vb.y;
            bounds.maxY = vb.y + vb.height;
        }

        // Strict Bounds Check (Cursor in SVG Units vs Bounds in SVG Units)
        if (cursor.x < bounds.minX || cursor.x > bounds.maxX ||
            cursor.y < bounds.minY || cursor.y > bounds.maxY) {
            this.overlay.style.display = 'none';
            return;
        }

        this.overlay.style.display = 'block';

        // --- STEP 3: Convert Chart Bounds to Screen Coordinates ---
        // Transform the chart area edges (SVG Units) back to Screen Pixels.
        // This accounts for "letterboxing" (preserveAspectRatio) automatically.

        // 1. Transform Top-Left of Chart Area
        this._pt.x = bounds.minX;
        this._pt.y = bounds.minY;
        const screenTL = this._pt.matrixTransform(ctm);

        // 2. Transform Bottom-Right of Chart Area
        this._pt.x = bounds.maxX;
        this._pt.y = bounds.maxY;
        const screenBR = this._pt.matrixTransform(ctm);

        // 3. Determine screen boundaries (handle potential axis flips)
        const screenMinX = Math.min(screenTL.x, screenBR.x);
        const screenMaxX = Math.max(screenTL.x, screenBR.x);
        const screenMinY = Math.min(screenTL.y, screenBR.y);
        const screenMaxY = Math.max(screenTL.y, screenBR.y);

        // 4. Clamp visual crosshair lines to these precise screen boundaries
        const clampedX = Math.max(screenMinX, Math.min(clientX, screenMaxX));
        const clampedY = Math.max(screenMinY, Math.min(clientY, screenMaxY));

        // Draw Lines
        this.xLine.style.transform = `translate(${clampedX}px, ${screenMinY}px)`;
        this.xLine.style.height = `${screenMaxY - screenMinY}px`;

        this.yLine.style.transform = `translate(${screenMinX}px, ${clampedY}px)`;
        this.yLine.style.width = `${screenMaxX - screenMinX}px`;

        // Update Labels with the PRECISE cursor position (SVG Units)
        this._updateLabels(clampedX, clampedY, cursor.x, cursor.y, screenMinX, screenMinY, screenMaxY);
    }

    _updateLabels(clientX, clientY, logicalSvgX, logicalSvgY, boundsLeft, boundsTop, boundsBottom) {
        const xlm = this._parseMapping('xlm');
        const ylm = this._parseMapping('ylm');

        if (!xlm || !ylm) return;

        // Map SVG User Units -> Chart Logical Units
        const logicalValX = this._toLogical(logicalSvgX, xlm);
        const logicalValY = this._toLogical(logicalSvgY, ylm);

        let xText = logicalValX.toFixed(this.settings.numberPrecision);
        if (logicalValX > 599266080000000000) xText = this._ticksToDate(logicalValX);

        this.xLabel.textContent = xText;
        this.yLabel.textContent = logicalValY.toFixed(this.settings.numberPrecision);

        // Position Labels
        this.xLabel.style.left = `${clientX}px`;
        this.xLabel.style.top = `${boundsBottom + 5}px`;
        this.xLabel.style.transform = 'translateX(-50%)';

        this.yLabel.style.left = `${boundsLeft - 5}px`;
        this.yLabel.style.top = `${clientY}px`;
        this.yLabel.style.transform = 'translate(-100%, -50%)';
    }

    _parseMapping(attr) {
        const val = this.svg.getAttribute(attr);
        try { return val ? (val.startsWith('[') ? JSON.parse(val) : val.split(',').map(Number)) : null; } catch (e) { return null; }
    }

    _toLogical(v, m) { return m[0] + (v - m[2]) * (m[1] - m[0]) / (m[3] - m[2]); }

    _ticksToDate(ticks) {
        try {
            const date = new Date((Math.floor(ticks / 10000000) - 62135596800) * 1000);
            return date.toISOString().replace('T', ' ').substring(0, 16);
        } catch (e) { return "Date Error"; }
    }
}