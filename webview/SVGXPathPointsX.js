/**
 * A class to parse SVG path data and transform points.
 */
class SVGXPathPointsX {
    constructor() {
        // Regex for tokenizing the SVG 'd' attribute string.
        this.tokenizerRegex = /([MLHVCSQTAZmlhvcsqtaz])|([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/gi;

        // A single regex that can find 'translate' or 'scale' transforms.
        // This is not used by _processPathData but may be used by other methods.
        const numberPattern = "[+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?";
        this.transformRegex = new RegExp(
            `(translate|scale)\\s*\\(\\s*(${numberPattern})\\s*(?:[\\s,]+\\s*(${numberPattern}))?\\s*\\)`, "g"
        );
    }

    /**
     * Processes an SVG path 'd' attribute string and returns a detailed breakdown of its segments.
     * @param {string} dAttribute - The SVG path 'd' attribute string.
     * @returns {Array<object>} An array of segment objects with parsed information.
     * @private
     */
    _processPathData(dAttribute) {
        if (!dAttribute) {
            return [];
        }

        // State object to manage the parser's context.
        const state = {
            d: dAttribute,
            tokens: dAttribute.match(this.tokenizerRegex) || [],
            tokenIndex: 0,
            
            resolvedSegments: [],

            currentX: 0,
            currentY: 0,
            subpathStartX: 0,
            subpathStartY: 0,
            lastControlX: null,
            lastControlY: null,
            lastCommandType: '',
        };

        let commandLetter = '';
        let params = [];

        while (state.tokenIndex < state.tokens.length) {
            const token = state.tokens[state.tokenIndex];
            const isCommand = isNaN(parseFloat(token));

            if (isCommand) {
                if (commandLetter) {
                    // Process the previously collected command and its parameters
                    this._parseSegment(commandLetter, params, state);
                }
                commandLetter = token;
                params = [];
            } else {
                params.push(parseFloat(token));
            }
            state.tokenIndex++;
        }

        // Process the last command in the string
        if (commandLetter) {
            this._parseSegment(commandLetter, params, state);
        }

        return state.resolvedSegments;
    }

    /**
     * Parses a single command segment (e.g., "C 10,20 30,40 50,60") and updates the parser state.
     * @param {string} cmd - The command letter (e.g., 'M', 'c').
     * @param {number[]} params - The array of numeric parameters for the command.
     * @param {object} state - The mutable state object of the parser.
     * @private
     */
    _parseSegment(cmd, params, state) {
        const commandType = cmd.toUpperCase();
        const isRelative = cmd === cmd.toLowerCase();
        let pIndex = 0;

        const toAbsolute = (val, axis) =>
            axis === 'x' ? state.currentX + val : state.currentY + val;

        while (pIndex < params.length) {
            const segmentAbsolutePoints = [];
            const segmentAbsoluteParameters = [];
            let finalSegmentX = state.currentX;
            let finalSegmentY = state.currentY;
            let tempLastControlX = null;
            let tempLastControlY = null;

            let effectiveLastControlX = state.lastControlX ?? state.currentX;
            let effectiveLastControlY = state.lastControlY ?? state.currentY;

            if (commandType === 'S' && !['C', 'S'].includes(state.lastCommandType)) {
                effectiveLastControlX = state.currentX;
                effectiveLastControlY = state.currentY;
            }
            if (commandType === 'T' && !['Q', 'T'].includes(state.lastCommandType)) {
                effectiveLastControlX = state.currentX;
                effectiveLastControlY = state.currentY;
            }

            switch (commandType) {
                case 'M': {
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    state.subpathStartX = x;
                    state.subpathStartY = y;
                    state.lastControlX = null;
                    state.lastControlY = null;
                    break;
                }
                case 'L': {
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    break;
                }
                case 'H': {
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = state.currentY;
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    break;
                }
                case 'V': {
                    const x = state.currentX;
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    break;
                }
                case 'C': {
                    const x1 = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y1 = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    const x2 = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y2 = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push(
                        { x: x1, y: y1, subType: 'control1' },
                        { x: x2, y: y2, subType: 'control2' },
                        { x, y, subType: 'endpoint' }
                    );
                    segmentAbsoluteParameters.push(x1, y1, x2, y2, x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    tempLastControlX = x2;
                    tempLastControlY = y2;
                    break;
                }
                case 'S': {
                    const x1 = 2 * state.currentX - effectiveLastControlX;
                    const y1 = 2 * state.currentY - effectiveLastControlY;
                    const x2 = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y2 = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push(
                        { x: x1, y: y1, subType: 'reflectedControl1' },
                        { x: x2, y: y2, subType: 'control2' },
                        { x, y, subType: 'endpoint' }
                    );
                    segmentAbsoluteParameters.push(x1, y1, x2, y2, x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    tempLastControlX = x2;
                    tempLastControlY = y2;
                    break;
                }
                case 'Q': {
                    const x1 = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y1 = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push(
                        { x: x1, y: y1, subType: 'control1' },
                        { x, y, subType: 'endpoint' }
                    );
                    segmentAbsoluteParameters.push(x1, y1, x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    tempLastControlX = x1;
                    tempLastControlY = y1;
                    break;
                }
                case 'T': {
                    const x1 = 2 * state.currentX - effectiveLastControlX;
                    const y1 = 2 * state.currentY - effectiveLastControlY;
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push(
                        { x: x1, y: y1, subType: 'reflectedControl1' },
                        { x, y, subType: 'endpoint' }
                    );
                    segmentAbsoluteParameters.push(x1, y1, x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    tempLastControlX = x1;
                    tempLastControlY = y1;
                    break;
                }
                case 'A': {
                    const rx = params[pIndex++];
                    const ry = params[pIndex++];
                    const xAxisRotation = params[pIndex++];
                    const largeArcFlag = params[pIndex++] ? 1 : 0;
                    const sweepFlag = params[pIndex++] ? 1 : 0;
                    const x = isRelative ? toAbsolute(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbsolute(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(rx, ry, xAxisRotation, largeArcFlag, sweepFlag, x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    break;
                }
                case 'Z': {
                    const x = state.subpathStartX;
                    const y = state.subpathStartY;
                    segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x;
                    finalSegmentY = y;
                    tempLastControlX = null;
                    tempLastControlY = null;
                    pIndex = params.length;
                    break;
                }
            }

                    const segmentData = {
                originalCommand: cmd,
                commandType,
                isRelative,
                points: segmentAbsolutePoints,
                absoluteParameters: segmentAbsoluteParameters,
                startPoint: { x: state.currentX, y: state.currentY },
                endPoint: { x: finalSegmentX, y: finalSegmentY },
            };

            if (commandType === 'A') {
                segmentData.arcDetails = {
                    rx: segmentAbsoluteParameters[0],
                    ry: segmentAbsoluteParameters[1],
                    xAxisRotation: segmentAbsoluteParameters[2],
                    largeArcFlag: segmentAbsoluteParameters[3],
                    sweepFlag: segmentAbsoluteParameters[4],
                };
            }

            state.resolvedSegments.push(Object.freeze(segmentData));

            // Update parser state
            state.currentX = finalSegmentX;
            state.currentY = finalSegmentY;
            state.lastControlX = tempLastControlX;
            state.lastControlY = tempLastControlY;
            state.lastCommandType = commandType;

            // Handle implicit lineto chaining after moveto
            if (commandType === 'M' && pIndex < params.length) {
                cmd = isRelative ? 'l' : 'L';
            }
        }
    }
}