class SVGXPathSegments {
    constructor() {
        this.tokenizerRegex = /([MLHVCSQTAZmlhvcsqtaz])|([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/gi;
    }

    _getSegmentType(commandType) {
        // Commands that draw straight lines are classified as 'line'.
        if (['L', 'H', 'V', 'Z'].indexOf(commandType) !== -1) {
            return 'line';
        }
        
        // Everything else (C, S, Q, T, A) is assumed to be part of a 'glyph'.
        return 'glyph';
    }

    processPathData(dAttribute) {
        if (!dAttribute) {
            return [];
        }
        
        const state = {
            tokens: dAttribute.match(this.tokenizerRegex) || [],
            tokenIndex: 0,
            rawCommands: [], 
            currentX: 0, currentY: 0,
            subpathStartX: 0, subpathStartY: 0,
            lastControlX: null, lastControlY: null,
            lastCommandType: '',
        };

        let commandLetter = '';
        let params = [];
        while (state.tokenIndex < state.tokens.length) {
            const token = state.tokens[state.tokenIndex];
            if (isNaN(parseFloat(token))) {
                if (commandLetter) this._parseCommand(commandLetter, params, state);
                commandLetter = token;
                params = [];
            } else {
                params.push(parseFloat(token));
            }
            state.tokenIndex++;
        }
        if (commandLetter) this._parseCommand(commandLetter, params, state);

        return state.rawCommands.filter(function(cmd) {
            return cmd.commandType !== 'M';
        });
    }



    getFinalPathData(segments) {
        if (!segments || segments.length === 0) {
            return "";
        }

        let pathString = "";
        let lastEndPoint = null;

        // The first M is always taken from the very first segment in the original list.
        pathString += "M " + segments[0].startPoint.x + " " + segments[0].startPoint.y + " ";
        lastEndPoint = segments[0].startPoint;

        for (let i = 0; i < segments.length; i++) {
            const segment = segments[i];

            if (!segment.matched) {
                // This is an unmatched segment. We must ensure it connects to the last point.
                // Note: The command is absolute, so we don't need a moveto.
                pathString += this._rebuildCommandString(segment) + " ";
                lastEndPoint = segment.endPoint;
            } else {
                // This is the start of a matched group that needs to be replaced.
                let endOfGroupPoint = segment.endPoint;
                let groupSize = 1;
                
                // Find the end of the contiguous matched group.
                for (let j = i + 1; j < segments.length; j++) {
                    const nextSegment = segments[j];
                    const isConnected = (endOfGroupPoint.x === nextSegment.startPoint.x && endOfGroupPoint.y === nextSegment.startPoint.y);
                    
                    if (nextSegment.matched && isConnected) {
                        endOfGroupPoint = nextSegment.endPoint; // Update the final endpoint
                        groupSize++;
                    } else {
                        break; // End of the group
                    }
                }

                // ** THE FIX IS HERE **
                // Replace the entire group with a single 'lineto' command.
                // This connects the previous point (lastEndPoint) to the end of the whole group.
                pathString += "L " + endOfGroupPoint.x + " " + endOfGroupPoint.y + " ";
                
                lastEndPoint = endOfGroupPoint;

                // Advance the main loop counter to skip the segments we just processed.
                i += groupSize - 1;
            }
        }

        return pathString.trim();
    }

    getFinalPathDataArray(segments) {
        if (!segments || segments.length === 0) {
            return [];
        }

        const pathStrings = [];
        let currentPathString = "";
        let lastEndPoint = null;

        for (const segment of segments) {
            const currentStartPoint = segment.startPoint;

            if (lastEndPoint === null || lastEndPoint.x !== currentStartPoint.x || lastEndPoint.y !== currentStartPoint.y) {
                if (currentPathString) {
                    pathStrings.push(currentPathString.trim());
                }
                currentPathString = `M ${currentStartPoint.x} ${currentStartPoint.y} `;
            }

            currentPathString += this._rebuildCommandString(segment) + " ";
            lastEndPoint = segment.endPoint;
        }

        if (currentPathString) {
            pathStrings.push(currentPathString.trim());
        }

        return pathStrings;
    }

    _rebuildCommandString(segment) {
        const params = segment.absoluteParameters;
        switch (segment.commandType) {
            case 'L': return "L " + params[0] + " " + params[1];
            case 'H': return "H " + params[0];
            case 'V': return "V " + params[1];
            case 'C': return "C " + params[0] + " " + params[1] + ", " + params[2] + " " + params[3] + ", " + params[4] + " " + params[5];
            case 'S': return "S " + params[2] + " " + params[3] + ", " + params[4] + " " + params[5];
            case 'Q': return "Q " + params[0] + " " + params[1] + ", " + params[2] + " " + params[3];
            case 'T': return "T " + params[2] + " " + params[3];
            case 'A': return "A " + params[0] + " " + params[1] + " " + params[2] + " " + params[3] + " " + params[4] + " " + params[5] + " " + params[6];
            case 'Z': return 'Z';
            default: return '';
        }
    }

    _parseCommand(cmd, params, state) {
        const commandType = cmd.toUpperCase();
        const isRelative = cmd === cmd.toLowerCase();
        let pIndex = 0;
        const toAbs = function(val, axis) {
             return (axis === 'x' ? state.currentX : state.currentY) + val;
        };

        while (pIndex < params.length) {
            let finalSegmentX = state.currentX;
            let finalSegmentY = state.currentY;
            let tempLastControlX = null;
            let tempLastControlY = null;
            let segmentAbsoluteParameters = [];
            let segmentPoints = [];
            
            let effectiveLastControlX = (state.lastControlX !== null) ? state.lastControlX : state.currentX;
            let effectiveLastControlY = (state.lastControlY !== null) ? state.lastControlY : state.currentY;

            if (commandType === 'S' && ['C', 'S'].indexOf(state.lastCommandType) === -1) {
                effectiveLastControlX = state.currentX;
                effectiveLastControlY = state.currentY;
            }
            if (commandType === 'T' && ['Q', 'T'].indexOf(state.lastCommandType) === -1) {
                effectiveLastControlX = state.currentX;
                effectiveLastControlY = state.currentY;
            }
            
            let startPoint = { x: state.currentX, y: state.currentY };
            segmentPoints.push(startPoint);

            switch (commandType) {
                case 'M': {
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x, y);
                    finalSegmentX = x; finalSegmentY = y;
                    state.subpathStartX = x; state.subpathStartY = y;
                    break;
                }
                case 'L': {
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x, y);
                    segmentPoints.push({ x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    break;
                }
                case 'H': {
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    segmentAbsoluteParameters.push(x, state.currentY);
                    segmentPoints.push({ x, y: state.currentY });
                    finalSegmentX = x;
                    break;
                }
                case 'V': {
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(state.currentX, y);
                    segmentPoints.push({ x: state.currentX, y });
                    finalSegmentY = y;
                    break;
                }
                case 'C': {
                    const x1 = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y1 = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    const x2 = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y2 = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x1, y1, x2, y2, x, y);
                    segmentPoints.push({ x: x1, y: y1 }, { x: x2, y: y2 }, { x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    tempLastControlX = x2; tempLastControlY = y2;
                    break;
                }
                case 'S': {
                    const x1 = 2 * state.currentX - effectiveLastControlX;
                    const y1 = 2 * state.currentY - effectiveLastControlY;
                    const x2 = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y2 = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x1, y1, x2, y2, x, y);
                    segmentPoints.push({ x: x1, y: y1 }, { x: x2, y: y2 }, { x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    tempLastControlX = x2; tempLastControlY = y2;
                    break;
                }
                case 'Q': {
                    const x1 = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y1 = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x1, y1, x, y);
                    segmentPoints.push({ x: x1, y: y1 }, { x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    tempLastControlX = x1; tempLastControlY = y1;
                    break;
                }
                case 'T': {
                    const x1 = 2 * state.currentX - effectiveLastControlX;
                    const y1 = 2 * state.currentY - effectiveLastControlY;
                    const x = isRelative ? toAbs(params[pIndex++], 'x') : params[pIndex++];
                    const y = isRelative ? toAbs(params[pIndex++], 'y') : params[pIndex++];
                    segmentAbsoluteParameters.push(x1, y1, x, y);
                    segmentPoints.push({ x: x1, y: y1 }, { x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    tempLastControlX = x1; tempLastControlY = y1;
                    break;
                }
                case 'A': {
                    const rx = params[pIndex++];
                    const ry = params[pIndex++];
                    const rot = params[pIndex++];
                    const large = params[pIndex++];
                    const sweep = params[pIndex++];
                    const x_raw = params[pIndex++];
                    const y_raw = params[pIndex++];
                    const x = isRelative ? toAbs(x_raw, 'x') : x_raw;
                    const y = isRelative ? toAbs(y_raw, 'y') : y_raw;
                    segmentAbsoluteParameters.push(rx, ry, rot, large, sweep, x, y);
                    segmentPoints.push({ x, y });
                    finalSegmentX = x; finalSegmentY = y;
                    break;
                }
                case 'Z': {
                    finalSegmentX = state.subpathStartX;
                    finalSegmentY = state.subpathStartY;
                    segmentPoints.push({ x: finalSegmentX, y: finalSegmentY });
                    pIndex = params.length;
                    break;
                }
            }

            const segmentObject = {
                originalCommand: cmd,
                commandType: commandType,
                segmentType: this._getSegmentType(commandType),
                startPoint: startPoint,
                endPoint: { x: finalSegmentX, y: finalSegmentY },
                points: segmentPoints,
                absoluteParameters: segmentAbsoluteParameters,
                matched: false
            };
            // The object is NOT frozen, so the 'matched' flag can be modified.
            state.rawCommands.push(segmentObject);

            state.currentX = finalSegmentX;
            state.currentY = finalSegmentY;
            state.lastControlX = tempLastControlX;
            state.lastControlY = tempLastControlY;
            state.lastCommandType = commandType;

            if (commandType === 'M' && pIndex < params.length) {
                cmd = isRelative ? 'l' : 'L';
            }
        }
    }
}