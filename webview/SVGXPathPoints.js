/**
 * A utility class for processing SVG path data and interacting with DOM elements.
 * It provides methods to extract points, handle transformations, and decompose composite paths.
 */
class SVGXPathPoints {
    constructor() {
        // Regex for tokenizing the SVG 'd' attribute string.
        this.tokenizerRegex = /([MLHVCSQTAZmlhvcsqtaz])|([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/gi;
            
        // A single regex that finds 'translate' or 'scale' sequentially AND validates their parameters are numbers.
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
        let match;
        let commandLetter = '';
        let params = [];
        let currentX = 0, currentY = 0, subpathStartX = 0, subpathStartY = 0;
        let lastControlX = null, lastControlY = null, lastCommandType = '';
        const resolvedSegments = [];
        let segmentString = ''; // To store the original string for the current segment

        const toAbsolute = (val, type) => (type === 'x' ? currentX + val : currentY + val);
        this.tokenizerRegex.lastIndex = 0;

        while ((match = this.tokenizerRegex.exec(dAttribute)) !== null) {
            if (match[1]) { // Command letter
                if (commandLetter) {
                    processCollectedCommand(commandLetter, params, segmentString.trim());
                }
                commandLetter = match[1];
                params = [];
                segmentString = commandLetter;
            } else if (match[2]) { // Parameter
                params.push(parseFloat(match[2]));
                segmentString += ` ${match[2]}`;
            }
        }

        if (commandLetter) {
            processCollectedCommand(commandLetter, params, segmentString.trim());
        }

        function processCollectedCommand(cmd, p, originalString) {
            const commandType = cmd.toUpperCase();
            const isRelative = cmd === cmd.toLowerCase();
            let pIndex = 0;

            while (pIndex < p.length) {
                const segmentAbsolutePoints = [];
                let segmentAbsoluteParameters = [];
                let finalSegmentX = currentX;
                let finalSegmentY = currentY;
                let tempLastControlX = null;
                let tempLastControlY = null;
                let effectiveLastControlX = lastControlX !== null ? lastControlX : currentX;
                let effectiveLastControlY = lastControlY !== null ? lastControlY : currentY;

                if (commandType === 'S' && (lastCommandType !== 'C' && lastCommandType !== 'S')) {
                    effectiveLastControlX = currentX;
                    effectiveLastControlY = currentY;
                }
                if (commandType === 'T' && (lastCommandType !== 'Q' && lastCommandType !== 'T')) {
                    effectiveLastControlX = currentX;
                    effectiveLastControlY = currentY;
                }

                switch (commandType) {
                    case 'M': {
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        subpathStartX = x;
                        subpathStartY = y;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        break;
                    }
                    case 'L': {
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        break;
                    }
                    case 'H': {
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = currentY;
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        break;
                    }
                    case 'V': {
                        const x = currentX;
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        break;
                    }
                    case 'C': {
                        const x1 = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y1 = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        const x2 = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y2 = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x: x1, y: y1, subType: 'control1' });
                        segmentAbsolutePoints.push({ x: x2, y: y2, subType: 'control2' });
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x1, y1, x2, y2, x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = x2;
                        tempLastControlY = y2;
                        break;
                    }
                    case 'S': {
                        const x1 = 2 * currentX - effectiveLastControlX;
                        const y1 = 2 * currentY - effectiveLastControlY;
                        const x2 = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y2 = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x: x1, y: y1, subType: 'reflectedControl1' });
                        segmentAbsolutePoints.push({ x: x2, y: y2, subType: 'control2' });
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x2, y2, x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = x2;
                        tempLastControlY = y2;
                        break;
                    }
                    case 'Q': {
                        const x1 = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y1 = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x: x1, y: y1, subType: 'control1' });
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x1, y1, x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = x1;
                        tempLastControlY = y1;
                        break;
                    }
                    case 'T': {
                        const x1 = 2 * currentX - effectiveLastControlX;
                        const y1 = 2 * currentY - effectiveLastControlY;
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x: x1, y: y1, subType: 'reflectedControl1' });
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = x1;
                        tempLastControlY = y1;
                        break;
                    }
                    case 'A': {
                        const rx = p[pIndex++];
                        const ry = p[pIndex++];
                        const xAxisRotation = p[pIndex++];
                        const largeArcFlag = p[pIndex++];
                        const sweepFlag = p[pIndex++];
                        const x = isRelative ? toAbsolute(p[pIndex++], 'x') : p[pIndex++];
                        const y = isRelative ? toAbsolute(p[pIndex++], 'y') : p[pIndex++];
                        segmentAbsolutePoints.push({ x, y, subType: 'endpoint' });
                        segmentAbsoluteParameters.push(rx, ry, xAxisRotation, largeArcFlag, sweepFlag, x, y);
                        finalSegmentX = x;
                        finalSegmentY = y;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        break;
                    }
                    case 'Z': {
                        segmentAbsolutePoints.push({ x: subpathStartX, y: subpathStartY, subType: 'endpoint' });
                        segmentAbsoluteParameters = [];
                        finalSegmentX = subpathStartX;
                        finalSegmentY = subpathStartY;
                        tempLastControlX = null;
                        tempLastControlY = null;
                        pIndex = p.length;
                        break;
                    }
                    default:
                        pIndex = p.length;
                        break;
                }

                resolvedSegments.push({
                    originalString: originalString,
                    originalCommand: cmd,
                    commandType: commandType,
                    isRelative: isRelative,
                    points: segmentAbsolutePoints,
                    absoluteParameters: segmentAbsoluteParameters,
                    startPoint: { x: currentX, y: currentY },
                    endPoint: { x: finalSegmentX, y: finalSegmentY },
                    arcDetails: commandType === 'A' ? {
                        rx: segmentAbsoluteParameters[0], ry: segmentAbsoluteParameters[1],
                        xAxisRotation: segmentAbsoluteParameters[2], largeArcFlag: segmentAbsoluteParameters[3],
                        sweepFlag: segmentAbsoluteParameters[4]
                    } : undefined,
                });

                currentX = finalSegmentX;
                currentY = finalSegmentY;
                lastControlX = tempLastControlX;
                lastControlY = tempLastControlY;
                lastCommandType = commandType;

                if (commandType === 'M' && pIndex < p.length) {
                    commandLetter = isRelative ? 'l' : 'L';
                    lastControlX = null;
                    lastControlY = null;
                }
            }
        }
        return resolvedSegments;
    }


    /**
     * Accumulates all 'translate' and 'scale' transformations up the DOM tree,
     * correctly respecting the order of operations and using strict validation.
     * @param {SVGElement} startElement - The SVG element to start from.
     * @returns {{totalTx: number, totalTy: number, totalSx: number, totalSy: number}}
     * @private
     */
    _accumulateTranslateAndScale(startElement) {
        let totalTx = 0, totalTy = 0, totalSx = 1, totalSy = 1;

        const elements = [];
        let currentElement = startElement;
        while (currentElement && currentElement.tagName !== 'svg' && currentElement.getAttribute) {
            elements.push(currentElement);
            currentElement = currentElement.parentNode;
        }
        if (currentElement && currentElement.tagName === 'svg') {
            elements.push(currentElement);
        }

        for (let i = elements.length - 1; i >= 0; i--) {
            const transformAttr = elements[i].getAttribute('transform');
            if (!transformAttr) {
                continue;
            }

            let match;
            this.transformRegex.lastIndex = 0;
            
            while ((match = this.transformRegex.exec(transformAttr)) !== null) {
                // match[1] is 'translate' or 'scale'
                // match[2] is the first number (string)
                // match[3] is the second number (string) or undefined

                const command = match[1];
                
                if (command === 'translate') {
                    const tx = parseFloat(match[2] || 0);
                    const ty = parseFloat(match[3] || 0);
                    totalTx += tx * totalSx;
                    totalTy += ty * totalSy;
                } else if (command === 'scale') {
                    const sx = parseFloat(match[2] || 1);
                    const sy = parseFloat(match[3] || sx);
                    totalSx *= sx;
                    totalSy *= sy;
                    totalTx *= sx;
                    totalTy *= sy;
                }
            }
        }
        return { totalTx, totalTy, totalSx, totalSy };
    }

    /**
     * Extracts all unique [x, y] coordinates from a path 'd' attribute.
     * @param {string} dAttribute - The SVG path 'd' attribute string.
     * @returns {Array<[number, number]>} An array of [x, y] coordinate pairs.
     */
    getPathPoints(dAttribute) {
        const resolvedSegments = this._processPathData(dAttribute);
        const allPoints = [];
        const addedPointsMap = new Set();
        const addPoint = (x, y) => {
            const pointKey = `${x},${y}`;
            if (!addedPointsMap.has(pointKey)) {
                allPoints.push([x, y]);
                addedPointsMap.add(pointKey);
            }
        };
        resolvedSegments.forEach(segment => {
            segment.points.forEach(point => {
                addPoint(point.x, point.y);
            });
        });
        return allPoints;
    }

    /**
     * Extracts all unique [x, y] coordinates from an SVG path element.
     * @param {SVGPathElement} pathElement - The SVG path element.
     * @returns {Array<[number, number]>} An array of [x, y] coordinate pairs.
     */
    getPathPointsFromNode(pathElement) {
        if (!pathElement || typeof pathElement.getAttribute !== 'function') {
            throw new Error("Input 'pathElement' must be a valid DOM element with a 'getAttribute' method.");
        }
        const dAttribute = pathElement.getAttribute('d');
        return this.getPathPoints(dAttribute);
    }

    /**
     * Extracts and transforms all unique [x, y] coordinates from an SVG path element,
     * considering its and its parents' 'translate' and 'scale' transformations.
     * @param {SVGPathElement} pathElement - The SVG path element.
     * @returns {Array<[number, number]>} An array of transformed [x, y] coordinate pairs.
     */
    getTranslatedAndScaledPathPoints(pathElement) {
        if (!pathElement || typeof pathElement.getAttribute !== 'function') {
            throw new Error("Input 'pathElement' must be a valid DOM element with a 'getAttribute' method.");
        }
        const dAttribute = pathElement.getAttribute('d');
        if (!dAttribute) return [];
        
        const localPoints = this.getPathPoints(dAttribute);

        const { totalTx, totalTy, totalSx, totalSy } = this._accumulateTranslateAndScale(pathElement);
        
        return localPoints.map(point => [
            (point[0] * totalSx) + totalTx,
            (point[1] * totalSy) + totalTy,
        ]);
    }

     /**
     * Extracts and transforms all unique [x, y] coordinates from an SVG path element,
     * correctly applying ALL parent transformations (scale, translate, rotate, skew, matrix)
     * by leveraging the browser's native SVG engine.
     * @param {SVGPathElement} pathElement - The SVG path element.
     * @returns {Array<[number, number]>|null} An array of transformed [x, y] coordinate pairs.
     */
    getTransformedPathPoints(pathElement) {
        if (!pathElement || typeof pathElement.getAttribute !== 'function') {
            throw new Error("Input 'pathElement' must be a valid DOM element with a 'getAttribute' method.");
        }
        
        const dAttribute = pathElement.getAttribute('d');
        if (!dAttribute) {
            return [];
        }
        
        const svg = pathElement.ownerSVGElement;
        if (!svg) {
            console.error("Path element must be inside an <svg> element to calculate transformations correctly.");
            return null;
        }

        // 1. Get the path's points in their own local coordinate system.
        const localPoints = this.getPathPoints(dAttribute);

        // 2. Get the total transformation matrix from the browser's SVG engine.
        // This single matrix represents the combined effect of ALL transforms
        // (translate, scale, rotate, skew) on the path and its parents, in the correct order.
        const matrix = pathElement.getTransformToElement(svg);

        // 3. Apply the matrix to every point to get its final position on the SVG canvas.
        return localPoints.map(point => {
            const domPoint = new DOMPoint(point[0], point[1]);
            const transformedPoint = domPoint.matrixTransform(matrix);
            return [transformedPoint.x, transformedPoint.y];
        });
    }

    /**
     * Returns the detailed segment breakdown for a path 'd' attribute.
     * @param {string} dAttribute - The SVG path 'd' attribute string.
     * @returns {Array<object>} An array of segment objects with parsed information.
     */
    getPathCommandandPoints(dAttribute) {
        return this._processPathData(dAttribute);
    }

    /**
     * Checks if a path described by a 'd' attribute is a curve with more than a certain number of points.
     * @param {string} dAttribute - The SVG path 'd' attribute string.
     * @param {number} [pointsLen=4] - The minimum number of points to be considered a curve.
     * @returns {boolean} True if the path is considered a curve.
     */
    isCurveOnly(dAttribute, pointsLen = 4) {
        if (!dAttribute) {
            return false;
        }
        const pointCount = this.getPathPoints(dAttribute).length;
        return pointCount > pointsLen;
    }

    /**
     * Breaks a composite SVG path element into an array of separate path elements for each subpath.
     * Note: This method may not correctly handle relative 'm' commands in all contexts.
     * @param {SVGPathElement} pathNode - The composite SVG path element.
     * @param {boolean} [copyAttributes=true] - Whether to copy attributes from the original path.
     * @returns {Array<SVGPathElement>} An array of new SVG path elements.
     * @private
     */
    _breakApartCompositePathToSubpaths(pathNode, copyAttributes = true) {
        const pathData = pathNode.getAttribute('d');
        if (!pathData) {
            return [];
        }

        // Regular expression to split the path data by "M" or "m" commands.
        // The lookahead `(?=[Mm])` ensures that the delimiter is included in the next split.
        const subpathDataArray = pathData.trim().split(/(?=[Mm])/);

        const subpaths = subpathDataArray.map(subpathData => {
            if (subpathData.trim() === '') {
                return null;
            }
            const newPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            newPath.setAttribute('d', subpathData.trim());

            // Copy attributes from the original path to the new subpaths
            if (copyAttributes) {
                for (let i = 0; i < pathNode.attributes.length; i++) {
                    const attr = pathNode.attributes[i];
                    if (attr.name !== 'd' && attr.name !== 'id') {
                        newPath.setAttribute(attr.name, attr.value);
                    }
                }
            }
            return newPath;
        }).filter(path => path !== null);

        return subpaths;
    }

    /**
     * Decomposes a composite SVG path into subpaths, correctly handling relative 'm' commands
     * by converting them to absolute 'M' commands.
     * @param {SVGPathElement} pathNode - The composite SVG path element.
     * @param {boolean} [copyAttributes=true] - Whether to copy attributes from the original path.
     * @param {number} [groupIndex] - An optional index to add as a 'data-group-index' attribute.
     * @returns {Array<SVGPathElement>} An array of new SVG path elements.
     * @private
     */
    _decomposeCompositePathToSubpaths(pathNode, copyAttributes = true, groupIndex) {
        const pathData = pathNode.getAttribute('d');
        if (!pathData) {
            return [];
        }

        const subpathDataArray = pathData.trim().split(/(?=[Mm])/);
        const subpaths = [];
        let lastX = 0;
        let lastY = 0;

        subpathDataArray.forEach(subpathData => {
            subpathData = subpathData.trim();
            if (subpathData === '') {
                return;
            }

            const commandMatch = /([Mm])\s*(-?[\d.]+)\s*,?\s*(-?[\d.]+)/.exec(subpathData);
            if (!commandMatch) {
                return;
            }

            const command = commandMatch[1];
            let currentX = parseFloat(commandMatch[2]);
            let currentY = parseFloat(commandMatch[3]);
            let transformedSubpathData = subpathData;

            if (command === 'm') {
                const absoluteX = lastX + currentX;
                const absoluteY = lastY + currentY;
                transformedSubpathData = 'M ' + absoluteX + ' ' + absoluteY + subpathData.substring(commandMatch[0].length);
            }

            const newPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            newPath.setAttribute('d', transformedSubpathData);

            if (groupIndex !== undefined) {
                newPath.setAttribute('data-group-index', groupIndex);
            }

            if (copyAttributes) {
                for (let i = 0; i < pathNode.attributes.length; i++) {
                    const attr = pathNode.attributes[i];
                    if (attr.name !== 'd' && attr.name !== 'id') {
                        newPath.setAttribute(attr.name, attr.value);
                    }
                }
            }
            subpaths.push(newPath);

            if (command === 'm') {
                lastX += currentX;
                lastY += currentY;
            } else {
                lastX = currentX;
                lastY = currentY;
            }
        });

        return subpaths;
    }

    /**
     * Decomposes a composite path element into a group (<g>) of individual subpath elements.
     * @param {SVGPathElement} pathNode - The composite SVG path element.
     * @param {number} groupIndex - An index to assign to the group and its subpaths.
     * @returns {SVGGElement | null} A new <g> element containing the subpaths, or null if input is invalid.
     */
    decomposeToSubpathsGroup(pathNode, groupIndex) {
        if (!pathNode || typeof pathNode.getAttribute !== 'function') {
            console.error("Invalid element passed to decomposeToSubpathsGroup. Expected an SVG element.", pathNode);
            return null;
        }

        const d = pathNode.getAttribute("d");
        if (!d) return null;

        // Create the group that will contain the subpaths.
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

        // Set the group index on the <g> container itself.
        g.setAttribute("data-group-index", groupIndex);

        // Copy original attributes (fill, stroke, class, etc.) to the new group.
        for (let i = 0; i < pathNode.attributes.length; i++) {
            const attr = pathNode.attributes[i];
            if (attr.name !== "d") { // Don't copy the path data.
                g.setAttribute(attr.name, attr.value);
            }
        }

        // Decompose the path, passing the groupIndex to be added to each subpath.
        const subpaths = this._decomposeCompositePathToSubpaths(pathNode, false, groupIndex);

        // Append each subpath to the group and add a subpath index for ordering.
        subpaths.forEach((subPath, index) => {
            subPath.setAttribute("data-subpath-index", index);
            g.appendChild(subPath);
        });

        // console.log(`Decomposed path into group ${groupIndex} with ${subpaths.length} subpaths.`);

        return g;
    }
}