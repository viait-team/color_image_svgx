/**
 * A class to visualize and manipulate an SVG document based on Tesseract TSV data.
 * It correctly handles nested <g> transforms and scales OCR coordinates to match the SVG's viewBox.
 */
class SVGXTsvVisual {
  /**
   * Initializes the visualizer with TSV data, the root <svg> element, and a scale factor.
   * @param {string} tsvContent The raw string content of the TSV file from Tesseract.
   * @param {Element} svgNode The root <svg> DOM element.
   * @param {number} scale The scale factor to convert OCR pixel coordinates to SVG user units.
   */
  constructor(tsvContent, svgNode, scale = 1.0) {
    if (!tsvContent || !svgNode || svgNode.tagName.toLowerCase() !== 'svg') {
      throw new Error("TSV content and a valid root SVG element are required.");
    }
    if (typeof scale !== 'number' || isNaN(scale)) {
      throw new Error("A valid number for the scale factor must be provided.");
    }
    this.svgNode = svgNode;
    this.scale = scale;
    this.words = this._parseTsv(tsvContent);
    this.matchedPaths = new Set(); // Use a single Set to store paths to be removed.
    this.svgxPathPoints = new SVGXPathPointsX();
    this.svgxPathSegments = new SVGXPathSegments();

    console.log(`Initialized with ${this.words.length} valid OCR words, using scale factor: ${this.scale}.`);
  }

  /**
   * Parses the raw TSV string into an array of structured word objects,
   * applying the scale factor to all coordinate data.
   * @private
   */
  _parseTsv(tsvContent) {
    const lines = tsvContent.trim().split(/\r?\n/);
    if (lines.length < 2) return [];

    const headers = lines[0].split('\t');

    return lines.slice(1).map(line => {
      const values = line.split('\t');
      const wordObj = headers.reduce((obj, header, i) => {
        obj[header] = values[i];
        return obj;
      }, {});

      wordObj.left = Math.round(parseInt(wordObj.left, 10) * this.scale);
      wordObj.top = Math.round(parseInt(wordObj.top, 10) * this.scale);
      wordObj.width = Math.round(parseInt(wordObj.width, 10) * this.scale);
      wordObj.height = Math.round(parseInt(wordObj.height, 10) * this.scale);
      wordObj.conf = parseFloat(wordObj.conf);

      return wordObj;
    }).filter(w => w.conf && w.conf > 60 && w.text && w.text.trim() !== "");
  }

  /**
   * Calculates the cumulative transformation for a given SVG element by walking up the DOM tree.
   * This is the corrected version of the algorithm.
   * @param {Element} element The SVG element to calculate the transform for.
   * @returns {{scaleX: number, scaleY: number, translateX: number, translateY: number}} The cumulative transform.
   * @private
   */
  _getCumulativeTransform(element) {
    let cumulativeTransform = { scaleX: 1.0, scaleY: 1.0, translateX: 0.0, translateY: 0.0 };
    let current = element;

    const ancestors = [];
    while (current && current.tagName.toLowerCase() !== 'svg') {
      ancestors.push(current);
      current = current.parentElement;
    }

    for (let i = ancestors.length - 1; i >= 0; i--) {
      const node = ancestors[i];
      const transformAttr = node.getAttribute('transform');
      if (transformAttr) {
        const translateMatch = transformAttr.match(/translate\(\s*([\d\.-]+)\s*,?\s*([\d\.-]+)?\s*\)/);
        if (translateMatch) {
          const tx = parseFloat(translateMatch[1]);
          const ty = translateMatch[2] !== undefined ? parseFloat(translateMatch[2]) : 0;
          cumulativeTransform.translateX += tx * cumulativeTransform.scaleX;
          cumulativeTransform.translateY += ty * cumulativeTransform.scaleY;
        }

        const scaleMatch = transformAttr.match(/scale\(\s*([\d\.-]+)\s*,?\s*([\d\.-]+)?\s*\)/);
        if (scaleMatch) {
          const sx = parseFloat(scaleMatch[1]);
          const sy = scaleMatch[2] !== undefined ? parseFloat(scaleMatch[2]) : sx;
          cumulativeTransform.scaleX *= sx;
          cumulativeTransform.scaleY *= sy;
        }
      }
    }

    return cumulativeTransform;
  }

  /**
   * Calculates the bounding box of a path element by parsing all its points.
   * Requires this.svgxPathPoints to be an instance of SVGXPathPoints.
   * @param {Element} pathElement The SVG <path> element.
   * @returns {{x: number, y: number, width: number, height: number}|null} The absolute bounding box.
   * @private
   */
  /*
  _getPathBoundingBox(pathElement) {
    // getTransformedPathPoints returns an array of [x, y] pairs.
    const points = this.svgxPathPoints.getTransformedPathPoints(pathElement);
    if (!points || points.length === 0) {
        return null;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (const point of points) {
        const [x, y] = point;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    if (!isFinite(minX)) {
        return null; // No valid points found
    }

    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY
    };
  }
  */

  /**
   * Calculates the absolute start coordinates of a <path> element.
   * @param {Element} pathElement The SVG <path> element.
   * @returns {{x: number, y: number, transform: Object}|null} The absolute coordinates and the transform used.
   * @private
   */
  _getPathStartPoint(pathElement) {
    const d = pathElement.getAttribute('d');
    if (!d) return null;

    const moveMatch = d.match(/^[Mm]\s*([\d\.-]+)[\s,]+([\d\.-]+)/);
    if (moveMatch) {
      const transform = this._getCumulativeTransform(pathElement);
      const x0 = parseFloat(moveMatch[1]);
      const y0 = parseFloat(moveMatch[2]);
      const absX = (x0 * transform.scaleX) + transform.translateX;
      const absY = (y0 * transform.scaleY) + transform.translateY;
      return { x: absX, y: absY, transform: transform };
    }
    return null;
  }

  /**
   * Finds all SVG paths that fall within the bounding box of any OCR'd word.
   */
  matchPathsToWords() {
    console.log("Matching paths to scaled OCR words...");
    this.matchedPaths.clear();

    const paths = this.svgNode.querySelectorAll("path");

    paths.forEach(path => {

      const startPoint = this._getPathStartPoint(path);
      if (!startPoint) return;

      // The reason it failed is that **`path.getBBox()` is unreliable.** 
      // IT SHOULD NOT BE USED HERE.
      // It does not include transformations from parent elements (like `<g transform="scale(...)">`)
      // const bbox = path.getBBox();

      const bbox = path.getBoundingClientRect(); // Use this instead of getBBox()
      const svgRect = this.svgNode.getBoundingClientRect();

      // Convert bbox from screen pixels to SVG user units
      const pathWidth = bbox.width * (this.svgNode.viewBox.baseVal.width / svgRect.width);
      const pathHeight = bbox.height * (this.svgNode.viewBox.baseVal.height / svgRect.height);

      // This should work, but it is very slow for complex paths.
      // It is kept here for reference. here bbox width and height is the same as above, pathWidth and pathHeight. 
      // const bbox = this._getPathBoundingBox(path);

      // the most important part:
      // OCR coordinates are already scaled to SVG user units in this.words (see constructor).
      // The path coordinates are now already in the correct scaled SVG space.
      // We just need to check if the start point of the path is inside any word bounding box.
      // If it is, we consider it a match.

      const threshold = 2; // Tolerance in user units
      for (const word of this.words) {
        // The word coordinates are now already in the correct scaled SVG space.
        const w_left = word.left;
        const w_top = word.top;
        const w_right = w_left + word.width;
        const w_bottom = w_top + word.height;

        const isStartInside = startPoint.x >= (w_left - threshold) && startPoint.x <= (w_right + threshold) &&
          startPoint.y >= (w_top - threshold) && startPoint.y <= (w_bottom + threshold);

        if (isStartInside) {
          // if (bbox.width < (word.width + 2*threshold)) {
          if ((pathWidth * pathHeight) <= (word.width * word.height)) {
            this.matchedPaths.add(path);
            break;
          }
          // }
        }

      }
    });

    console.log(`Matching complete. Found ${this.matchedPaths.size} paths corresponding to words.`);
  }

  /**
   * Finds all SVG paths that represent OCR'd words, while excluding larger container graphics.
   */
  matchPathsToWordsZZZ() {
    console.log("Matching paths to scaled OCR words...");
    this.matchedPaths.clear();

    const paths = this.svgNode.querySelectorAll("path");

    // --- PERFORMANCE IMPROVEMENT: Calculate these constant values only once. ---
    const svgRect = this.svgNode.getBoundingClientRect();
    const svgViewBox = this.svgNode.viewBox.baseVal;
    const scaleX = svgViewBox.width / svgRect.width;
    const scaleY = svgViewBox.height / svgRect.height;

    paths.forEach(path => {
      const startPoint = this._getPathStartPoint(path);
      if (!startPoint) return;

      const threshold = 2; // Tolerance for the start point check, in SVG user units

      for (const word of this.words) {
        // All word coordinates are pre-scaled to the SVG coordinate system.
        const w_left = word.left;
        const w_top = word.top;
        const w_right = w_left + word.width;
        const w_bottom = w_top + word.height;

        const isStartInside = startPoint.x >= (w_left - threshold) && startPoint.x <= (w_right + threshold) &&
          startPoint.y >= (w_top - threshold) && startPoint.y <= (w_bottom + threshold);

        if (isStartInside) {
          // This path is a candidate. Now, confirm it's the text itself, not a larger container.

          // 1. Get the path's final rendered size in screen pixels using the reliable method.
          const pathScreenBbox = path.getBoundingClientRect();

          // 2. Convert pixel dimensions to SVG user units for a correct comparison.
          const pathWidth = pathScreenBbox.width * scaleX;
          const pathHeight = pathScreenBbox.height * scaleY;

          const pathArea = pathWidth * pathHeight;
          const wordArea = word.width * word.height;

          // --- CRITICAL FIX: Add a tolerance to the area comparison. ---
          // If the path's area is reasonably close to the word's area, it's a match.
          // This prevents incorrectly rejecting paths that are slightly larger due to tracing artifacts.
          if (pathArea <= (wordArea * 1.1)) {
            this.matchedPaths.add(path);
            break;
          }
        }
      }
    });

    console.log(`Matching complete. Found ${this.matchedPaths.size} paths corresponding to words.`);
  }

  /**
   * Performs a second-pass matching using a JSON string of OCR line groups.
   * This is called AFTER matchPathsToWords() to find paths missed by the first pass.
   *
   * @param {string} jsonOCRGroupsString - A JSON string representing an array of OCR group objects.
   *   Each object is expected to have `left`, `top`, `width`, and `height` properties.
   */
  finalMatchPathsToWords(jsonOCRGroupsString) {
    console.log("Performing final group-based matching for missed paths...");

    let ocrGroups;
    try {
      ocrGroups = JSON.parse(jsonOCRGroupsString);
      if (!Array.isArray(ocrGroups)) {
        throw new Error("JSON content must parse to an array of group objects.");
      }
    } catch (e) {
      console.error("Failed to parse JSON for final matching:", e);
      return;
    }

    if (ocrGroups.length === 0) {
      console.log("No OCR groups provided for final matching.");
      return;
    }

    const paths = this.svgNode.querySelectorAll("path");
    const unmatchedPaths = Array.from(paths).filter(path => !this.matchedPaths.has(path));

    if (unmatchedPaths.length === 0) {
      console.log("No unmatched paths to check.");
      return;
    }

    // --- Calculate scaling factors once for efficiency, same as in the primary function. ---
    const svgRect = this.svgNode.getBoundingClientRect();
    const svgViewBox = this.svgNode.viewBox.baseVal;
    const scaleX = svgViewBox.width / svgRect.width;
    const scaleY = svgViewBox.height / svgRect.height;

    let newlyMatchedCount = 0;

    unmatchedPaths.forEach(path => {
      const startPoint = this._getPathStartPoint(path);
      if (!startPoint) return;

      const threshold = 2;

      for (const group of ocrGroups) {
        if (group.left === undefined || group.top === undefined || group.width === undefined || group.height === undefined) continue;

        const g_left = group.left;
        const g_top = group.top;
        const g_right = g_left + group.width;
        const g_bottom = g_top + group.height;

        const isStartInside = startPoint.x >= (g_left - threshold) && startPoint.x <= (g_right + threshold) &&
          startPoint.y >= (g_top - threshold) && startPoint.y <= (g_bottom + threshold);

        if (isStartInside) {
          // --- BOUNDING BOX CHECK ADDED ---
          // This sanity check prevents a large graphic surrounding an entire text line
          // from being incorrectly matched.

          // 1. Get the path's final rendered size and convert to SVG units.
          const pathScreenBbox = path.getBoundingClientRect();
          const pathWidth = pathScreenBbox.width * scaleX;
          const pathHeight = pathScreenBbox.height * scaleY;
          const pathArea = pathWidth * pathHeight;

          // 2. Get the area of the entire OCR text line group.
          const groupArea = group.width * group.height;

          // 3. Compare the areas. A valid path (even merged words) will have an area
          //    smaller than or equal to the entire line's area.
          if (pathArea <= (groupArea * 1.1)) {
            this.matchedPaths.add(path);
            newlyMatchedCount++;
            break;
          }
        }
      }
    });

    console.log(`Final matching complete. Found ${newlyMatchedCount} additional path(s). Total matched: ${this.matchedPaths.size}`);
  }

  /**
   * Remove all matched SVG <path> elements from the SVG document.
   */
  removeMatchedPaths() {
    if (this.matchedPaths.size === 0) {
      console.log("No paths have been matched. Run matchPathsToWords() first or there is nothing to remove.");
      return;
    }

    let parentNode = null;
    console.log(`Removing ${this.matchedPaths.size} matched paths...`);
    this.matchedPaths.forEach(path => {
      if (path && path.parentNode) {

        if (parentNode === null && path.parentNode.nodeName === 'g') {
          parentNode = path.parentNode;
        }

        path.parentNode.removeChild(path);
      }
    });

    // Clear the set after removal
    this.matchedPaths.clear();

    return parentNode;
  }

  //
  // remove segments of paths that match words
  //
  removeSegmentsByMatchingWords() {
    console.log("Analyzing remaining paths for removable text segments...");

    const remainingPaths = this.svgNode.querySelectorAll("path");
    if (remainingPaths.length === 0) {
      return;
    }

    let modifiedPathCount = 0;

    remainingPaths.forEach(path => {
      // 1. Use your trusted, self-contained function to get the transformation.
      // This is the correct and stable method to use after DOM modification.
      const transform = this._getCumulativeTransform(path);
      if (!transform) return;

      const dAttribute = path.getAttribute('d');
      const pathSegments = this.svgxPathSegments.processPathData(dAttribute);
      if (!pathSegments || pathSegments.length < 1) {
        return;
      }

      const segmentsToKeep = [];
      let foundTextSegments = false;
      const threshold = 2;


      pathSegments.forEach(segment => {

        if (segment.segmentType === 'line') {
          segmentsToKeep.push(segment);
          return; // Done with this segment, move to the next.
        }

        let isTextSegment = false;

        const startPoint = segment.startPoint
        // 2. Manually apply the calculated transform to the segment's points.
        const transformedStartPoint = {
          x: (startPoint.x * transform.scaleX) + transform.translateX,
          y: (startPoint.y * transform.scaleY) + transform.translateY
        };

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        segment.points.forEach(p => {
          const transformedX = (p.x * transform.scaleX) + transform.translateX;
          const transformedY = (p.y * transform.scaleY) + transform.translateY;
          minX = Math.min(minX, transformedX);
          maxX = Math.max(maxX, transformedX);
          minY = Math.min(minY, transformedY);
          maxY = Math.max(maxY, transformedY);
        });

        if (!isFinite(minX)) {
          segmentsToKeep.push(segment);
          return;
        }

        const areaThreshold = 8; // Minimum area to consider for removal
        const segmentArea = (maxX - minX) * (maxY - minY);
        const segmentWidth = (maxX - minX);
        const segmentHeight = (maxY - minY);

        // 3. Perform the comparison using the correctly transformed points.
        for (const word of this.words) {
          const isStartInside = transformedStartPoint.x >= (word.left - threshold) &&
            transformedStartPoint.x <= (word.left + word.width + threshold) &&
            transformedStartPoint.y >= (word.top - threshold) &&
            transformedStartPoint.y <= (word.top + word.height + threshold);

          if (isStartInside) {

            if (segmentArea > 0 && segmentWidth < areaThreshold && segmentHeight < areaThreshold) {
              modifiedPathCount++;
              isTextSegment = true;
              foundTextSegments = true;
              break;
            }
          }
        }

        if (!isTextSegment) {
          segmentsToKeep.push(segment);
        }
      });

      // 4. If text segments were found, reconstruct d string
      if (foundTextSegments) {

        // 1. Get the array of new, simple d strings.
        const newDStrings = this.svgxPathSegments.getFinalPathDataArray(segmentsToKeep);

        if (newDStrings.length > 0) {
          // 2. Update the first path with the first new d string.
          //    Also, copy all styling attributes from the original path.
          path.setAttribute('d', newDStrings[0]);

          // 3. For any subsequent d strings, create NEW <path> elements.
          for (let i = 1; i < newDStrings.length; i++) {
            const newPath = document.createElementNS("http://www.w3.org/2000/svg", "path");

            // 4. Copy all attributes from the original path (fill, stroke, transform, etc.)
            for (const attr of path.attributes) {
              if (attr.name !== 'id') {
                newPath.setAttribute(attr.name, attr.value);
              }
            }

            // 5. Set the new d attribute.
            newPath.setAttribute('d', newDStrings[i]);

            // 6. Insert the new path right after the original one.
            path.parentNode.insertBefore(newPath, path.nextSibling);
          }
        }
      }
    });

    console.log(`Segment removal complete. Modified or removed ${modifiedPathCount} path(s).`);
  }


  replaceSegmentsByMatchingWords() {
    console.log("Analyzing remaining paths for removable text segments...");

    const remainingPaths = this.svgNode.querySelectorAll("path");
    if (remainingPaths.length === 0) {
      return;
    }

    let modifiedPathCount = 0;

    remainingPaths.forEach(path => {
      // 1. Use your trusted, self-contained function to get the transformation.
      // This is the correct and stable method to use after DOM modification.
      const transform = this._getCumulativeTransform(path);
      if (!transform) return;

      const dAttribute = path.getAttribute('d');
      const pathSegments = this.svgxPathSegments.processPathData(dAttribute);
      if (!pathSegments || pathSegments.length < 1) {
        return;
      }

      let foundTextSegments = false;
      const threshold = 2;


      pathSegments.forEach(segment => {

        if (segment.segmentType === 'line') {
          return; // Done with this segment, move to the next.
        }

        let isTextSegment = false;

        const startPoint = segment.startPoint
        // 2. Manually apply the calculated transform to the segment's points.
        const transformedStartPoint = {
          x: (startPoint.x * transform.scaleX) + transform.translateX,
          y: (startPoint.y * transform.scaleY) + transform.translateY
        };

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        segment.points.forEach(p => {
          const transformedX = (p.x * transform.scaleX) + transform.translateX;
          const transformedY = (p.y * transform.scaleY) + transform.translateY;
          minX = Math.min(minX, transformedX);
          maxX = Math.max(maxX, transformedX);
          minY = Math.min(minY, transformedY);
          maxY = Math.max(maxY, transformedY);
        });

        if (!isFinite(minX)) {
          return;
        }

        const areaThreshold = 8; // Minimum area to consider for removal
        const segmentArea = (maxX - minX) * (maxY - minY);
        const segmentWidth = (maxX - minX);
        const segmentHeight = (maxY - minY);

        // 3. Perform the comparison using the correctly transformed points.
        for (const word of this.words) {
          const isStartInside = transformedStartPoint.x >= (word.left - threshold) &&
            transformedStartPoint.x <= (word.left + word.width + threshold) &&
            transformedStartPoint.y >= (word.top - threshold) &&
            transformedStartPoint.y <= (word.top + word.height + threshold);

          if (isStartInside) {

            if (segmentArea > 0 && segmentWidth < word.width && segmentHeight < word.height) {
              modifiedPathCount++;
              foundTextSegments = true;
              segment.matched = true;
              break;
            }
          }
        }

      });

      // 4. If text segments were found, reconstruct d string
      if (foundTextSegments) {

        const newDStrings = this.svgxPathSegments.getFinalPathData(pathSegments);
        path.setAttribute('d', newDStrings);

      }
    });

    console.log(`Segment replace  complete. Modified or removed ${modifiedPathCount} path(s).`);
  }

  //
  // Coloring for RGB channels traces
  //
  applyColorToSvg() {
    // Original functionality: Apply RGB coloring to SVG
    const groups = this.svgNode.querySelectorAll('g');

    if (groups.length === 3) {
      groups.forEach((g, i) => {
        const colors = ['red', 'green', 'gray'];
        g.setAttribute('fill', colors[i % 3]);
      });

      console.log('[LOG] Applied RGB coloring');
    }

  }

  /**
  * Parses JSON content and inserts it into the SVG as <text> elements.
  * Note: This function assumes the JSON content ALSO has scaled coordinates.
  *
  * @param {string} jsonContent The JSON string of structured OCR group data.
  */
  insertText(jsonContent, replacedParentNode = null) {
    let ocrGroups;
    try {
      ocrGroups = JSON.parse(jsonContent);
      if (!Array.isArray(ocrGroups)) {
        throw new Error("JSON content must be an array of group objects.");
      }
    } catch (e) {
      console.error("Failed to parse JSON content:", e);
      return;
    }

    if (ocrGroups.length === 0) {
      console.log("No text groups found in the JSON to insert.");
      return;
    }

    console.log(`Inserting ${ocrGroups.length} text groups from JSON...`);

    const textGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    textGroup.setAttribute("id", "ocr_text_layer");
    this.svgNode.appendChild(textGroup);

    if (replacedParentNode != null) {

      for (let i = 0; i < replacedParentNode.attributes.length; i++) {
        const attr = replacedParentNode.attributes[i];
        if (attr.name !== "d" && attr.name !== "id" && attr.name !== "transform") {
          textGroup.setAttribute(attr.name, attr.value);
        }
      }

      if (textGroup.hasAttribute("style")) {
        let style = textGroup.getAttribute("style");

        // Replace 'fill' with 'stroke' in the style string
        let updatedStyle = style.replace(/fill\s*:\s*([^;]+);?/i, 'stroke: $1;');

        textGroup.setAttribute("style", updatedStyle);
      }
    }

    ocrGroups.forEach(group => {
      // The coordinates from the JSON are expected to be already scaled.
      const left = group.left;
      const top = group.top;
      const height = group.height;
      const width = group.width;
      const text = group.text;

      if (!text || !text.trim()) return;

      if (height < 2.0 * width) {

        const fontSize = Math.round(height * 1.0);
        const x = left;
        const y = top + height * 2.0 / 3.0; // Use the bottom of the bounding box as the baseline

        const textElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
        textElement.setAttribute("x", x.toFixed(2));
        textElement.setAttribute("y", y.toFixed(2));
        textElement.setAttribute("font-size", `${fontSize}px`);
        textElement.setAttribute("fill", "#666666");
        textElement.setAttribute("font-family", "Arial, sans-serif");
        textElement.textContent = text;

        textGroup.appendChild(textElement);

      } else {
        // append text vertically
        const fontSize = Math.round(width * 1.2);
        const x = left; // The x-coordinate for the start of the text
        const y = top + height * 2.0 / 3.0;  // The y-coordinate for the start of the text

        const textElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
        textElement.setAttribute("x", x.toFixed(2));
        textElement.setAttribute("y", y.toFixed(2));
        textElement.setAttribute("font-size", `${fontSize}px`);
        textElement.setAttribute("fill", "#666666");
        textElement.setAttribute("font-family", "Arial, sans-serif");

        // Apply a transform to rotate the text 90 degrees clockwise around its starting point.
        // We adjust the x position by the width to align it to the right edge of the bounding box.
        textElement.setAttribute("transform", `rotate(270, ${x + width}, ${y})`);
        textElement.textContent = text;

        textGroup.appendChild(textElement);

      }

    });

    console.log("Text insertion complete.");
  }


}