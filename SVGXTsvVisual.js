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
    console.log(`Initialized with ${this.words.length} valid OCR words, using scale factor: ${this.scale}.`);

    this.wordToPaths = new Map();
    this.pathToWord = new Map();
  }

  /**
   * Parses the raw TSV string into an array of structured word objects,
   * applying the scale factor to all coordinate data.
   * @private
   */
  _parseTsv(tsvContent) {
    const lines = tsvContent.trim().split('\n');
    if (lines.length < 2) return [];
    
    // Create a dynamic header map from the first line.
    const headers = lines[0].split('\t');
    
    return lines.slice(1).map(line => {
      const values = line.split('\t');
      const wordObj = headers.reduce((obj, header, i) => {
        obj[header] = values[i];
        return obj;
      }, {});

      // Apply the scale factor to the coordinate data during parsing.
      // This ensures all subsequent logic uses the correct coordinate space.
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
   * Finds all SVG paths that fall within the bounding box of a scaled OCR'd word.
   */
  matchPathsToWords() {
    console.log("Matching paths to scaled OCR words...");
    this.wordToPaths.clear();
    this.pathToWord.clear();

    const paths = this.svgNode.querySelectorAll("path");

    paths.forEach(path => {
      const startPoint = this._getPathStartPoint(path);
      if (!startPoint) return;

      for (const word of this.words) {
        // The word coordinates are now already in the correct scaled SVG space.
        const w_left = word.left;
        const w_top = word.top;
        const w_right = w_left + word.width;
        const w_bottom = w_top + word.height;

        if (startPoint.x >= w_left && startPoint.x <= w_right &&
            startPoint.y >= w_top && startPoint.y <= w_bottom) {
          const key = `${word.left}-${word.top}`;
          if (!this.wordToPaths.has(key)) {
            this.wordToPaths.set(key, []);
          }
          this.wordToPaths.get(key).push({ path: path, anchor: startPoint });
          this.pathToWord.set(path, word);
          break;
        }
      }
    });

    console.log(`Matching complete. Found ${this.pathToWord.size} paths corresponding to words.`);
  }

  /**
   * Remove all matched SVG <path> elements
   */
  removeMatchedPaths() {
    if (this.pathToWord.size === 0) {
      console.log("No paths have been matched. Run matchPathsToWords() first.");
      return;
    }

    console.log("Removing matched paths...");
    this.wordToPaths.forEach((matchedPaths) => {
      if (matchedPaths.length === 0) return;

      const word = this.pathToWord.get(matchedPaths[0].path);

      matchedPaths.forEach(entry => {
        if (entry.path && entry.path.parentNode) {
            entry.path.parentNode.removeChild(entry.path);
        }
      });
      
      console.log(`Removed ${matchedPaths.length} paths for text: '${word.text}'`);

    });
  }

  // Coloring
  applyColorToSvg() {
    // Original functionality: Apply RGB coloring to SVG
    const groups = this.svgNode.querySelectorAll('g');
    
    if(groups.length === 3) {
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
  insertText(jsonContent) {
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

    ocrGroups.forEach(group => {
      // The coordinates from the JSON are expected to be already scaled.
      const left = group.left;
      const top = group.top;
      const height = group.height;
      const text = group.text;

      if (!text || !text.trim()) return;

      const fontSize = Math.round(height * 1.1);
      const x = left;
      const y = top + height; // Use the bottom of the bounding box as the baseline

      const textElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
      textElement.setAttribute("x", x.toFixed(2));
      textElement.setAttribute("y", y.toFixed(2));
      textElement.setAttribute("font-size", `${fontSize}px`);
      textElement.setAttribute("fill", "#666666");
      textElement.setAttribute("font-family", "Arial, sans-serif");
      textElement.textContent = text;
  
      textGroup.appendChild(textElement);
    });

    console.log("Text insertion complete.");
  }

}