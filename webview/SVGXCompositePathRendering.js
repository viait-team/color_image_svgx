// This code requires Paper.js to be loaded in your project.
// <script src="https://cdnjs.cloudflare.com/ajax/libs/paper.js/0.12.15/paper-full.min.js"></script>

//
// It is correcting the rendering the paths in this svg, 
// which was decomposed from the composite path.
// The mistake to be corrected is that the paths from the composite path 
// which is done in the previous processes, did not consider the winding rule
// when those paths were created due to the two-step design.
// So the default winding rule should be used to identify solid or hole for these paths
// and then apply mask to the solid paths to create holes
// while keeping all paths in the DOM for rendering correctly as before decomposition.
//
class SVGXCompositePathRendering {
    /**
     * @param {string} svgElementId The ID of the SVG element containing the subpaths to process.
     */
    constructor(svgElement) {
        this.svg = svgElement
        if (!this.svg) {
            throw new Error(`SVG element is not defined.`);
        }
        
        // Find all direct <path> children within the target SVG element.
        // Using ':scope > path' ensures we only get immediate children.
        this.subpaths = Array.from(this.svg.querySelectorAll('path'));
        
        // Setup a hidden Paper.js canvas. It is only needed once for all calculations.
        if (!paper.project) {
            const canvas = document.createElement('canvas');
            paper.setup(canvas);
        }
    }


    _findClosestVertex(testPoint, path) {
        if (!path || !path.segments || path.segments.length === 0) {
            return null; // Return null if the path is invalid or empty
        }

        let closestPoint = null;
        let minDistance = Infinity;

        // Iterate through every segment in the path
        for (const segment of path.segments) {
            const currentPoint = segment.point;
            const distance = testPoint.getDistance(currentPoint);

            // If this point is closer than any we've seen before, record it
            if (distance < minDistance) {
                minDistance = distance;
                closestPoint = currentPoint;
            }
        }

        return closestPoint;
    }

    correctSubPathRendering() {
        if (this.subpaths.length < 2) {
            console.log("Not enough paths to analyze.");
            return;
        }

        // --- Step 1: Geometric Analysis using Your Winding Rule Algorithm ---

        const paperPaths = this.subpaths.map((el, index) => {
            if (!el.id) {
                el.id = `subpath-${index}-${Math.random().toString(36).substr(2, 9)}`;
            }
            const paperPath = new paper.Path(el.getAttribute('d'));
            return {
                element: el,
                paper: paperPath,
                // THE BUG FIX IS HERE: Inverted the orientation logic to account for SVG's Y-down axis.
                orientation: paperPath.clockwise ? -1 : 1,
                bg_group_id: -1 
            };
        });

        const solids = [];
        const holes = [];
        const bg_paths = [];

        let bggroup_count = 0;
        // For each path, calculate the winding number of its location.
        paperPaths.forEach(testPath => {
            let locationWindingNumber = 0;
            const testPoint = testPath.paper.firstSegment.point;

            // Sum the orientations of all other paths that contain this one.
            paperPaths.forEach(outerPath => {
                if (testPath.element.id === outerPath.element.id) return; // Don't check against self.
              
                if (outerPath.paper.contains(testPoint)) {
                    locationWindingNumber += outerPath.orientation;
                }     
                else { 
                    // Check 2: If not, is it on the boundary?
                    if (testPath.bg_group_id === -1 && outerPath.bg_group_id === -1) {

                        if (solids.indexOf(testPath) != -1 || solids.indexOf(outerPath) != -1) {
                            const closestPoint = this._findClosestVertex(testPoint, outerPath.paper);
                            const isOnBoundary = testPoint.getDistance(closestPoint) < 10;

                            // If either condition is true, the outerPath is a container.
                            if (isOnBoundary) {
                                locationWindingNumber += outerPath.orientation;
                                
                                bg_paths.push(outerPath);
                                bg_paths.push(testPath);
                                
                                bggroup_count++;
                                testPath.bg_group_id = bggroup_count;
                                outerPath.bg_group_id = bggroup_count;
                            }
                        }

                    }
                } 

            });

            // The final winding number for the area inside testPath is its own
            // orientation plus the winding number of its location.
            const totalWindingNumber = testPath.orientation + locationWindingNumber;

            if (totalWindingNumber === 0) {
                // A total winding number of 0 means the area is unfilled. This is a hole.
                holes.push(testPath);
            } else {
                // Any non-zero winding number means the area is filled. This is a solid.
                solids.push(testPath);
            }
        });

        if (holes.length === 0) {
            console.log("No hole paths were identified by the winding rule analysis.");
            return;
        }

        // --- Step 2: Build Relationships and Create SVG Masks ---
       
        const holesWithContainers = [];
        holes.forEach(hole => {
            let smallestContainer = null;
            let smallestContainerArea = Infinity;

            solids.forEach(solid => {
                if (solid.paper.contains(hole.paper.firstSegment.point)) {
                    if (solid.paper.area < smallestContainerArea) {
                        smallestContainer = solid;
                        smallestContainerArea = solid.paper.area;
                    }
                }
            });

            if (smallestContainer) {
                holesWithContainers.push({
                    hole: hole,
                    container: smallestContainer
                });
            }
        });

        const holesByContainer = holesWithContainers.reduce((acc, current) => {
            const containerId = current.container.element.id;
            if (!acc[containerId]) {
                acc[containerId] = { container: current.container.element, holes: [] };
            }
            acc[containerId].holes.push(current.hole.element);
            return acc;
        }, {});
        
        let defs = this.svg.querySelector('defs');
        if (!defs) {
            defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            this.svg.prepend(defs);
        }

        Object.values(holesByContainer).forEach(group => {
            const containerElement = group.container;
            const holeElements = group.holes;
            const maskId = `mask-for-${containerElement.id}`;

            const mask = document.createElementNS('http://www.w3.org/2000/svg', 'mask');
            mask.setAttribute('id', maskId);
            mask.setAttribute('maskUnits', 'userSpaceOnUse');

            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', '0');
            rect.setAttribute('y', '0');
            rect.setAttribute('width', '100%');
            rect.setAttribute('height', '100%');
            rect.setAttribute('fill', 'white');
            mask.appendChild(rect);

            holeElements.forEach(holeEl => {
                const holeMaskPath = holeEl.cloneNode(true);
                holeMaskPath.style.fill = 'black'; 
                holeMaskPath.setAttribute('fill', 'black');
                mask.appendChild(holeMaskPath);
                
                holeEl.style.fill = 'none';
                holeEl.style.pointerEvents = 'all';

            });
            
            defs.appendChild(mask);

            const computedStyle = window.getComputedStyle(containerElement);
            const fillColor = computedStyle.getPropertyValue('fill');
            containerElement.style.fill = fillColor;

            containerElement.setAttribute('mask', `url(#${maskId})`);
         
        });


       
        // After all masking is done, find the single largest solid path.
       
        let backgroundPath = null;
        let maxArea = -1;

        solids.forEach(solid => {
            if (solid.paper.area > maxArea) {
                maxArea = solid.paper.area;
                backgroundPath = solid;
            }
        });

   
        // If we found a background path, set its fill to transparent as instructed.
        let bg_area = 0;
        if (backgroundPath) {
            console.log(`Identified '${backgroundPath.element.id}' as the background path. Setting to transparent.`);
            backgroundPath.element.style.fill = 'transparent';
            bg_area =  Math.abs(backgroundPath.paper.area);
        } else {
            console.log("Could not identify a background path to make transparent.");
        }
        
        
        bg_paths.forEach( bg_path => {
            const path_area = Math.abs(bg_path.paper.area);
            if (path_area >= 0.04 * bg_area) {
                console.log(`Identified '${bg_path.element.id}' as a background path. Setting to transparent.`);
                bg_path.element.style.opacity = '0.4';
            }
         
        });
    

    }

    /**
     * Finds subpaths by their 'data-group-index', combines them into single
     * composite paths, and replaces the original subpath groups within the SVG.
     * This method directly modifies the SVG DOM and does not return a value.
     */
    combineSubPathsToCompositePath() {
        const subpaths = this.svg.querySelectorAll('path[data-group-index]');
        if (subpaths.length === 0) {
            console.log("No subpaths with 'data-group-index' found to combine.");
            return;
        }

        // Group subpaths by their shared 'data-group-index'.
        const subpathsByGroup = new Map();
        subpaths.forEach(path => {
            const groupIndex = path.getAttribute('data-group-index');
            if (!subpathsByGroup.has(groupIndex)) {
                subpathsByGroup.set(groupIndex, []);
            }
            subpathsByGroup.get(groupIndex).push(path);
        });

        // Process each group of subpaths.
        for (const [groupIndex, pathGroup] of subpathsByGroup.entries()) {
            if (pathGroup.length === 0) continue;

            // The subpaths are assumed to be inside a <g> element that was created during decomposition.
            // This group element holds the original styles and is what we need to replace.
            const groupElement = pathGroup[0].parentElement;

            // Ensure the parent is a <g> element corresponding to the group index before proceeding.
            if (groupElement && groupElement.tagName.toLowerCase() === 'g' && groupElement.getAttribute('data-group-index') === groupIndex) {
                
                // Sort paths by their subpath index to ensure the 'd' attribute is in the correct order.
                pathGroup.sort((a, b) => {
                    const indexA = parseInt(a.getAttribute('data-subpath-index'), 10);
                    const indexB = parseInt(b.getAttribute('data-subpath-index'), 10);
                    return indexA - indexB;
                });

                // Combine the 'd' attributes from all subpaths into one string.
                const combinedD = pathGroup.map(p => p.getAttribute('d')).join(' ');

                // Create the new composite <path> element.
                const newCompositePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                newCompositePath.setAttribute('d', combinedD);

                // Copy all attributes from the original <g> element to the new composite <path>
                // to preserve fill, stroke, class, etc.
                for (const attr of groupElement.attributes) {
                    if (attr.name !== 'data-group-index') { // Don't copy the temporary group index.
                        newCompositePath.setAttribute(attr.name, attr.value);
                    }
                }

                // The main action: Replace the entire <g> (and all its subpath children)
                // with the single, new composite <path> in the SVG.
                groupElement.parentNode.replaceChild(newCompositePath, groupElement);
                
                // console.log(`Group ${groupIndex}: Replaced ${pathGroup.length} subpaths with a single composite path.`);

            } else {
                console.warn(`Could not combine group ${groupIndex}. The subpaths are not contained within a valid group (<g>) element.`);
            }
        }

    }

    removeEmptyElementG() {        
        let svgNode = this.svg;
        if (!svgNode) return "";
        if (!svgNode || !(svgNode instanceof SVGElement)) {
            console.warn("Invalid SVG node provided.");
            return;
        }

        const groups = svgNode.querySelectorAll('g');
        groups.forEach(group => {
            if (group.children.length === 0) {
            group.remove();
            }
        });
    }


  
}