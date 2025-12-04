// Geometric Simulation of SVG 'fill-rule: nonzero' Using Clipper2
// Based on nonzero-impl.md algorithm

class SVGXCompositePathRendering {
    constructor(svgElement) {
        this.svg = svgElement;
        if (!this.svg) {
            throw new Error(`SVG element is not defined.`);
        }

        this.subpaths = Array.from(this.svg.querySelectorAll('path'));

        if (!paper.project) {
            const canvas = document.createElement('canvas');
            paper.setup(canvas);
        }
    }

    pathDataToClipper(pathData) {
        const scale = 1000;
        const path = new paper.Path(pathData);
        path.flatten(0.5);
        const clipperPath = [];

        if (path.segments) {
            path.segments.forEach(segment => {
                clipperPath.push({
                    x: Math.round(segment.point.x * scale),
                    y: Math.round(segment.point.y * scale)
                });
            });
        }

        return [clipperPath];
    }

    clipperToPathData(clipperPaths) {
        const scale = 1000;
        let d = "";

        if (!clipperPaths || clipperPaths.length === 0) return "";

        clipperPaths.forEach(path => {
            if (path.length === 0) return;

            d += `M ${path[0].x / scale} ${path[0].y / scale} `;
            for (let i = 1; i < path.length; i++) {
                d += `L ${path[i].x / scale} ${path[i].y / scale} `;
            }
            d += "Z ";
        });

        return d.trim();
    }

    // Step 2: Calculate orientation using shoelace formula
    computeOrientation(paperPath) {
        // Paper.js provides clockwise property
        // clockwise = true means negative area (clockwise in screen coords)
        // clockwise = false means positive area (counterclockwise in screen coords)
        return paperPath.clockwise ? -1 : 1;
    }

    // Step 3: Classify subpaths as outers or holes using winding numbers
    classifySubpaths(subpaths) {
        subpaths.forEach((testPath, idx) => {
            // Pick a point inside the subpath (use first segment point)
            const testPoint = testPath.paper.firstSegment.point;

            // Compute winding number with respect to all OTHER subpaths
            let windingNumber = 0;
            subpaths.forEach(otherPath => {
                if (testPath === otherPath) return;

                // If the test point is contained by the other path,
                // add the other path's orientation to the winding number
                if (otherPath.paper.contains(testPoint)) {
                    windingNumber += otherPath.orientation;
                }
            });

            // Classify: if winding number is 0, it's an outer; otherwise it's a hole
            testPath.isOuter = (windingNumber === 0);
            testPath.windingNumber = windingNumber;

            console.log(`  Subpath ${idx}: orientation=${testPath.orientation}, winding=${windingNumber}, isOuter=${testPath.isOuter}`);
        });
    }

    // Step 4: Find holes within a given outer subpath
    findHolesWithin(outer, allSubpaths) {
        const holes = [];

        allSubpaths.forEach(candidate => {
            if (candidate === outer) return;
            if (candidate.isOuter) return; // Only consider non-outers as potential holes

            // Check if the hole is contained within the outer
            const holePoint = candidate.paper.firstSegment.point;
            if (outer.paper.contains(holePoint)) {
                holes.push(candidate);
            }
        });

        return holes;
    }

    pathClipperAlgorithm() {
        // Group subpaths by data-group-index
        const subpathsByGroup = new Map();
        this.subpaths.forEach(path => {
            const groupIndex = path.getAttribute('data-group-index') || 'default';
            if (!subpathsByGroup.has(groupIndex)) {
                subpathsByGroup.set(groupIndex, []);
            }
            subpathsByGroup.get(groupIndex).push(path);
        });

        console.log(`Starting Clipper2 nonzero simulation for ${subpathsByGroup.size} groups.`);

        const { Clipper, FillRule } = window.Clipper2;

        let processedCount = 0;
        let skippedCount = 0;

        subpathsByGroup.forEach((groupPaths, groupIndex) => {
            if (groupPaths.length === 1) {
                console.log(`Group [${groupIndex}]: 1 subpath (skipped - no holes possible)`);
                skippedCount++;
                return;
            }

            processedCount++;
            console.log(`\nGroup [${groupIndex}]: ${groupPaths.length} subpaths`);

            // Step 1: Parse and create subpath objects
            let subpaths = groupPaths.map((el, index) => {
                const paperPath = new paper.Path(el.getAttribute('d'));
                return {
                    element: el,
                    paper: paperPath,
                    orientation: this.computeOrientation(paperPath),
                    d: el.getAttribute('d'),
                    index: index
                };
            });

            // Step 2 & 3: Classify subpaths as outers or holes
            this.classifySubpaths(subpaths);

            // Step 4 & 5: For each outer, process its holes using DIFFERENCE
            const solutionPaths = [];
            const outers = subpaths.filter(sp => sp.isOuter);

            console.log(`  Found ${outers.length} outer(s)`);

            outers.forEach((outer, outerIdx) => {
                console.log(`  Processing outer ${outerIdx}...`);

                // Find all holes within this outer
                const holes = this.findHolesWithin(outer, subpaths);
                console.log(`    Found ${holes.length} hole(s) within this outer`);

                // Start with the outer path
                let resultClipper = this.pathDataToClipper(outer.d);

                // Subtract each hole using DIFFERENCE
                holes.forEach((hole, holeIdx) => {
                    console.log(`    Subtracting hole ${holeIdx}...`);
                    const holeClipper = this.pathDataToClipper(hole.d);

                    try {
                        resultClipper = Clipper.difference(resultClipper, holeClipper, FillRule.NonZero);
                    } catch (e) {
                        console.warn(`    DIFFERENCE failed for hole ${holeIdx}:`, e);
                    }
                });

                // Convert back to path data and store
                const resultPathData = this.clipperToPathData(resultClipper);
                solutionPaths.push({
                    d: resultPathData,
                    originalElement: outer.element
                });
            });

            console.log(`  Generated ${solutionPaths.length} solution path(s)`);

            // Step 6: Update DOM - replace original paths with solution paths
            if (groupPaths.length > 0) {
                const parent = groupPaths[0].parentNode;
                if (!parent) {
                    console.warn(`No parent node for group ${groupIndex}, skipping DOM update`);
                    return;
                }

                const lastPath = groupPaths[groupPaths.length - 1];
                const referenceNode = lastPath.nextSibling;

                // Store attributes from first path (excluding 'd' and 'id')
                const templateAttrs = [];
                for (const attr of groupPaths[0].attributes) {
                    if (attr.name !== 'd' && attr.name !== 'id') {
                        templateAttrs.push({ name: attr.name, value: attr.value });
                    }
                }

                // Remove all original paths
                groupPaths.forEach(p => p.remove());

                // Insert solution paths
                solutionPaths.forEach((sol, idx) => {
                    const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    pathEl.setAttribute('d', sol.d);

                    // Copy attributes
                    templateAttrs.forEach(attr => {
                        pathEl.setAttribute(attr.name, attr.value);
                    });
                    pathEl.setAttribute('data-group-index', groupIndex);
                    pathEl.setAttribute('id', `${groupIndex}-result-${idx}`);

                    // Insert into DOM
                    if (referenceNode && referenceNode.parentNode === parent) {
                        parent.insertBefore(pathEl, referenceNode);
                    } else {
                        parent.appendChild(pathEl);
                    }
                });
            }
        });

        console.log(`\nClipper2 nonzero simulation complete. Processed: ${processedCount}, Skipped: ${skippedCount}`);
    }
}
