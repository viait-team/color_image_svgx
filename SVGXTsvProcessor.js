/**
 * A JavaScript class to process Tesseract TSV data from a string.
 * It takes TSV content, a scale factor, intelligently groups adjacent words, 
 * and returns the result with scaled coordinates as a JSON string.
 * This version operates purely on strings and has no Node.js file system dependencies.
 */
class SVGXTsvProcessor {

    /**
     * Initializes the processor with the TSV file content and a coordinate scale factor.
     * @param {string} fileContent - A string containing the full Tesseract TSV data.
     * @param {number} scale - The scale factor to convert OCR pixel coordinates to SVG user units (e.g., 0.8).
     */
    constructor(fileContent, scale = 1.0) {
        if (typeof fileContent !== 'string' || fileContent.length === 0) {
            throw new Error("TSV file content (string) must be provided.");
        }
        if (typeof scale !== 'number' || isNaN(scale)) {
            throw new Error("A valid number for the scale factor must be provided.");
        }
        this.fileContent = fileContent;
        this.scale = scale;
    }

    /**
     * Executes the full process: parses the TSV string, scales coordinates, groups words, 
     * filters results, and returns a JSON string.
     * @returns {string|null} A JSON string of the processed data, or null if an error occurs.
     */
    GetImageText() {
        // The entire logic is wrapped in a try/catch block for robust error handling during parsing.
        try {
            console.log(`Processing TSV content with a scale factor of ${this.scale}...`);

            // 1. --- DATA GATHERING AND SCALING ---
            // Process the provided file content string.
            const wordData = this.fileContent.split('\n')
                .slice(1) // Skip header row
                .map(line => {
                    const columns = line.split('\t');
                    // Handle potential empty or malformed lines gracefully.
                    if (columns.length < 12) return null;
                    
                    // Create a structured object for each word and apply the scale factor.
                    return {
                        level: parseInt(columns[0], 10),
                        page_num: parseInt(columns[1], 10),
                        block_num: parseInt(columns[2], 10),
                        par_num: parseInt(columns[3], 10),
                        line_num: parseInt(columns[4], 10),
                        word_num: parseInt(columns[5], 10),
                        left: Math.round(parseInt(columns[6], 10) * this.scale),
                        top: Math.round(parseInt(columns[7], 10) * this.scale),
                        width: Math.round(parseInt(columns[8], 10) * this.scale),
                        height: Math.round(parseInt(columns[9], 10) * this.scale),
                        conf: parseFloat(columns[10]),
                        text: columns[11] ? columns[11].trim() : ''
                    };
                })
                // Filter out any null entries and non-word data.
                .filter(word => word && word.level === 5 && word.text);

            // Group words by their line identifier.
            const lineGroups = wordData.reduce((groups, word) => {
                const key = `${word.page_num}-${word.block_num}-${word.par_num}-${word.line_num}`;
                if (!groups[key]) groups[key] = [];
                groups[key].push(word);
                return groups;
            }, {});

            const allFinalGroups = [];

            // 2. --- CORE GROUPING LOGIC ---
            for (const key in lineGroups) {
                const wordsOnLine = lineGroups[key].sort((a, b) => a.left - b.left);
                if (wordsOnLine.length === 0) continue;

                let currentGroup = {
                    left: wordsOnLine[0].left,
                    top: wordsOnLine[0].top,
                    width: wordsOnLine[0].width,
                    height: wordsOnLine[0].height,
                    text: wordsOnLine[0].text
                };

                for (let i = 1; i < wordsOnLine.length; i++) {
                    const lastWord = wordsOnLine[i - 1];
                    const currentWord = wordsOnLine[i];

                    // Note: The pixelGap and avgCharWidth are now in the scaled coordinate space,
                    // which is correct as their relationship remains proportional.
                    const pixelGap = currentWord.left - (lastWord.left + lastWord.width);
                    let avgCharWidth = (lastWord.text.length > 0) ? (lastWord.width / lastWord.text.length) : 10;
                    if (avgCharWidth < 1) avgCharWidth = 10;
                    const spaceThreshold = Math.round(avgCharWidth * 1.5);

                    if (pixelGap <= spaceThreshold) {
                        currentGroup.text += ` ${currentWord.text}`;
                        currentGroup.width = (currentWord.left + currentWord.width) - currentGroup.left;
                        currentGroup.height = Math.max(currentGroup.height, currentWord.height);
                    } else {
                        allFinalGroups.push(currentGroup);
                        currentGroup = {
                            left: currentWord.left,
                            top: currentWord.top,
                            width: currentWord.width,
                            height: currentWord.height,
                            text: currentWord.text
                        };
                    }
                }
                allFinalGroups.push(currentGroup);
            }

            // 2.5 --- ADDITIONAL FILTERING ---
            console.log(`Applying additional filter: ${allFinalGroups.length} groups before...`);
            const filteredGroups = allFinalGroups.filter(group => {
                const isShort = group.text.length < 3;
                const isNumeric = /^-?[\d\.]+$/.test(group.text);
                return !isShort || isNumeric;
            });
            console.log(`                             ${filteredGroups.length} groups after.`);

            // 3. --- RETURN FINAL DATA ---
            console.log(`\nProcessing complete. Returning JSON string.`);
            return JSON.stringify(filteredGroups, null, 2);

        } catch (error) {
            // Catch any errors, display them, and return null.
            console.error(`\nError: ${error.message}`);
            return null;
        }
    }
}

// --- HOW TO USE ---
// 1. Save this code as a JavaScript file (e.g., `process_ocr.js`).
// 2. Obtain the content of a Tesseract TSV file as a string.
// 3. Uncomment the following lines to create an instance and run the process.

/*
// Example TSV content string (replace with your actual data)
const tsvContent = `level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	1247	704	-1	
2	1	1	0	0	0	120	119	1004	35	-1	
3	1	1	1	0	0	120	119	1004	35	-1	
4	1	1	1	1	0	120	119	1004	35	-1	
5	1	1	1	1	1	120	122	153	29	96.882835	Sample
5	1	1	1	1	2	293	119	101	35	96.448517	Text
5	1	1	1	1	3	999	120	125	28	96.640526	Here
4	1	1	1	2	0	121	200	50	20	-1	
5	1	1	1	2	1	121	200	50	20	95.123456	123`;

// Create an instance of the processor with the string content.
const processor = new SVGXTsvProcessor(tsvContent);
const jsonResult = processor.GetImageText();

// The method returns the JSON string, which you can then use.
if (jsonResult) {
    console.log("\n--- JSON OUTPUT ---");
    console.log(jsonResult);
}
*/