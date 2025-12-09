# Design Document: "Encode Path Legend" Feature for the SVGX Extension

This document outlines the design for the "Encode Path Legend" feature within the SVGX VS Code extension. That feature will be implemented as command in the SVGX extension. It provides both manual and automatic methods for associating data paths with their corresponding legend entries in SVGX files.

## 1. Introduction

The SVGX extension will be enhanced with a feature to encode relationships between graphical data elements (`<path>`) and their descriptive text legends (`<text>`). This encoding will be based on the `lc_legend_id`, `lc_legend_ref`, and `lc_legend_instance` attributes as specified in the "VIAIT Technical Note VIAIT-TN-010". This will enable more interactive and intelligent data visualizations by creating a machine-readable link between the visual representation of data and its meaning.

The feature will have two primary modes of operation:

- **Manual Encoding:** Allows the user to explicitly select a data path and a text element to create a legend association.
- **Automatic Encoding:** When no elements are selected, the feature will attempt to automatically associate all data paths with their corresponding legend texts within the active SVG document using a series of prioritized algorithms.

## 2. System Architecture

The feature will be built using TypeScript and will leverage the VS Code Extension API. It will consist of the following key components:

- **Command Contributor:** Registers the "Encode Path Legend" command in the VS Code command palette and context menus.
- **SVG Parser and Manipulator:** A chosen third-party library to parse the SVG document into a manipulable DOM-like structure and serialize it back to a string for updating the editor. Based on research, `svg.js` appears to be a strong candidate due to its comprehensive API for SVG manipulation.
- **User Interaction Handler:** Utilizes the VS Code API to get the user's current selection from the active text editor.
- **Legend Encoding Engine:** Contains the core logic for both manual and automatic encoding. This engine will be responsible for:
  - Generating unique `lc_legend_id` values.
  - Applying the `lc_legend_id`, `lc_legend_ref`, and `lc_legend_instance` attributes to the appropriate SVG elements.
  - Executing the automatic encoding algorithms in the specified order.
- **UI Notifier:** Provides feedback to the user via information messages, warnings, or error notifications.

## 3. Manual Path Legend Encoding

The manual encoding process will be initiated when the user has selected one `<path>` element and one `<text>` element in the active SVG editor and invokes the "Encode Path Legend" command.

### 3.1. User Workflow

1.  The user selects a single `<path>` element and a single `<text>` element within the SVG file in the VS Code editor.
2.  The user right-clicks and selects "Encode Path Legend" from the context menu, or triggers the command from the command palette.
3.  The extension applies the necessary `lc_legend_*` attributes to the selected elements.

### 3.2. Implementation Details

- **Get User Selection:** The `User Interaction Handler` will use `vscode.window.activeTextEditor.selection` to get the selected text ranges. It will then extract the full XML of the selected `<path>` and `<text>` elements.

- **Generate Unique `lc_legend_id`:**

  - The `Legend Encoding Engine` will use the content of the selected `<text>` element to create a base ID (e.g., by converting the text to a URL-friendly slug).
  - It will then scan the entire SVG document for existing `lc_legend_id` attributes to ensure the new ID is unique.
  - If a conflict is found, it will append a counter to the base ID (e.g., `my-legend-1`, `my-legend-2`) until a unique ID is generated.

- **Attribute Application:**
  - The generated unique ID will be added as the `lc_legend_id` attribute to the selected `<text>` element.
  - The same ID will be added to the `lc_legend_ref` attribute of the selected `<path>` element. If the `<path>` already has an `lc_legend_ref` attribute, the new ID will be appended to the existing array of references.

## 4. Automatic Path Legend Encoding

When the "Encode Path Legend" command is executed without any specific elements selected, the extension will attempt to automatically encode the entire SVG document. The following algorithms will be attempted in order, and the process will stop as soon as one of them successfully creates at least one legend association.

### 4.1. Algorithm 1: Style and Class Matching

This algorithm, inspired by the provided example implementation, will associate paths and legend symbols based on their visual styling.

- **Build a Legend-to-Style Map:**

  - The `Legend Encoding Engine` will iterate through all `<g>` elements that appear to be legend items (e.g., by looking for a combination of a `<text>` element and a shape element like `<rect>`, `<circle>`, or `<path>`).
  - For each legend item, it will extract the text from the `<text>` element and the computed style of the associated shape. The key style properties to capture will be `fill`, `stroke`, `stroke-width`, and `stroke-dasharray`. The `window.getComputedStyle()` method will be used to get the final rendered style.
  - This information will be stored in a `Map`, with the legend text as the key and the style object as the value.

- **Match Data Paths to Styles:**

  - The engine will then iterate through all `<path>` elements in the main plot area of the chart.
  - For each data path, it will get its computed style and compare it to the styles stored in the legend-to-style map. A match will be considered if the `fill`, `stroke`, and `stroke-dasharray` properties are identical. A tolerance can be allowed for `stroke-width`.

- **Apply Encoding:**
  - When a match is found, a unique `lc_legend_id` will be generated from the legend text.
  - This ID will be applied to the corresponding legend's `<text>` element as `lc_legend_id` and to the matching data path's `lc_legend_ref` attribute.

### 4.2. Algorithm 2: Order of Appearance

If the style matching algorithm fails to create any associations, the extension will fall back to matching based on the order of elements in the SVG document.

- **Gather Paths and Legend Texts:**
  - The engine will create two ordered lists: one of all data `<path>` elements and one of all legend `<text>` elements, based on their order of appearance in the SVG source.
- **Pair Elements:**
  - The extension will associate the first data path in the list with the first legend text, the second path with the second text, and so on.
- **Apply Encoding:**
  - For each pair, a unique `lc_legend_id` will be generated from the legend text and applied as `lc_legend_id` to the `<text>` element and as `lc_legend_ref` to the `<path>` element.

### 4.3. Algorithm 3: Proximity

If both preceding algorithms fail, the extension will attempt to associate paths and legend texts based on their geometric proximity.

- **Gather Element Bounding Boxes:**
  - The engine will calculate the bounding box (x, y, width, height) for every data `<path>` and every legend `<text>` element using the `getBoundingClientRect()` method.
- **Find Closest Text for Each Path:**
  - For each data path, the extension will calculate the Euclidean distance between the center of its bounding box and the center of every legend text's bounding box. The legend text with the shortest distance is considered its match.
  - To optimize this for large SVG files, a quadtree data structure (e.g., using a library like `d3-quadtree` or `quadtree-js`) can be used to spatially index the legend text elements for faster nearest-neighbor searches.
- **Apply Encoding:**
  - Once a path-text pair is identified, a unique `lc_legend_id` will be generated from the text and applied to both elements as described in the previous algorithms.

## 5. Error Handling and User Feedback

The extension will provide clear feedback to the user throughout the encoding process:

- **Successful Encoding:** A `vscode.window.showInformationMessage` will be displayed, indicating the number of path-legend associations that were created.
- **No Selection (Manual Mode):** If the user attempts manual encoding without selecting a `<path>` and a `<text>` element, a `vscode.window.showWarningMessage` will prompt them to make the correct selection.
- **No Associations Found (Automatic Mode):** If none of the automatic encoding algorithms create any associations, a `vscode.window.showInformationMessage` will inform the user that no matches could be found.

## 6. Future Enhancements

- **Multi-Select Manual Encoding:** Allow the user to select multiple paths and a single text element to associate all of them with that legend.
- **GUI for Legend Mapping:** A webview-based UI could allow users to visually drag and drop to create or edit legend associations.
- **Support for Other Shapes:** Extend the automatic encoding to support other data visualization shapes like `<rect>` (for bar charts) and `<circle>` (for scatter plots).
- **Configurable Matching Tolerance:** Allow users to configure the tolerance for style matching in the extension's settings.
