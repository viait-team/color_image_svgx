
// ==============================================================================
// Definitive Script for Adding Logical Mapping to PLOTLY-generated SVGs
// Converted from R to JavaScript.
// ==============================================================================

// -- 1. Core Conversion Function --
function convert_date_to_ticks(r_posixct_date) {

  const EPOCH_DIFF_DAYS = 719163;
  const EPOCH_DIFF_SECONDS = BigInt(EPOCH_DIFF_DAYS) * BigInt(86400);
  const TICKS_PER_SECOND = BigInt(10000000);
  const seconds_from_unix_epoch = BigInt(Math.round(r_posixct_date.getTime() / 1000));
  const total_seconds = seconds_from_unix_epoch + EPOCH_DIFF_SECONDS;
  return total_seconds * TICKS_PER_SECOND;

}

// -- 2. Helper Functions Specifically for PLOTLY --
function parse_plotly_transform(transform_str) {

  if (!transform_str) return { x: 0, y: 0 };
  const matches = transform_str.match(/translate\(([^,]+),([^\)]+)\)/);
  if (!matches) return { x: 0, y: 0 };
  return { x: parseFloat(matches[1]), y: parseFloat(matches[2]) };

}

function find_y_grid_points_plotly(svg_doc) {

  const grid_paths = Array.from(svg_doc.querySelectorAll(".gridlayer .y path"));
  let grid_points_list = [];
  if (grid_paths.length > 0) {
    for (const p of grid_paths) {
      const transform_coords = parse_plotly_transform(p.getAttribute("transform"));
      const final_grid_y = transform_coords.y;
      const d_attr = p.getAttribute("d");
      const match = d_attr.match(/M\s*([\d\.-]+)/);

      if (match) {
        const final_grid_x = parseFloat(match[1]);
        grid_points_list.push({ grid_x: final_grid_x, grid_y: final_grid_y });
      }
    }
  }

  if (grid_points_list.length === 0) return null;
  return grid_points_list;
}

function extract_yaxis_mapping_plotly(svg_doc) {

  const grid_points = find_y_grid_points_plotly(svg_doc);
  if (!grid_points || grid_points.length < 2) {
    throw new Error("Plotly Y-Axis: Could not find at least 2 horizontal grid lines.");
  }

  const ytick_groups = Array.from(svg_doc.querySelectorAll(".ytick"));
  let labels_list = [];
  if (ytick_groups.length > 0) {

    for (const g of ytick_groups) {
      const text_node = g.querySelector("text");
      if (text_node) {
        const label_text = text_node.textContent;
        const transform_coords = parse_plotly_transform(text_node.getAttribute("transform"));
        const text_x = parseFloat(text_node.getAttribute("x"));
        const text_y = parseFloat(text_node.getAttribute("y"));
        const final_label_x = transform_coords.x + text_x;
        const final_label_y = transform_coords.y + text_y;
        labels_list.push({ text: label_text, label_x: final_label_x, label_y: final_label_y });
      }
    }
  }

  const raw_labels = labels_list;
  const labels = raw_labels.map(l => ({...l, logical: parseFloat(l.text) })).filter(l => !isNaN(l.logical));

  if (labels.length < 2) throw new Error("Plotly Y-Axis: Could not find at least 2 numeric labels.");

  const sorted_grid_y = [...new Set(grid_points.map(p => p.grid_y))].sort((a, b) => a - b);
  const avg_spacing = sorted_grid_y.slice(1).map((y, i) => y - sorted_grid_y[i]).reduce((a, b) => a + b, 0) / (sorted_grid_y.length - 1);
  const threshold = avg_spacing * 0.8;

  let reference_points_list = [];
  if (grid_points.length > 0) {

    for (const grid_row of grid_points) {
      const distances = labels.map(l => ({...l, distance: Math.sqrt(Math.pow(l.label_x - grid_row.grid_x, 2) + Math.pow(l.label_y - grid_row.grid_y, 2)) })).filter(l => l.distance < threshold);
      if (distances.length > 0) {
        const min_distance = distances.reduce((min, l) => l.distance < min.distance ? l : min, distances[0]);
        reference_points_list.push({ logical: min_distance.logical, visual: grid_row.grid_y });
      }
    }
  }

  const reference_points = [...new Map(reference_points_list.map(item => [JSON.stringify(item, (key, value) => typeof value === 'bigint' ? value.toString() : value), item])).values()];

  if (reference_points.length < 2) throw new Error("Plotly Y-Axis: Robust evaluation failed.");

  return reference_points;
}

function find_x_grid_points_plotly(svg_doc) {

  const grid_paths = Array.from(svg_doc.querySelectorAll(".gridlayer .x path"));
  let grid_points_list = [];

  if (grid_paths.length > 0) {

    for (const p of grid_paths) {
      const transform_coords = parse_plotly_transform(p.getAttribute("transform"));
      const final_grid_x = transform_coords.x;
      const d_attr = p.getAttribute("d");
      const match = d_attr.match(/M\s*[\d\.-]+\s*,?\s*([\d\.-]+)\s*v\s*([\d\.-]+)/);

      if (match) {
        const start_y_offset = parseFloat(match[1]);
        const line_length = parseFloat(match[2]);
        const top_y = transform_coords.y + start_y_offset;
        const bottom_y = top_y + line_length;

        grid_points_list.push({ grid_x: final_grid_x, grid_y: bottom_y });
      }
    }
  }

  if (grid_points_list.length === 0) return null;

  return grid_points_list;
}

function parseDateLabel(labelObject) {
  const text = labelObject.text.replace(',', ' ');
  const dateFromText = new Date(text);

  if (!isNaN(dateFromText.getTime())) {
    return { ...labelObject, parsed_date: dateFromText };
  }

  const yearFromFloat = parseFloat(text);
  if (!isNaN(yearFromFloat)) {
    const dateFromYear = new Date(yearFromFloat, 1, 1);
    return { ...labelObject, parsed_date: dateFromYear };
  }

  return { ...labelObject, parsed_date: new Date(NaN) };
}

function extract_xaxis_mapping_date_plotly(svg_doc) {

  const grid_points = find_x_grid_points_plotly(svg_doc);
  if (!grid_points || grid_points.length < 2) {
    throw new Error("Plotly X-Axis: Could not find at least 2 vertical grid lines.");
  }

  const xtick_groups = Array.from(svg_doc.querySelectorAll(".xtick"));
  let labels_list = [];
  if (xtick_groups.length > 0) {

    for (const g of xtick_groups) {
      const text_node = g.querySelector("text");
      if (text_node) {
        const transform_coords = parse_plotly_transform(text_node.getAttribute("transform"));
        const text_x = parseFloat(text_node.getAttribute("x"));
        const text_y = parseFloat(text_node.getAttribute("y"));
        const final_label_x = transform_coords.x + text_x;
        const final_label_y = transform_coords.y + text_y;

        labels_list.push({ text: text_node.textContent, label_x: final_label_x, label_y: final_label_y });
      }
    }
  }

  //
  const raw_labels = labels_list;

  const labels = raw_labels
    .map(parseDateLabel)
    .filter(l => !isNaN(l.parsed_date.getTime()))
    .map(l => ({ ...l, logical: convert_date_to_ticks(l.parsed_date) }));

  if (labels.length < 2) {
    throw new Error("Plotly X-Axis: Could not find at least 2 parsable date labels.");
  }
 
  const reference_points = labels.map(l => {
    const closest_grid_x = grid_points.reduce((closest, p) => Math.abs(p.grid_x - l.label_x) < Math.abs(closest.grid_x - l.label_x) ? p : closest, grid_points[0]).grid_x;
    return {...l, closest_grid_x };
  }).filter(l => Math.abs(l.label_x - l.closest_grid_x) < 2)
    .map(l => ({ logical: l.logical, visual: l.closest_grid_x }));

  const distinct_reference_points = [...new Map(reference_points.map(item => [JSON.stringify(item, (key, value) => typeof value === 'bigint' ? value.toString() : value), item])).values()];

  if (distinct_reference_points.length < 2) throw new Error("Plotly X-Axis: Robust evaluation failed.");

  return distinct_reference_points;
}

function addLogicMapping(svg_string) {
  const parser = new DOMParser();
  const svg_doc = parser.parseFromString(svg_string, "image/svg+xml");

  const y_ref_points = extract_yaxis_mapping_plotly(svg_doc);

  const x_ref_points = extract_xaxis_mapping_date_plotly(svg_doc);

  const x_ref_points_sorted = [...x_ref_points].sort((a, b) => (a.logical < b.logical) ? -1 : ((a.logical > b.logical) ? 1 : 0));
  const y_ref_points_sorted = [...y_ref_points].sort((a, b) => a.logical - b.logical);

  const d_x_1 = x_ref_points_sorted[0].logical;
  const v_x_1 = x_ref_points_sorted[0].visual;
  const d_x_2 = x_ref_points_sorted[x_ref_points_sorted.length - 1].logical;
  const v_x_2 = x_ref_points_sorted[x_ref_points_sorted.length - 1].visual;

  const d_y_1 = y_ref_points_sorted[0].logical;
  const v_y_1 = y_ref_points_sorted[0].visual;
  const d_y_2 = y_ref_points_sorted[y_ref_points_sorted.length - 1].logical;
  const v_y_2 = y_ref_points_sorted[y_ref_points_sorted.length - 1].visual;

  const xlm_string = `[${d_x_1.toString()}, ${d_x_2.toString()}, ${v_x_1.toFixed(2)}, ${v_x_2.toFixed(2)}]`;
  const ylm_string = `[${d_y_1.toFixed(2)}, ${d_y_2.toFixed(2)}, ${v_y_1.toFixed(2)}, ${v_y_2.toFixed(2)}]`;

  const svg_root = svg_doc.documentElement;

  svg_root.setAttribute("ylm", ylm_string);
  svg_root.setAttribute("xlm", xlm_string);

  const serializer = new XMLSerializer();
  const new_svg_string = serializer.serializeToString(svg_doc);

  // console.log("Successfully added logical mapping attributes to SVG");
  // console.log(`  (Found ${y_ref_points.length} confident Y-axis pairs and ${x_ref_points.length} confident X-axis pairs)`);
  // console.log(`  xlm = ${xlm_string}`);
  // console.log(`  ylm = ${ylm_string}`);
  // return svg_string;

  return new_svg_string;
}

