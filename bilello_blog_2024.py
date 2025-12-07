import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Normalize title for filename
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text

# Target blog URL
blog_url = "https://johncandeto.com/information-flow/research-charlie-bilello-put-these-charts-on-your-wall-2024-edition"

# Output directory
output_dir = "inputs_2024"
os.makedirs(output_dir, exist_ok=True)

# Fetch and parse the blog page
response = requests.get(blog_url)
soup = BeautifulSoup(response.text, "html.parser")

# Traverse DOM: find each <li> and collect following <div class="notion-image"> blocks
chart_global_index = 1
for ul in soup.find_all("ul", class_="notion-bulleted-list"):
    li = ul.find("li")
    if not li:
        continue

    title = normalize(li.get_text(strip=True))
    image_index = 1

    # Traverse siblings until next <ul> or end
    next_tag = ul.find_next_sibling()
    while next_tag and next_tag.name != "ul":
        if next_tag.name == "div" and "notion-image" in next_tag.get("class", []):
            img = next_tag.find("img")
            if img and img.get("src") and "cdn-cgi/imagedelivery" in img["src"]:
                img_url = urljoin(blog_url, img["src"])
                filename = os.path.join(
                    output_dir,
                    f"chart_{chart_global_index:03d}_{title}_{image_index:02d}.jpg"
                )

                try:
                    img_data = requests.get(img_url).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"Saved: {filename}")
                    image_index += 1
                    chart_global_index += 1
                except Exception as e:
                    print(f"Failed to download {img_url}: {e}")

        next_tag = next_tag.find_next_sibling()

print(f"\n✅ Done. Saved {chart_global_index - 1} charts with indexed filenames.")