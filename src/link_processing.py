import pandas as pd
from collections import OrderedDict
import requests, re, time
from bs4 import BeautifulSoup
from ast import literal_eval
from tqdm import tqdm
import json
import os
BASE = "https://chefjackovens.com"
ARCHIVE = f"{BASE}/recipes/"
HEADERS = {"User-Agent": "your-scraper/0.1 (+asd@gmail.com)"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level to project root

def extract_recipe_links(html, processed : set):
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in tqdm(soup.select("a[href]")):
        href = a["href"]
        if href in processed:
            continue
        processed.add(href)
        try:
            r = requests.get(href, headers=HEADERS, timeout=20)
        except ValueError as err:
            print(err)
            continue
        if not is_recipe_page(r.text):
            continue
        if href.startswith(BASE) and re.search(r"/[a-z0-9-]+/$", href):
            if not any(x in href for x in ("/tag/","/category/","/recipes/")):
                out.add(href)
    return out

def is_recipe_page(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")

    # Look at every <script type="application/ld+json">
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        # Make sure we handle both dict and list
        blocks = data if isinstance(data, list) else [data]

        for block in blocks:
            # Sometimes wrapped in @graph
            if isinstance(block, dict) and "@graph" in block:
                subblocks = block["@graph"]
                if not isinstance(subblocks, list):
                    subblocks = [subblocks]
            else:
                subblocks = [block]

            for obj in subblocks:
                if isinstance(obj, dict):
                    t = obj.get("@type")
                    # @type can be a string or a list
                    if t == "Recipe" or (isinstance(t, list) and "Recipe" in t):
                        return True

    return False

    
def get_links():
    links = set()
    page = 1
    processed = set()
    while True:
        url = ARCHIVE if page == 1 else f"{ARCHIVE}page/{page}/"
        print(f"processing url {url}")
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 404:
            break
        new = extract_recipe_links(r.text, processed)
        if not new or new.issubset(links):
            break
        links |= new
        page += 1
        time.sleep(1.0)  # be polite

    # print("Total recipe links:", len(links))
    return links


if __name__ == "__main__":
    links = get_links()
    # Be explicit about the DataFrame structure
    df = pd.DataFrame(list(links), columns=['url'])  # Convert set to list and name the column
    OUT = os.path.join(PROJECT_ROOT, "data", "processed_links.csv")
    df.to_csv(OUT, index=False)
    print(f"Saved {len(links)} links to {OUT}")