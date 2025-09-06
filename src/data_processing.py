import pandas as pd
from collections import OrderedDict
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import numpy as np
import hashlib
import os
import json
BASE = "https://chefjackovens.com"
ARCHIVE = f"{BASE}/recipes/"
HEADERS = {"User-Agent": "your-scraper/0.1 (+asd@gmail.com)"}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
LINKS_PATH = os.path.join(DATA_PATH, "processed_links.csv")
CACHE_DIR = os.path.join(DATA_PATH, "html_cache")

os.makedirs(CACHE_DIR, exist_ok=True)

class HTMLCache:
    @staticmethod
    def url_to_filename(url):
        # Use a hash to avoid filename issues
        return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".html")

    @staticmethod
    def save_html(url, html):
        with open(HTMLCache.url_to_filename(url), "w", encoding="utf-8") as f:
            f.write(html)

    @staticmethod
    def load_html(url):
        path = HTMLCache.url_to_filename(url)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    



class process_links:

    def __init__(self, links):
        self.links = links


    def extract_info(self, links):
        servings, titles, ingredients, nutrients, instructions = [] , [],  [], [], []
        for url in tqdm(links):
            html = HTMLCache.load_html(url)
            if html is None:
                r = requests.get(url, headers=HEADERS, timeout=20)
                html = r.text
                HTMLCache.save_html(url, html)
            soup = BeautifulSoup(html, "html.parser")
            print(f"Processing link {url}")
            servings.append(self.extract_servings(soup))
            titles.append(self.extract_title(soup))
            ingredients.append(self.extract_ingredients(soup))
            nutrients.append(self.extract_nutritional_values(soup))
            instructions.append(self.extract_instructions(soup))
        return servings, titles, ingredients, nutrients, instructions

    
    def extract_instructions(self, soup):

        instruction_groups = soup.select(".wprm-recipe-instruction-group")
        instruction_group_names = soup.select(".wprm-recipe-instruction-group-name")
        instructions = OrderedDict()
        try:
            for i in range(len(instruction_groups)):
                instruction_list = instruction_groups[i].select(".wprm-recipe-instruction-text")
                for i in range(len(instruction_list) - 1):
                    instruction = instruction_list[i].get_text()
                    group_name = instruction_group_names[i].get_text(strip=True) if i < len(instruction_group_names) else "General"
                    if group_name not in instructions:
                        instructions[group_name] = []
                    instructions[group_name].append(instruction)
        except IndexError:
            print("invalid index to fill later")
            instructions = np.nan
        return instructions

    def create_df(self, servings, titles, ingredients, nutrients, instructions):
        
        data = {
            "titles" : titles,
            "serving" : servings,
            "ingredients" : [json.dumps(ing) for ing in ingredients],
            "instructions" : [json.dumps(inst) for inst in instructions]
        }
        total_calories, total_protein, total_fat, total_carbs = [], [], [], []
        for d in nutrients:
            total_calories.append(d["calories"])
            total_protein.append(d["protein"])
            total_fat.append(d["fat"])
            total_carbs.append(d["carbohydrates"])
        data["calories"] = total_calories
        data["protein"] = total_protein
        data["fat"] = total_fat
        data["carbs"] = total_carbs
        # Create DataFrame and store full rows only
        df = pd.DataFrame(data).dropna()
        return df


    def extract_servings(self, soup):
        span = soup.select(".wprm-recipe-servings")
        serving = int(span[0].get_text(strip=True))
        return serving

    def extract_title(self, soup):
        full_title = soup.select_one(".entry-title").get_text(strip=True)
        return full_title

    def extract_ingredients(self, soup):
        ingredients_container = soup.select_one(".wprm-recipe-ingredients-container")
        groups = ingredients_container.select(".wprm-recipe-ingredient-group")
        return self.get_ingredients_from_groups(groups)
            

    def get_ingredients_from_groups(self,ingredient_groups):
        
        ingredient_categories = []
        for g in ingredient_groups:
            ingredient_per_group = OrderedDict()
            group_name_container = g.select_one(".wprm-recipe-group-name")
            group_name = group_name_container.get_text(strip=True) if group_name_container else "general ingredients"
            if group_name.lower() == "general ingredients":
                category = "general ingredients"
            else:
                category = f"ingredients for {group_name.lower()}"
            ingredient_per_group[category] = []
            ingredient_list = g.select("li.wprm-recipe-ingredient")
            for i in range(len(ingredient_list)):
                ingredient = ingredient_list[i]
                amount = ingredient.select_one(".wprm-recipe-ingredient-amount")
                amount = amount.get_text(strip=True) if amount else ""
                name = ingredient.select_one(".wprm-recipe-ingredient-name")
                name = name.get_text(strip=True) if name else ""
                ingredient_per_group[category].append(f"{amount} {name}")

            ingredient_categories.append(ingredient_per_group)
        return ingredient_categories

    def extract_nutritional_values(self, soup):
        nutritional_container = soup.select_one(".wprm-nutrition-label-layout")
        nutrients = {"protein" : np.nan , "fat" : np.nan, "calories" : np.nan, "carbohydrates" : np.nan}
        return self.get_nutritional_values(nutritional_container, nutrients)

    def get_nutritional_values(self, container, nutrients):
        if not container:
            return nutrients
        calories = int(container.select(".wprmp-nutrition-label-block-nutrient-extra-container")[0].get_text(strip=True))
        nutrients["calories"] = calories
        nutrients_containers = container.select(".wprmp-nutrition-label-block-nutrient-name-value-unit-container")[1:]
        for container in nutrients_containers:
            nutrient_name = container.select(".wprmp-nutrition-label-block-nutrient-name")[0].get_text(strip=True).lower()
            if nutrient_name not in nutrients:
                continue
            nutrient_val = float(container.select(".wprmp-nutrition-label-block-nutrient-value")[0].get_text(strip=True))
            nutrients[nutrient_name] = nutrient_val
        return nutrients

    def create_json(self, df, output_json=None):
        if output_json is None:
            output_json = os.path.join(DATA_PATH, "processed_info.json")
        
        # Use the df parameter directly, don't re-read from CSV
        df_copy = df.copy()
        
        # Convert JSON string columns back to dictionaries/lists
        df_copy['ingredients'] = df_copy['ingredients'].apply(json.loads)
        df_copy['instructions'] = df_copy['instructions'].apply(json.loads)

        # Convert DataFrame to list of dictionaries (records)
        records = df_copy.to_dict('records')

        # Save as JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"Saved JSON to {output_json}")


if __name__ == "__main__":
    assert os.path.exists(LINKS_PATH), "processed_links.csv not found, run link_processing.py first."
    
    # Remove header=None to read the header properly
    links_df = pd.read_csv(LINKS_PATH)
    links = links_df['url'].tolist()  # Use the column name
    
    pl = process_links(links)
    data_frame = pl.create_df(*pl.extract_info(links))
    
    # Save CSV
    OUT = os.path.join(DATA_PATH, "processed_info.csv")
    print("Finished processing links, constructing data set...")
    data_frame.to_csv(OUT, index=False)
    print(f"Saved CSV to {OUT}")

    # Create JSON for later use
    pl.create_json(data_frame)


