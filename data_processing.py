import pandas as pd
from collections import OrderedDict
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import numpy as np
BASE = "https://chefjackovens.com"
ARCHIVE = f"{BASE}/recipes/"
HEADERS = {"User-Agent": "your-scraper/0.1 (+asd@gmail.com)"}


def extract_info(links):
    servings, titles, ingredients, nutrients, instructions = [] , [],  [], [], []
    for url in tqdm(links):
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        print(f"Processing link {url}")
        servings.append(extract_servings(soup))
        titles.append(extract_title(soup))
        ingredients.append(extract_ingredients(soup))
        nutrients.append(extract_nutritional_values(soup))
        instructions.append(extract_instructions(soup))
    return servings, titles, ingredients, nutrients, instructions

def extract_instructions(soup):
        instruction_list = soup.select(".wprm-recipe-instruction-text")
        instructions = ""
        try: 
            for i in range(len(instruction_list) - 1):
                instruction = instruction_list[i].get_text()
                instructions += f"{i + 1}. {instruction}, "
            instructions += f"{len(instruction_list)}. {instruction_list[-1]}"
        except IndexError:
            print("invalid index to fill later")
            instructions = np.nan
        return instructions

def process_info(servings, titles, ingredients, nutrients, instructions):
    
    data = {
        "titles" : titles,
        "serving" : servings,
        "ingredients" : ingredients,
        "instructions" : instructions
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
    df = pd.DataFrame(data)
    return df


def extract_servings(soup):
    span = soup.select(".wprm-recipe-servings")
    serving = int(span[0].get_text(strip=True))
    return serving

def extract_title(soup):
    full_title = soup.select_one(".entry-title").get_text(strip=True)
    return full_title

def extract_ingredients(soup):
    ingredients_container = soup.select_one(".wprm-recipe-ingredients-container")
    groups = ingredients_container.select(".wprm-recipe-ingredient-group")
    return get_ingredients_from_groups(groups)
        

def get_ingredients_from_groups(ingredient_groups):
    ingredient_per_group = ""
    for g in ingredient_groups:
        group_name_container = g.select_one(".wprm-recipe-group-name")
        group_name = group_name_container.get_text(strip=True) if group_name_container else "general ingredients"
        ingredient_per_group += f"[{group_name}] :"
        ingredient_list = g.select("li.wprm-recipe-ingredient")
        for i in range(len(ingredient_list)):
            ingredient = ingredient_list[i]
            amount = ingredient.select_one(".wprm-recipe-ingredient-amount")
            amount = amount.get_text(strip=True) if amount else ""
            name = ingredient.select_one(".wprm-recipe-ingredient-name")
            name = name.get_text(strip=True) if name else ""
            if i == len(ingredient_list) - 1:
                ingredient_per_group += amount + " " + name
            else:
                ingredient_per_group += amount + " " + name + ", "
    return ingredient_per_group    

def extract_nutritional_values(soup):
    nutritional_container = soup.select_one(".wprm-nutrition-label-layout")
    nutrients = {"protein" : np.nan , "fat" : np.nan, "calories" : np.nan, "carbohydrates" : np.nan}
    return get_nutritional_values(nutritional_container, nutrients)

def get_nutritional_values(container, nutrients):
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




if __name__ == "__main__":
    links = pd.read_csv("processed_links.csv", header=None)[0].tolist()
    data_frame = process_info(*extract_info(links))
    print("Finished processing links, constructing data set...")
    data_frame.to_csv("processed_info.csv")


    