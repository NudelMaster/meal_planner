"""Data processing utilities for recipe text cleaning and normalization."""

from typing import List, Optional
import re


class RecipeTextCleaner:
    """Cleans and normalizes recipe text for better processing."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in recipe text.
        
        Args:
            text: Raw recipe text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    @staticmethod
    def clean_ingredients(ingredients: List[str]) -> List[str]:
        """Clean and normalize ingredient list.
        
        Args:
            ingredients: List of raw ingredient strings
            
        Returns:
            Cleaned ingredient list
        """
        cleaned = []
        for ingredient in ingredients:
            # Normalize whitespace
            ingredient = RecipeTextCleaner.normalize_whitespace(ingredient)
            # Remove empty strings
            if ingredient:
                cleaned.append(ingredient)
        return cleaned
    
    @staticmethod
    def clean_directions(directions: List[str]) -> List[str]:
        """Clean and normalize cooking directions.
        
        Args:
            directions: List of raw direction strings
            
        Returns:
            Cleaned directions list
        """
        cleaned = []
        for i, direction in enumerate(directions, 1):
            # Normalize whitespace
            direction = RecipeTextCleaner.normalize_whitespace(direction)
            # Remove existing numbering if present
            direction = re.sub(r'^\d+\.\s*', '', direction)
            # Remove empty strings
            if direction:
                cleaned.append(direction)
        return cleaned
    
    @staticmethod
    def extract_recipe_components(text: str) -> dict:
        """Extract recipe components from formatted text.
        
        Args:
            text: Formatted recipe text
            
        Returns:
            Dictionary with title, ingredients, and directions
        """
        # Placeholder for future implementation
        # This could parse formatted recipe text and extract components
        return {
            'title': '',
            'ingredients': [],
            'directions': []
        }


def tokenize_recipe(text: str, max_length: Optional[int] = None) -> List[str]:
    """Tokenize recipe text for embedding or analysis.
    
    Args:
        text: Recipe text to tokenize
        max_length: Maximum token length
        
    Returns:
        List of tokens
    """
    # Simple whitespace tokenization
    # In production, you might use a proper tokenizer
    tokens = text.split()
    
    if max_length:
        tokens = tokens[:max_length]
    
    return tokens
