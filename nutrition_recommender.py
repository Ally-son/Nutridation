import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import joblib

class UserProfile:
    def __init__(self, age: int, weight: float, height: float, gender: str, activity_level: str):
        self.age = age
        self.weight = weight  # in kg
        self.height = height  # in cm
        self.gender = gender.lower()
        self.activity_level = activity_level.lower()
        self.bmi = self._calculate_bmi()
        self.daily_calories = self._calculate_daily_calories()
        self.nutritional_needs = self._calculate_nutritional_needs()
    
    def _calculate_bmi(self) -> float:
        return self.weight / ((self.height/100) ** 2)
    
    def _calculate_daily_calories(self) -> float:
        # Harris-Benedict equation for BMR
        if self.gender == 'male':
            bmr = 88.362 + (13.397 * self.weight) + (4.799 * self.height) - (5.677 * self.age)
        else:
            bmr = 447.593 + (9.247 * self.weight) + (3.098 * self.height) - (4.330 * self.age)
        
        # Activity multiplier
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly active': 1.375,
            'moderately active': 1.55,
            'very active': 1.725,
            'extra active': 1.9
        }
        
        return bmr * activity_multipliers.get(self.activity_level, 1.2)
    
    def _calculate_nutritional_needs(self) -> Dict[str, float]:
        # Calculate daily nutritional needs based on user profile
        calories = self.daily_calories
        
        needs = {
            'Calories': calories,
            'ProteinContent': max(0.8 * self.weight, 60),  # minimum 60g or 0.8 * weight
            'CarbohydrateContent': (calories * 0.5) / 4,  # 50% of calories from carbs (4 cal/g)
            'FatContent': (calories * 0.3) / 9,  # 30% of calories from fat (9 cal/g)
            'SugarContent': min(50, (calories * 0.1) / 4),  # max 50g or 10% of calories
            'FiberContent': 25 if self.age < 50 else 21,  # age-based fiber recommendation
            'SodiumContent': 2000  # 2000mg sodium limit
        }
        return needs

class NutritionRecommender:
    def __init__(self, data_path: str, scaler_path: str):
        self.data = pd.read_parquet(data_path)
        self.scaler = joblib.load(scaler_path)
        self.nutrition_cols = ['Calories', 'ProteinContent', 'CarbohydrateContent', 
                             'FatContent', 'SugarContent', 'FiberContent', 'SodiumContent']
        
        # Define which nutrients have upper limits (rather than targets)
        self.limit_nutrients = {'SugarContent', 'SodiumContent'}
        
        # Get standardized nutrition columns (without 'Original_' prefix)
        self.standardized_nutrition = self.data[self.nutrition_cols].values
        
        # Map original nutrition columns
        self.original_cols = [f'Original_{col}' for col in self.nutrition_cols]
        
        # Define importance weights for different nutrients
        self.nutrient_weights = np.array([
            1.0,  # Calories
            1.2,  # Protein
            1.0,  # Carbs
            1.0,  # Fat
            0.8,  # Sugar
            1.1,  # Fiber
            2.0   # Sodium - even higher weight to strongly prioritize low-sodium options
        ])
        
        # Apply weights to standardized nutrition values
        self.weighted_nutrition = self.standardized_nutrition * self.nutrient_weights[np.newaxis, :]
        
        # Normalize the weighted vectors for faster cosine similarity
        norms = np.linalg.norm(self.weighted_nutrition, axis=1)
        self.normalized_nutrition = self.weighted_nutrition / norms[:, np.newaxis]
        
        # Define meal ranges and preferences with stronger category preferences
        self.meal_target_ranges = {
            'breakfast': {
                'min': 0.2, 'max': 0.35,
                'preferred_categories': ['Breakfast', 'Breads', 'Eggs', 'Pancakes', 'Cereals', 'Muffins', 'French Toast'],
                'category_boost': 1.5  # Increased from 1.2
            },
            'lunch': {
                'min': 0.3, 'max': 0.4,
                'preferred_categories': ['Main Dish', 'Salads', 'Sandwiches', 'Soups', 'Vegetables', 'Chicken', 'Fish'],
                'category_boost': 1.4
            },
            'dinner': {
                'min': 0.25, 'max': 0.35,
                'preferred_categories': ['Main Dish', 'Meat', 'Seafood', 'Pasta', 'Rice', 'Vegetables', 'Chicken', 'Fish'],
                'category_boost': 1.4
            }
        }
        
        # Define stricter sodium limits per meal
        self.sodium_limits = {
            'breakfast': 500,  # Reduced from 600
            'lunch': 700,      # Reduced from 800
            'dinner': 500      # Reduced from 600
        }
    
    def _calculate_similarities(self, target_vector: np.ndarray) -> np.ndarray:
        # Normalize target vector
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return np.zeros(len(self.normalized_nutrition))
        normalized_target = target_vector / target_norm
        
        # Calculate cosine similarities using vectorized operations
        return np.dot(self.normalized_nutrition, normalized_target.flatten())
    
    def _calculate_meal_score(self, recipe_idx: int, meal_type: str, 
                            similarity: float, daily_needs: Dict[str, float]) -> float:
        """Calculate a comprehensive score for a recipe for a specific meal type."""
        recipe = self.data.iloc[recipe_idx]
        category = recipe['RecipeCategory']
        
        # Base score is the similarity
        score = similarity
        
        # Stronger boost for preferred meal categories
        if category in self.meal_target_ranges[meal_type]['preferred_categories']:
            score *= self.meal_target_ranges[meal_type]['category_boost']
        else:
            score *= 0.7  # Penalty for non-preferred categories
        
        # Penalize high sodium content relative to limit
        sodium_content = recipe[f'Original_SodiumContent']
        sodium_limit = self.sodium_limits[meal_type]
        if sodium_content > 0:
            sodium_ratio = sodium_content / sodium_limit
            if sodium_ratio > 0.8:
                score *= (1.0 - (sodium_ratio - 0.8))
        
        return score
    
    def recommend_meals(self, user_profile: UserProfile, n_meals: int = 3) -> List[Dict]:
        daily_needs = user_profile.nutritional_needs
        recommendations = []
        meal_types = ['breakfast', 'lunch', 'dinner']
        
        # Convert daily needs to array and apply weights
        needs_array = np.array([[daily_needs[col] for col in self.nutrition_cols]])
        weighted_needs = needs_array * self.nutrient_weights
        
        for meal_type in meal_types[:n_meals]:
            # Adjust needs for meal type
            meal_range = self.meal_target_ranges[meal_type]
            meal_needs = weighted_needs * meal_range['max']
            
            # Calculate similarities for all recipes at once
            similarities = self._calculate_similarities(meal_needs.T)
            
            # Calculate comprehensive scores for each recipe
            scores = np.array([
                self._calculate_meal_score(i, meal_type, sim, daily_needs)
                for i, sim in enumerate(similarities)
            ])
            
            # Get indices sorted by score
            sorted_indices = np.argsort(scores)[::-1]
            
            meal_options = []
            considered = 0
            
            while len(meal_options) < 5 and considered < len(sorted_indices):
                batch_size = min(20, len(sorted_indices) - considered)
                batch_indices = sorted_indices[considered:considered + batch_size]
                considered += batch_size
                
                # Get nutritional info for batch
                batch_nutrition = {
                    col.replace('Original_', ''): self.data.iloc[batch_indices][f'Original_{col.replace("Original_", "")}'].values
                    for col in self.original_cols
                }
                
                # Filter based on meal-specific constraints
                calories_min = daily_needs['Calories'] * meal_range['min']
                calories_max = daily_needs['Calories'] * meal_range['max']
                
                valid_mask = (
                    (batch_nutrition['Calories'] >= calories_min) &
                    (batch_nutrition['Calories'] <= calories_max) &
                    (batch_nutrition['SodiumContent'] <= self.sodium_limits[meal_type])
                )
                
                valid_indices = batch_indices[valid_mask]
                
                for idx in valid_indices:
                    if len(meal_options) >= 5:
                        break
                        
                    recipe = self.data.iloc[idx]
                    original_nutrition = {
                        col.replace('Original_', ''): recipe[f'Original_{col.replace("Original_", "")}']
                        for col in self.original_cols
                    }
                    
                    meal_options.append({
                        'Name': recipe['Name'],
                        'Category': recipe['RecipeCategory'],
                        'Nutritional_Info': original_nutrition,
                        'Instructions': recipe['RecipeInstructions'],
                        'Ingredients': recipe['RecipeIngredientParts'],
                        'Similarity_Score': similarities[idx],
                        'Overall_Score': scores[idx],
                        'Percentage_of_Daily': {
                            nutrient: (amount / daily_needs[nutrient]) * 100
                            for nutrient, amount in original_nutrition.items()
                        }
                    })
            
            recommendations.append({
                meal_type: meal_options
            })
        
        return recommendations
    
    def evaluate_meal_combination(self, daily_needs: Dict[str, float], 
                                meal_choices: List[Dict]) -> Dict[str, float]:
        """Evaluate a specific combination of meal choices."""
        total_nutrition = {col: 0 for col in self.nutrition_cols}
        
        for meal in meal_choices:
            for nutrient, amount in meal['Nutritional_Info'].items():
                total_nutrition[nutrient] += amount
        
        coverage = {}
        for nutrient, amount in total_nutrition.items():
            target = daily_needs[nutrient]
            if nutrient in self.limit_nutrients:
                # For limit nutrients, 100% means hitting the limit
                # Being under limit is good
                coverage[nutrient] = (amount / target) * 100
                if coverage[nutrient] <= 100:
                    # Adjust score to show that being under limit is good
                    coverage[nutrient] = 100 + (100 - coverage[nutrient]) * 0.1
            else:
                coverage[nutrient] = (amount / target) * 100
        
        return {
            'Total_Nutrition': total_nutrition,
            'Coverage_Percentage': coverage
        }
    
    def evaluate_recommendations(self, user_profile: UserProfile, 
                               recommendations: List[Dict]) -> Dict:
        daily_needs = user_profile.nutritional_needs
        
        # Get all possible meal combinations
        meal_options = []
        for meal_dict in recommendations:
            meal_type = list(meal_dict.keys())[0]
            options = meal_dict[meal_type]
            meal_options.append(options[:3])
        
        # Evaluate first combination (top choice from each meal)
        best_combo = [options[0] for options in meal_options]
        best_eval = self.evaluate_meal_combination(daily_needs, best_combo)
        
        return {
            'Daily_Needs': daily_needs,
            'Best_Combination': {
                'Meals': best_combo,
                'Total_Nutrition': best_eval['Total_Nutrition'],
                'Coverage_Percentage': best_eval['Coverage_Percentage']
            },
            'Note': 'You can mix and match different meal options to better meet your nutritional needs.'
        }

# Example usage:
if __name__ == "__main__":
    # Create a user profile
    user = UserProfile(
        age=20,
        weight=72,  # kg
        height=175,  # cm
        gender='male',
        activity_level='lightly active'
    )
    
    # Initialize recommender
    recommender = NutritionRecommender(
        data_path=r"C:\Users\65868\OneDrive - Nanyang Technological University\VS code\sc1304 proj\cleaned_recipes.parquet",
        scaler_path=r"C:\Users\65868\OneDrive - Nanyang Technological University\VS code\sc1304 proj\scaler.joblib"
    )
    
    # Get recommendations
    recommendations = recommender.recommend_meals(user, n_meals=3)
    
    # Evaluate recommendations
    evaluation = recommender.evaluate_recommendations(user, recommendations)
    
    # Print detailed results
    print("\n" + "="*80)
    print("NUTRITION RECOMMENDATION SYSTEM - RESULTS")
    print("="*80)
    
    print("\nUSER PROFILE:")
    print(f"Age: {user.age} years")
    print(f"Weight: {user.weight} kg")
    print(f"Height: {user.height} cm")
    print(f"Gender: {user.gender}")
    print(f"Activity Level: {user.activity_level}")
    print(f"BMI: {user.bmi:.2f}")
    print(f"Daily Calorie Needs: {user.daily_calories:.2f} kcal")
    
    print("\nDAILY NUTRITIONAL TARGETS:")
    for nutrient, amount in evaluation['Daily_Needs'].items():
        unit = 'mg' if nutrient == 'SodiumContent' else 'g'
        print(f"{nutrient.replace('Content', '')}: {amount:.1f}{unit}")
    
    print("\nRECOMMENDED MEALS:")
    for meal_dict in recommendations:
        meal_name = list(meal_dict.keys())[0]
        options = meal_dict[meal_name]
        
        print(f"\n{meal_name.upper()}:")
        print("="*40)
        
        for i, option in enumerate(options, 1):
            print(f"\nOption {i}: {option['Name']}")
            print(f"Category: {option['Category']}")
            
            print("\nNutritional Information:")
            print("-" * 30)
            for nutrient, amount in option['Nutritional_Info'].items():
                unit = 'mg' if nutrient == 'SodiumContent' else 'g'
                daily_pct = option['Percentage_of_Daily'][nutrient]
                nutrient_name = nutrient.replace('Content', '')
                print(f"  {nutrient_name:<12}: {amount:>6.1f}{unit:<4} ({daily_pct:>5.1f}% of daily)")
            
            print(f"\nSimilarity Score: {option['Similarity_Score']:.2f}")
            print(f"Overall Score: {option['Overall_Score']:.2f}")
            
            # Show ingredients preview (first 3)
            ingredients = option['Ingredients'].split(',')[:3]
            print("\nKey Ingredients:", ', '.join(ingredients) + "...")
    
    print("\nNUTRITIONAL COVERAGE ANALYSIS:")
    print("=" * 40)
    print("\nBest Combination (using first option from each meal):")
    for nutrient, percentage in evaluation['Best_Combination']['Coverage_Percentage'].items():
        status = "✓" if 90 <= percentage <= 110 else "!" if 75 <= percentage <= 125 else "✗"
        nutrient_name = nutrient.replace('Content', '')
        target = evaluation['Daily_Needs'][nutrient]
        actual = evaluation['Best_Combination']['Total_Nutrition'][nutrient]
        unit = 'mg' if nutrient == 'SodiumContent' else 'g'
        
        print(f"\n{nutrient_name}:")
        print(f"  Target: {target:>6.1f}{unit}")
        print(f"  Actual: {actual:>6.1f}{unit}")
        print(f"  Coverage: {percentage:>6.1f}% {status}")
    
    print("\nRECOMMENDATION SUMMARY:")
    print("=" * 40)
    coverage_scores = list(evaluation['Best_Combination']['Coverage_Percentage'].values())
    avg_coverage = sum(coverage_scores) / len(coverage_scores)
    print(f"Average Nutritional Coverage: {avg_coverage:.1f}%")
    
    # Print overall assessment
    if all(75 <= p <= 125 for p in coverage_scores):
        if all(90 <= p <= 110 for p in coverage_scores):
            print("Overall Assessment: Excellent! All nutritional needs are well balanced.")
        else:
            print("Overall Assessment: Good. Nutritional needs are adequately met with some room for improvement.")
    else:
        print("Overall Assessment: Fair. Some nutritional targets are significantly under or over the recommended values.")
    
    print("\nNOTE: You can mix and match different meal options to better meet your nutritional needs.")
    print("      Each meal option shows its percentage of your daily requirements to help you make choices.") 