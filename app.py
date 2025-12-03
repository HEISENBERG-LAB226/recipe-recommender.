import streamlit as st
from recommender import RecipeRecommender
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍳",
    layout="wide"
)

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return RecipeRecommender('recipes.csv')

recommender = load_recommender()

# App title and description
st.title("🍳 Ingredient-Based Recipe Recommendation System")
st.markdown("""
Welcome! Enter the ingredients you have, and I'll recommend the best recipes for you.
The system uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant matches.
""")

# Sidebar for filters
st.sidebar.header("Dietary Filters")
show_all = st.sidebar.checkbox("Show all recipes", value=True)

if not show_all:
    filter_options = st.sidebar.multiselect(
        "Exclude allergens:",
        ["eggs", "dairy", "nuts", "soy"],
        default=[]
    )
else:
    filter_options = []

# Main input area
st.header("Enter Your Ingredients")
user_input = st.text_area(
    "Type ingredients separated by commas (e.g., chicken, rice, garlic, onion):",
    height=100,
    placeholder="chicken, garlic, onion, tomato"
)

# Number of recommendations
num_recipes = st.slider("Number of recipes to show:", min_value=1, max_value=20, value=5)

# Search button
if st.button("🔍 Find Recipes", type="primary"):
    if user_input.strip():
        with st.spinner("Searching for the best recipes..."):
            # Get recommendations
            recommendations = recommender.recommend(
                user_input,
                top_n=num_recipes,
                exclude_allergens=filter_options
            )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recipes for you!")
                
                # Display results
                for idx, recipe in enumerate(recommendations, 1):
                    with st.expander(f"#{idx} - {recipe['recipe']} (Match: {recipe['similarity']:.1%})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Ingredients:**")
                            st.write(recipe['ingredients'])
                        
                        with col2:
                            allergens = recipe['allergens']
                            if allergens and allergens.lower() != 'none':
                                st.warning(f"⚠️ Contains: {allergens}")
                            else:
                                st.success("✅ No common allergens")
                        
                        # Similarity score visualization
                        st.progress(recipe['similarity'])
            else:
                st.warning("No recipes found. Try different ingredients or remove some filters.")
    else:
        st.error("Please enter at least one ingredient!")

# Footer with information
st.divider()
st.markdown("""
### How it works:
1. **TF-IDF Vectorization**: Converts ingredient lists into numerical vectors that capture the importance of each ingredient
2. **Cosine Similarity**: Measures how similar your ingredients are to each recipe (0 = no match, 1 = perfect match)
3. **Ranking**: Shows you the top matches based on similarity scores

**Dataset**: Contains over 130 recipes with allergen information (eggs, dairy, nuts, soy)
""")
