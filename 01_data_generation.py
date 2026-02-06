"""
Topic 3 In-Class Activity - Part 1: Data Generation
====================================================

PURPOSE:
--------
This script creates a synthetic dataset that will help us understand how regularization works.
We create synthetic (artificial) data because we KNOW the true relationships, which helps us
evaluate how well our models recover those relationships.

WHAT WE'RE CREATING:
-------------------
- A dataset with 500 samples
- 8 predictor variables (features)
- 1 target variable (what we're trying to predict)
- Some features are truly important, others are just noise
- Some features are correlated with each other (like in real data)

KEY LEARNING:
------------
By controlling exactly what relationships exist in the data, we can see whether
regularization helps us identify the truly important features.
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility (so everyone gets the same results)
np.random.seed(42)

print("Libraries imported successfully!")
print("=" * 60)

# ============================================================================
# STEP 2: Generate Predictor Variables (Features)
# ============================================================================

# Number of samples (observations) in our dataset
n_samples = 500

# Number of features (predictor variables)
n_features = 8

print(f"\nGenerating dataset with {n_samples} samples and {n_features} features...")

# Generate the base features from a normal distribution
# This creates random numbers with mean=0 and standard deviation=1
X = np.random.randn(n_samples, n_features)

# TO EXPERIMENT: Try changing the seed above and see how results change
# TO EXPERIMENT: Try changing n_samples to see how it affects model performance

# ============================================================================
# STEP 3: Add Correlation Between Features (Simulate Real-World Data)
# ============================================================================

# In real data, features are often correlated. For example, in housing data:
# - Lot size and house size are often correlated
# - Number of bedrooms and bathrooms are often correlated
# We'll simulate this by making some features partially dependent on others

print("\nAdding correlations between features...")

# Make feature 2 partially dependent on feature 1
# This simulates correlated features (like house size and lot size)
X[:, 1] = X[:, 1] + 0.7 * X[:, 0]

# Make feature 4 partially dependent on feature 3
X[:, 3] = X[:, 3] + 0.6 * X[:, 2]

# QUESTION TO CONSIDER: 
# Why does correlation between features matter for regression?
# HINT: When two features are highly correlated, it's hard to determine
# which one is truly responsible for predicting the target.

# ============================================================================
# STEP 4: Create the Target Variable (What We're Trying to Predict)
# ============================================================================

# Define TRUE coefficients for our features
# These are the REAL relationships we want our models to discover
true_coefficients = np.array([
    5.0,    # Feature 0: IMPORTANT (large coefficient)
    0.0,    # Feature 1: NOT IMPORTANT (coefficient = 0)
    3.0,    # Feature 2: IMPORTANT (medium coefficient)
    0.0,    # Feature 3: NOT IMPORTANT (coefficient = 0)
    -2.0,   # Feature 4: IMPORTANT (negative relationship)
    0.0,    # Feature 5: NOT IMPORTANT (coefficient = 0)
    0.0,    # Feature 6: NOT IMPORTANT (coefficient = 0)
    0.0     # Feature 7: NOT IMPORTANT (coefficient = 0)
])

print("\nTrue coefficients (what we want to recover):")
for i, coef in enumerate(true_coefficients):
    importance = "IMPORTANT" if coef != 0 else "NOISE"
    print(f"  Feature {i}: {coef:6.1f}  ({importance})")

# Calculate the target variable using the true coefficients
# y = X * true_coefficients + noise
y = X @ true_coefficients  # @ is matrix multiplication in Python

# Add some random noise to make it realistic
# In real data, predictions are never perfect
noise_level = 2.0
noise = np.random.randn(n_samples) * noise_level
y = y + noise

print(f"\nAdded noise with standard deviation = {noise_level}")

# KEY INSIGHT:
# We know that features 0, 2, and 4 are important (non-zero coefficients)
# Features 1, 3, 5, 6, 7 are just noise (zero coefficients)
# Let's see if regularization helps us identify this!

# ============================================================================
# STEP 5: Convert to DataFrame for Easier Handling
# ============================================================================

# Create column names for our features
feature_names = [f'Feature_{i}' for i in range(n_features)]

# Create a pandas DataFrame (like a spreadsheet in Python)
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

print("\n" + "=" * 60)
print("Dataset created successfully!")
print(f"Shape: {df.shape} (rows, columns)")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())

# ============================================================================
# STEP 6: Split into Training and Testing Sets
# ============================================================================

# We need separate data for:
# - TRAINING: Teaching the model
# - TESTING: Evaluating how well it learned

# Separate features (X) from target (y)
X_full = df[feature_names].values
y_full = df['Target'].values

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print("\n" + "=" * 60)
print("Data Split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# WHY DO WE SPLIT THE DATA?
# - Training set: Used to fit the model (learn the coefficients)
# - Testing set: Used to evaluate how well the model generalizes to NEW data
# - This helps us detect overfitting!

# ============================================================================
# STEP 7: Standardize Features (IMPORTANT for Regularization!)
# ============================================================================

# CRITICAL CONCEPT:
# Regularization penalizes large coefficients. But coefficient magnitude depends
# on the scale of features! For example:
# - Feature in dollars ($50,000) vs. feature in meters (50)
# - Same importance, but dollar feature would have much smaller coefficient
# 
# SOLUTION: Standardize all features to have mean=0 and std=1
# This puts all features on the same scale, making regularization fair.

scaler = StandardScaler()

# Fit the scaler on TRAINING data only (to prevent data leakage)
scaler.fit(X_train)

# Transform both training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("Features standardized!")
print("All features now have mean ≈ 0 and standard deviation ≈ 1")
print("\nTraining set statistics after scaling:")
print(f"  Mean: {X_train_scaled.mean(axis=0).round(3)}")
print(f"  Std:  {X_train_scaled.std(axis=0).round(3)}")

# ============================================================================
# STEP 8: Visualize the Data
# ============================================================================

# Let's visualize the relationship between some features and the target

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Feature vs Target Relationships', fontsize=16, fontweight='bold')

for i in range(n_features):
    ax = axes[i // 4, i % 4]
    
    # Scatter plot of feature vs target
    ax.scatter(X_train_scaled[:, i], y_train, alpha=0.5, s=20)
    
    # Add title showing true coefficient
    true_coef = true_coefficients[i]
    importance = "IMPORTANT" if true_coef != 0 else "NOISE"
    ax.set_title(f'Feature {i} (True coef={true_coef:.1f})\n{importance}', 
                 fontsize=10, fontweight='bold' if true_coef != 0 else 'normal')
    ax.set_xlabel(f'Feature {i} (standardized)')
    ax.set_ylabel('Target')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_feature_relationships.png', dpi=100, bbox_inches='tight')
print("\n" + "=" * 60)
print("Visualization saved as '01_feature_relationships.png'")
print("\nLook at the plots:")
print("  - Features 0, 2, 4 should show clear relationships (they're important)")
print("  - Features 1, 3, 5, 6, 7 should look random (they're noise)")

# ============================================================================
# STEP 9: Save the Data for Use in Other Scripts
# ============================================================================

# Save as numpy arrays for easy loading in subsequent scripts
np.savez('data_for_regularization.npz',
         X_train=X_train_scaled,
         X_test=X_test_scaled,
         y_train=y_train,
         y_test=y_test,
         feature_names=feature_names,
         true_coefficients=true_coefficients)

print("\n" + "=" * 60)
print("Data saved to 'data_for_regularization.npz'")
print("This file will be used by the other scripts in this activity.")

# ============================================================================
# YOUR TURN: Experiment!
# ============================================================================

print("\n" + "=" * 60)
print("EXPERIMENTS TO TRY:")
print("=" * 60)
print("\n1. Change noise_level (line 118) to see how noise affects the data")
print("   - Try noise_level = 0.5 (less noise, clearer relationships)")
print("   - Try noise_level = 5.0 (more noise, harder to find relationships)")
print("\n2. Change the true_coefficients (lines 108-117)")
print("   - Make different features important")
print("   - See if regularization still identifies them correctly")
print("\n3. Add more correlation between features")
print("   - Modify the correlation section (lines 80-87)")
print("   - See how this affects model performance")
print("\n" + "=" * 60)

# ============================================================================
# QUESTIONS TO DISCUSS:
# ============================================================================

"""
DISCUSSION QUESTIONS:

1. Why do we standardize features before applying regularization?
   What would happen if we didn't?

2. We created 3 important features and 5 noise features. 
   Can you tell which is which just by looking at the scatter plots?

3. What problems might arise from having correlated features in our data?

4. Why do we split data into training and testing sets?
   What would happen if we used all data for training?

NEXT STEPS:
Run this script, examine the outputs and visualizations, then move on to
02_standard_regression.py to fit a baseline model!
"""
