"""
Topic 3 In-Class Activity - Part 2: Standard Linear Regression Baseline
========================================================================

PURPOSE:
--------
Before we explore regularization, we need a BASELINE. This script fits a standard
(ordinary least squares) linear regression model WITHOUT any regularization.

WHY START WITH NO REGULARIZATION?
----------------------------------
This baseline shows us what happens when we don't penalize model complexity:
- Do we overfit the training data?
- Do we pick up noise features as if they were important?
- How well does the model generalize to test data?

We'll compare ridge and lasso regression against this baseline to see if
regularization improves our model.
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("STANDARD LINEAR REGRESSION (NO REGULARIZATION)")
print("=" * 70)

# ============================================================================
# STEP 2: Load the Data We Created in Script 01
# ============================================================================

# Load the data we saved from the previous script
data = np.load('data_for_regularization.npz')

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
true_coefficients = data['true_coefficients']

print("\nData loaded successfully!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# STEP 3: Fit Standard Linear Regression
# ============================================================================

# Create the model
# LinearRegression uses Ordinary Least Squares (OLS)
# It minimizes: sum((y_true - y_predicted)^2)
# With NO penalty on coefficient size!

print("\n" + "=" * 70)
print("FITTING STANDARD LINEAR REGRESSION")
print("=" * 70)

# Create and fit the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\nModel fitted!")

# ============================================================================
# STEP 4: Examine the Learned Coefficients
# ============================================================================

# Extract the coefficients (slopes) learned by the model
learned_coefficients = lr_model.coef_
intercept = lr_model.intercept_

print("\n" + "=" * 70)
print("LEARNED COEFFICIENTS VS. TRUE COEFFICIENTS")
print("=" * 70)

# Create a comparison table
print("\n{:<12} {:>12} {:>12} {:>12}".format(
    "Feature", "True Coef", "Learned Coef", "Error"
))
print("-" * 50)

for i in range(len(feature_names)):
    true_coef = true_coefficients[i]
    learned_coef = learned_coefficients[i]
    error = abs(learned_coef - true_coef)
    
    print("{:<12} {:>12.3f} {:>12.3f} {:>12.3f}".format(
        feature_names[i], true_coef, learned_coef, error
    ))

print(f"\nIntercept: {intercept:.3f}")

# KEY QUESTIONS TO CONSIDER:
# 1. Did the model recover the true coefficients accurately?
# 2. Did it assign non-zero coefficients to the noise features (1, 3, 5, 6, 7)?
# 3. Why might the learned coefficients differ from the true ones?

# ============================================================================
# STEP 5: Visualize Coefficient Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(feature_names))
width = 0.35

# Plot true coefficients
bars1 = ax.bar(x_pos - width/2, true_coefficients, width, 
               label='True Coefficients', alpha=0.8, color='green')

# Plot learned coefficients
bars2 = ax.bar(x_pos + width/2, learned_coefficients, width,
               label='Learned Coefficients', alpha=0.8, color='blue')

# Customize the plot
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Standard Linear Regression: True vs. Learned Coefficients', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in range(len(feature_names))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('02_standard_regression_coefficients.png', dpi=100, bbox_inches='tight')
print("\nCoefficient comparison saved as '02_standard_regression_coefficients.png'")

# OBSERVATION QUESTIONS:
# - Do the learned coefficients match the true ones?
# - Are there non-zero coefficients for features that should be zero (noise)?
# - This is a sign of OVERFITTING - the model is fitting to noise!

# ============================================================================
# STEP 6: Evaluate Model Performance
# ============================================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE EVALUATION")
print("=" * 70)

# Make predictions on both training and testing data
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Calculate performance metrics

# R-squared (R²): Proportion of variance explained
# - R² = 1.0 means perfect predictions
# - R² = 0.0 means model is no better than predicting the mean
# - R² can be negative if model is worse than predicting the mean
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Root Mean Squared Error (RMSE): Average prediction error
# - Lower is better
# - Same units as target variable
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Display results
print("\nTRAINING SET PERFORMANCE:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE:     {train_rmse:.4f}")

print("\nTESTING SET PERFORMANCE:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE:     {test_rmse:.4f}")

# Compare training vs. testing performance
print("\nTRAINING vs. TESTING COMPARISON:")
print(f"  R² difference:   {train_r2 - test_r2:.4f}")
print(f"  RMSE difference: {test_rmse - train_rmse:.4f}")

# INTERPRETATION:
# - If training performance >> testing performance → OVERFITTING
# - If both are similar → Good generalization
# - If both are poor → UNDERFITTING

if train_r2 - test_r2 > 0.1:
    print("\n⚠️  WARNING: Significant overfitting detected!")
    print("    The model performs much better on training than testing data.")
elif train_r2 - test_r2 < 0.05:
    print("\n✓ Good generalization! Training and testing performance are similar.")

# ============================================================================
# STEP 7: Visualize Predictions vs. Actual Values
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training set predictions
ax1.scatter(y_train, y_train_pred, alpha=0.5, s=30)
ax1.plot([y_train.min(), y_train.max()], 
         [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
ax1.set_title(f'Training Set\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f}', 
              fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Testing set predictions
ax2.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='orange')
ax2.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
ax2.set_title(f'Testing Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_standard_regression_predictions.png', dpi=100, bbox_inches='tight')
print("\nPrediction plots saved as '02_standard_regression_predictions.png'")

# WHAT TO LOOK FOR:
# - Points close to the red line → Good predictions
# - Points far from the red line → Poor predictions
# - Training plot better than testing plot → Overfitting

# ============================================================================
# STEP 8: Analyze Residuals (Prediction Errors)
# ============================================================================

print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

# Calculate residuals (errors)
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Plot residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot for training
ax1.scatter(y_train_pred, train_residuals, alpha=0.5, s=30)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
ax1.set_title('Training Set Residuals', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Residual plot for testing
ax2.scatter(y_test_pred, test_residuals, alpha=0.5, s=30, color='orange')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
ax2.set_title('Testing Set Residuals', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_standard_regression_residuals.png', dpi=100, bbox_inches='tight')
print("\nResidual plots saved as '02_standard_regression_residuals.png'")

# WHAT GOOD RESIDUALS LOOK LIKE:
# - Randomly scattered around zero
# - No clear patterns
# - Similar spread across all predicted values (homoscedasticity)

# ============================================================================
# STEP 9: Save Results for Comparison with Regularized Models
# ============================================================================

# Save the results for later comparison
results = {
    'model_name': 'Standard Linear Regression',
    'coefficients': learned_coefficients,
    'intercept': intercept,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_predictions': y_train_pred,
    'test_predictions': y_test_pred
}

np.savez('02_standard_regression_results.npz', **results)
print("\nResults saved to '02_standard_regression_results.npz'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF STANDARD LINEAR REGRESSION")
print("=" * 70)

print("\nKEY OBSERVATIONS:")
print(f"  1. The model assigned coefficients to ALL features")
print(f"     - Including noise features that should be zero!")

# Count how many noise features got non-trivial coefficients
noise_indices = [i for i, c in enumerate(true_coefficients) if c == 0]
noise_coefs = [abs(learned_coefficients[i]) for i in noise_indices]
significant_noise = sum(1 for c in noise_coefs if c > 0.1)

print(f"  2. Number of noise features with |coefficient| > 0.1: {significant_noise}/5")

print(f"\n  3. Model Performance:")
print(f"     - Training R²: {train_r2:.4f}")
print(f"     - Testing R²:  {test_r2:.4f}")
print(f"     - Difference:  {train_r2 - test_r2:.4f}")

if train_r2 - test_r2 > 0.05:
    print("\n  ⚠️  The model may be overfitting to the training data!")
    print("     It's fitting to noise instead of just the signal.")

print("\n" + "=" * 70)
print("WHY REGULARIZATION MIGHT HELP:")
print("=" * 70)
print("\nProblems we observed:")
print("  ✗ Model assigns non-zero coefficients to noise features")
print("  ✗ May be overfitting to training data")
print("  ✗ Model complexity not penalized")
print("\nWhat regularization can do:")
print("  ✓ Shrink or eliminate coefficients for unimportant features")
print("  ✓ Reduce overfitting")
print("  ✓ Improve generalization to new data")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\nNow let's try RIDGE REGRESSION (script 03) to see if penalizing")
print("coefficient size helps us recover the true model better!")

# ============================================================================
# YOUR TURN: Experiment!
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS TO TRY:")
print("=" * 70)
print("\n1. Go back to script 01 and increase the noise level")
print("   - Then rerun this script")
print("   - Does performance get worse? By how much?")
print("\n2. Look at the coefficient comparison plot")
print("   - Which features did the model get most wrong?")
print("   - Are these the noise features or the important ones?")
print("\n3. Compare training vs. testing RMSE")
print("   - Is there evidence of overfitting?")
print("   - Would regularization help?")

# ============================================================================
# DISCUSSION QUESTIONS:
# ============================================================================

"""
DISCUSSION QUESTIONS:

1. Why does standard linear regression assign non-zero coefficients 
   to features that are just noise?

2. What is overfitting, and did we observe it here?
   - Look at training vs. testing performance
   - Look at coefficients assigned to noise features

3. If you had to present this model to a business stakeholder,
   would you be confident in it? Why or why not?

4. Based on these results, what do you expect regularization to do?
   Make specific predictions before moving to the next script!

READY FOR NEXT STEP:
Move on to 03_ridge_regression.py to see how L2 regularization
can help address these issues!
"""
