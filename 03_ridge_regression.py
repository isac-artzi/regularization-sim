"""
Topic 3 In-Class Activity - Part 3: Ridge Regression (L2 Regularization)
=========================================================================

PURPOSE:
--------
Ridge regression adds an L2 penalty to the standard regression objective.
It minimizes: sum((y_true - y_pred)^2) + λ * sum(coefficients^2)

The penalty term (λ * sum(coefficients^2)) discourages large coefficients,
which helps reduce overfitting and makes the model more robust.

KEY CONCEPTS:
-------------
1. L2 Penalty: Penalizes the sum of SQUARED coefficients
2. Shrinkage: Coefficients get smaller (shrink toward zero) as λ increases
3. No Feature Selection: Ridge shrinks ALL coefficients but never sets them exactly to zero
4. Handles Multicollinearity: Works well when features are correlated

LEARNING GOALS:
--------------
- Understand how λ (alpha) controls the strength of regularization
- See how ridge regression shrinks coefficients
- Learn to use cross-validation to select optimal λ
- Compare ridge regression to standard regression
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("RIDGE REGRESSION (L2 REGULARIZATION)")
print("=" * 70)

# ============================================================================
# STEP 2: Load the Data
# ============================================================================

data = np.load('data_for_regularization.npz')

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
true_coefficients = data['true_coefficients']

print("\nData loaded successfully!")

# Also load standard regression results for comparison
baseline = np.load('02_standard_regression_results.npz')
baseline_coefs = baseline['coefficients']

# ============================================================================
# STEP 3: Ridge Regression with Different Lambda Values
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT: EFFECT OF DIFFERENT λ (ALPHA) VALUES")
print("=" * 70)

# Test different values of lambda (called 'alpha' in scikit-learn)
# Lambda controls the strength of the penalty:
# - λ = 0: No penalty (equivalent to standard regression)
# - λ small: Weak penalty (small shrinkage)
# - λ large: Strong penalty (large shrinkage)

alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0]

print("\nTesting λ (alpha) values:", alphas_to_test)
print("\nRemember: Larger λ → Stronger penalty → More shrinkage")

# Store results for each alpha
results_by_alpha = []

for alpha in alphas_to_test:
    # Create and fit ridge regression model
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Store results
    results_by_alpha.append({
        'alpha': alpha,
        'coefficients': ridge_model.coef_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })
    
    print(f"\nλ (alpha) = {alpha:6.2f}:")
    print(f"  Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}")

# KEY OBSERVATION TO MAKE:
# As λ increases:
# - What happens to training performance?
# - What happens to testing performance?
# - Is there an optimal λ where test performance is best?

# ============================================================================
# STEP 4: Visualize Coefficient Shrinkage
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZING COEFFICIENT SHRINKAGE")
print("=" * 70)

# Create a plot showing how coefficients change with different λ values
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each feature's coefficient path
for feature_idx in range(len(feature_names)):
    # Extract coefficients for this feature across all alphas
    coef_path = [r['coefficients'][feature_idx] for r in results_by_alpha]
    
    # Determine if this is an important feature or noise
    is_important = true_coefficients[feature_idx] != 0
    
    # Plot with different styles for important vs. noise features
    if is_important:
        ax.plot(alphas_to_test, coef_path, 
                marker='o', linewidth=2.5, 
                label=f'{feature_names[feature_idx]} (Important)', 
                linestyle='-')
    else:
        ax.plot(alphas_to_test, coef_path, 
                marker='x', linewidth=1, alpha=0.6,
                label=f'{feature_names[feature_idx]} (Noise)', 
                linestyle='--')

ax.set_xscale('log')  # Logarithmic scale for better visualization
ax.set_xlabel('λ (alpha) - Regularization Strength', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Ridge Regression: Coefficient Shrinkage Path', 
             fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_ridge_coefficient_paths.png', dpi=100, bbox_inches='tight')
print("\nCoefficient paths saved as '03_ridge_coefficient_paths.png'")

# WHAT TO OBSERVE:
# - As λ increases, ALL coefficients shrink toward zero
# - BUT coefficients never reach exactly zero (unlike Lasso!)
# - Important features maintain larger coefficients even with high λ
# - Noise features shrink faster

# ============================================================================
# STEP 5: Compare Performance Across Different Lambda Values
# ============================================================================

# Plot how model performance changes with λ

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Extract metrics
train_r2_values = [r['train_r2'] for r in results_by_alpha]
test_r2_values = [r['test_r2'] for r in results_by_alpha]
train_rmse_values = [r['train_rmse'] for r in results_by_alpha]
test_rmse_values = [r['test_rmse'] for r in results_by_alpha]

# Plot R² scores
ax1.plot(alphas_to_test, train_r2_values, marker='o', 
         linewidth=2, label='Training R²', color='blue')
ax1.plot(alphas_to_test, test_r2_values, marker='s', 
         linewidth=2, label='Testing R²', color='orange')
ax1.set_xscale('log')
ax1.set_xlabel('λ (alpha)', fontsize=11, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance vs. Regularization Strength', 
              fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot RMSE
ax2.plot(alphas_to_test, train_rmse_values, marker='o', 
         linewidth=2, label='Training RMSE', color='blue')
ax2.plot(alphas_to_test, test_rmse_values, marker='s', 
         linewidth=2, label='Testing RMSE', color='orange')
ax2.set_xscale('log')
ax2.set_xlabel('λ (alpha)', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Error vs. Regularization Strength', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_ridge_performance_vs_lambda.png', dpi=100, bbox_inches='tight')
print("Performance plots saved as '03_ridge_performance_vs_lambda.png'")

# INTERPRETATION:
# - Training performance typically decreases as λ increases (we're constraining the model)
# - Testing performance often improves initially, then decreases
# - The optimal λ balances underfitting and overfitting

# ============================================================================
# STEP 6: Use Cross-Validation to Find Optimal Lambda
# ============================================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL λ USING CROSS-VALIDATION")
print("=" * 70)

# Instead of testing a few values manually, let's use cross-validation
# to systematically find the best λ

# RidgeCV automatically performs cross-validation
# It tries many alpha values and picks the best one

alphas_cv = np.logspace(-3, 3, 100)  # 100 values from 0.001 to 1000

print(f"\nTesting {len(alphas_cv)} different λ values using 5-fold cross-validation...")

ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)  # 5-fold cross-validation
ridge_cv.fit(X_train, y_train)

optimal_alpha = ridge_cv.alpha_

print(f"\nOptimal λ (alpha) found: {optimal_alpha:.4f}")

# Fit final model with optimal alpha
best_ridge = Ridge(alpha=optimal_alpha)
best_ridge.fit(X_train, y_train)

# Evaluate performance
y_train_pred = best_ridge.predict(X_train)
y_test_pred = best_ridge.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nPerformance with optimal λ:")
print(f"  Training R²:  {train_r2:.4f}")
print(f"  Testing R²:   {test_r2:.4f}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Testing RMSE:  {test_rmse:.4f}")

# ============================================================================
# STEP 7: Compare Ridge vs. Standard Regression
# ============================================================================

print("\n" + "=" * 70)
print("RIDGE vs. STANDARD REGRESSION COMPARISON")
print("=" * 70)

# Create comparison visualization
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(feature_names))
width = 0.25

# Plot true coefficients
bars1 = ax.bar(x_pos - width, true_coefficients, width, 
               label='True Coefficients', alpha=0.8, color='green')

# Plot standard regression coefficients
bars2 = ax.bar(x_pos, baseline_coefs, width,
               label='Standard Regression', alpha=0.8, color='blue')

# Plot ridge coefficients
bars3 = ax.bar(x_pos + width, best_ridge.coef_, width,
               label=f'Ridge (λ={optimal_alpha:.3f})', alpha=0.8, color='red')

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Coefficient Comparison: True vs. Standard vs. Ridge', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in range(len(feature_names))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('03_ridge_vs_standard_coefficients.png', dpi=100, bbox_inches='tight')
print("\nComparison plot saved as '03_ridge_vs_standard_coefficients.png'")

# Quantitative comparison
print("\nCoefficient Comparison Table:")
print("{:<12} {:>12} {:>12} {:>12}".format(
    "Feature", "True", "Standard", "Ridge"
))
print("-" * 50)

for i in range(len(feature_names)):
    print("{:<12} {:>12.3f} {:>12.3f} {:>12.3f}".format(
        feature_names[i], 
        true_coefficients[i],
        baseline_coefs[i],
        best_ridge.coef_[i]
    ))

# Calculate how close each model is to truth
standard_error = np.mean(np.abs(baseline_coefs - true_coefficients))
ridge_error = np.mean(np.abs(best_ridge.coef_ - true_coefficients))

print(f"\nMean Absolute Error from True Coefficients:")
print(f"  Standard Regression: {standard_error:.4f}")
print(f"  Ridge Regression:    {ridge_error:.4f}")

if ridge_error < standard_error:
    print(f"\n✓ Ridge is closer to true coefficients!")
    print(f"  Improvement: {((standard_error - ridge_error) / standard_error * 100):.1f}%")
else:
    print(f"\n⚠️  Standard regression is closer to true coefficients")

# ============================================================================
# STEP 8: Save Ridge Regression Results
# ============================================================================

results = {
    'model_name': 'Ridge Regression',
    'optimal_alpha': optimal_alpha,
    'coefficients': best_ridge.coef_,
    'intercept': best_ridge.intercept_,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_predictions': y_train_pred,
    'test_predictions': y_test_pred
}

np.savez('03_ridge_regression_results.npz', **results)
print("\nResults saved to '03_ridge_regression_results.npz'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHAT WE LEARNED ABOUT RIDGE REGRESSION")
print("=" * 70)

print("\n1. HOW RIDGE WORKS:")
print("   - Adds L2 penalty: λ * sum(coefficients²)")
print("   - Shrinks ALL coefficients toward zero")
print("   - NEVER sets coefficients exactly to zero")

print("\n2. EFFECT OF λ (ALPHA):")
print("   - λ = 0 → No penalty (same as standard regression)")
print("   - Small λ → Small shrinkage")
print("   - Large λ → Large shrinkage")
print(f"   - Optimal λ found: {optimal_alpha:.4f}")

print("\n3. ADVANTAGES OF RIDGE:")
print("   - Reduces overfitting")
print("   - Handles multicollinearity well")
print("   - Often improves generalization")

print("\n4. LIMITATIONS OF RIDGE:")
print("   - Keeps ALL features (doesn't do feature selection)")
print("   - Can be hard to interpret with many features")
print("   - Still includes noise features (with small coefficients)")

print("\n" + "=" * 70)
print("RIDGE vs. STANDARD REGRESSION:")
print("=" * 70)

# Load baseline for comparison
baseline = np.load('02_standard_regression_results.npz')

print(f"\nStandard Regression:")
print(f"  Test R²: {baseline['test_r2']:.4f}")
print(f"  Test RMSE: {baseline['test_rmse']:.4f}")

print(f"\nRidge Regression:")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")

if test_r2 > baseline['test_r2']:
    print(f"\n✓ Ridge improved test R² by {(test_r2 - baseline['test_r2']):.4f}!")
if test_rmse < baseline['test_rmse']:
    print(f"✓ Ridge reduced test RMSE by {(baseline['test_rmse'] - test_rmse):.4f}!")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\nRidge shrinks coefficients but keeps all features.")
print("What if we want to ELIMINATE unimportant features entirely?")
print("\nThat's what LASSO does! Move on to 04_lasso_regression.py")

# ============================================================================
# YOUR TURN: Experiment!
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS TO TRY:")
print("=" * 70)
print("\n1. Try different alpha ranges in RidgeCV (line 227)")
print("   - What if you test np.logspace(-5, 5, 100)?")
print("   - Does the optimal alpha change?")
print("\n2. Look at the coefficient shrinkage plot")
print("   - Do important features stay larger than noise features?")
print("   - At what λ do all coefficients become very small?")
print("\n3. Compare coefficient errors")
print("   - Is Ridge better at recovering true coefficients?")
print("   - Which features does it get most right/wrong?")

# ============================================================================
# DISCUSSION QUESTIONS:
# ============================================================================

"""
DISCUSSION QUESTIONS:

1. Why does Ridge regression shrink ALL coefficients but never to zero?
   - Think about the L2 penalty: sum(coefficients²)
   - What happens as a coefficient approaches zero?

2. When would Ridge be preferred over standard regression?
   - Consider overfitting, multicollinearity, and interpretability

3. What's the main limitation of Ridge for feature selection?
   - It keeps all features, even if some are pure noise
   - How could this be problematic in high-dimensional data?

4. Cross-validation found an optimal λ. Why is this better than 
   just trying a few values manually?

READY FOR LASSO:
Move on to 04_lasso_regression.py to see how L1 regularization
can actually eliminate features by setting coefficients to zero!
"""
