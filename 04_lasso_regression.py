"""
Topic 3 In-Class Activity - Part 4: Lasso Regression (L1 Regularization)
=========================================================================

PURPOSE:
--------
Lasso regression adds an L1 penalty to the regression objective.
It minimizes: sum((y_true - y_pred)^2) + λ * sum(|coefficients|)

The L1 penalty has a special property: it can drive coefficients to EXACTLY ZERO,
effectively performing automatic feature selection!

KEY CONCEPTS:
-------------
1. L1 Penalty: Penalizes the sum of ABSOLUTE VALUES of coefficients
2. Sparsity: Many coefficients become exactly zero
3. Feature Selection: Automatically identifies and keeps only important features
4. When to Use: High-dimensional data, or when interpretability is crucial

RIDGE vs. LASSO:
---------------
Ridge (L2): sum(coefficients²)  → Shrinks but never eliminates
Lasso (L1): sum(|coefficients|) → Can set coefficients to EXACTLY zero

LEARNING GOALS:
--------------
- See how Lasso performs feature selection
- Understand the role of λ in Lasso
- Compare Lasso to Ridge and standard regression
- Learn when to choose Lasso vs. Ridge
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("LASSO REGRESSION (L1 REGULARIZATION)")
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

# Load previous results for comparison
baseline = np.load('02_standard_regression_results.npz')
ridge_results = np.load('03_ridge_regression_results.npz')

baseline_coefs = baseline['coefficients']
ridge_coefs = ridge_results['coefficients']

# ============================================================================
# STEP 3: Lasso Regression with Different Lambda Values
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT: EFFECT OF DIFFERENT λ (ALPHA) VALUES")
print("=" * 70)

# Test different values of lambda (called 'alpha' in scikit-learn)
# Note: Lasso often needs smaller alphas than Ridge to avoid over-regularization

alphas_to_test = [0.001, 0.01, 0.1, 1.0, 10.0]

print("\nTesting λ (alpha) values:", alphas_to_test)
print("\nRemember: Lasso can set coefficients to EXACTLY ZERO!")

# Store results for each alpha
results_by_alpha = []

for alpha in alphas_to_test:
    # Create and fit lasso regression model
    # max_iter increased to ensure convergence
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = lasso_model.predict(X_train)
    y_test_pred = lasso_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Count how many coefficients are exactly zero
    n_zero_coefs = np.sum(lasso_model.coef_ == 0)
    n_nonzero_coefs = np.sum(lasso_model.coef_ != 0)
    
    # Store results
    results_by_alpha.append({
        'alpha': alpha,
        'coefficients': lasso_model.coef_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'n_zero': n_zero_coefs,
        'n_nonzero': n_nonzero_coefs
    })
    
    print(f"\nλ (alpha) = {alpha:6.3f}:")
    print(f"  Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}")
    print(f"  Features: {n_nonzero_coefs} non-zero, {n_zero_coefs} eliminated")

# KEY OBSERVATION:
# As λ increases:
# - More coefficients become exactly zero
# - The model becomes simpler (fewer features)
# - This is automatic FEATURE SELECTION!

# ============================================================================
# STEP 4: Visualize Coefficient Paths and Feature Elimination
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZING LASSO FEATURE SELECTION")
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

ax.set_xscale('log')
ax.set_xlabel('λ (alpha) - Regularization Strength', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Lasso Regression: Coefficient Path and Feature Elimination', 
             fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_lasso_coefficient_paths.png', dpi=100, bbox_inches='tight')
print("\nCoefficient paths saved as '04_lasso_coefficient_paths.png'")

# CRITICAL OBSERVATION:
# Notice how coefficients hit EXACTLY zero and stay there!
# This is the key difference from Ridge, which only shrinks toward zero.

# ============================================================================
# STEP 5: Analyze Feature Selection Pattern
# ============================================================================

# Create a heatmap showing which features are selected at each alpha
fig, ax = plt.subplots(figsize=(10, 6))

# Create matrix: rows = alphas, columns = features
# Value = 1 if feature is selected (non-zero), 0 if eliminated (zero)
selection_matrix = np.zeros((len(alphas_to_test), len(feature_names)))

for i, result in enumerate(results_by_alpha):
    selection_matrix[i, :] = (result['coefficients'] != 0).astype(int)

# Create heatmap
im = ax.imshow(selection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(feature_names)))
ax.set_yticks(np.arange(len(alphas_to_test)))
ax.set_xticklabels([f'F{i}' for i in range(len(feature_names))])
ax.set_yticklabels([f'{a:.3f}' for a in alphas_to_test])

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('λ (alpha)', fontsize=12, fontweight='bold')
ax.set_title('Lasso Feature Selection Pattern\n(Green = Selected, Red = Eliminated)', 
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Selected', rotation=270, labelpad=15)

# Add text annotations
for i in range(len(alphas_to_test)):
    for j in range(len(feature_names)):
        text = ax.text(j, i, int(selection_matrix[i, j]),
                      ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
plt.savefig('04_lasso_feature_selection_heatmap.png', dpi=100, bbox_inches='tight')
print("Feature selection heatmap saved as '04_lasso_feature_selection_heatmap.png'")

# Analyze which features are eliminated first
print("\n" + "=" * 70)
print("FEATURE ELIMINATION ORDER")
print("=" * 70)

print("\nAs λ increases, features are eliminated in this order:")
for i, result in enumerate(results_by_alpha):
    alpha = result['alpha']
    zero_features = [f'F{j}' for j in range(len(feature_names)) 
                     if result['coefficients'][j] == 0]
    
    if zero_features:
        print(f"λ = {alpha:6.3f}: Eliminated {zero_features}")
    else:
        print(f"λ = {alpha:6.3f}: All features retained")

# Check if noise features are eliminated before important ones
print("\nDoes Lasso correctly identify noise features?")
for i, result in enumerate(results_by_alpha):
    alpha = result['alpha']
    noise_indices = [j for j in range(len(true_coefficients)) if true_coefficients[j] == 0]
    important_indices = [j for j in range(len(true_coefficients)) if true_coefficients[j] != 0]
    
    noise_eliminated = sum(1 for j in noise_indices if result['coefficients'][j] == 0)
    important_eliminated = sum(1 for j in important_indices if result['coefficients'][j] == 0)
    
    print(f"λ = {alpha:6.3f}: Noise eliminated: {noise_eliminated}/5, "
          f"Important eliminated: {important_eliminated}/3")

# ============================================================================
# STEP 6: Use Cross-Validation to Find Optimal Lambda
# ============================================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL λ USING CROSS-VALIDATION")
print("=" * 70)

# LassoCV automatically performs cross-validation
alphas_cv = np.logspace(-4, 1, 100)  # 100 values from 0.0001 to 10

print(f"\nTesting {len(alphas_cv)} different λ values using 5-fold cross-validation...")

lasso_cv = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)

optimal_alpha = lasso_cv.alpha_

print(f"\nOptimal λ (alpha) found: {optimal_alpha:.6f}")

# Fit final model with optimal alpha
best_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

# Evaluate performance
y_train_pred = best_lasso.predict(X_train)
y_test_pred = best_lasso.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Count selected features
n_zero = np.sum(best_lasso.coef_ == 0)
n_selected = np.sum(best_lasso.coef_ != 0)

print(f"\nPerformance with optimal λ:")
print(f"  Training R²:  {train_r2:.4f}")
print(f"  Testing R²:   {test_r2:.4f}")
print(f"  Training RMSE: {train_rmse:.4f}")
print(f"  Testing RMSE:  {test_rmse:.4f}")
print(f"  Features selected: {n_selected}/{len(feature_names)}")
print(f"  Features eliminated: {n_zero}/{len(feature_names)}")

# Show which features were selected
selected_features = [feature_names[i] for i in range(len(feature_names)) 
                    if best_lasso.coef_[i] != 0]
eliminated_features = [feature_names[i] for i in range(len(feature_names)) 
                      if best_lasso.coef_[i] == 0]

print(f"\nSelected features: {selected_features}")
print(f"Eliminated features: {eliminated_features}")

# ============================================================================
# STEP 7: Three-Way Comparison: Standard vs. Ridge vs. Lasso
# ============================================================================

print("\n" + "=" * 70)
print("THREE-WAY COMPARISON: STANDARD vs. RIDGE vs. LASSO")
print("=" * 70)

# Create comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Coefficient comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(feature_names))
width = 0.2

bars1 = ax1.bar(x_pos - 1.5*width, true_coefficients, width, 
                label='True', alpha=0.9, color='green')
bars2 = ax1.bar(x_pos - 0.5*width, baseline_coefs, width,
                label='Standard', alpha=0.9, color='blue')
bars3 = ax1.bar(x_pos + 0.5*width, ridge_coefs, width,
                label='Ridge', alpha=0.9, color='orange')
bars4 = ax1.bar(x_pos + 1.5*width, best_lasso.coef_, width,
                label='Lasso', alpha=0.9, color='red')

ax1.set_xlabel('Features', fontweight='bold')
ax1.set_ylabel('Coefficient Value', fontweight='bold')
ax1.set_title('Coefficient Comparison Across Models', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'F{i}' for i in range(len(feature_names))])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 2: Number of features used
ax2 = axes[0, 1]
models = ['Standard', 'Ridge', 'Lasso']
n_features_used = [
    len(feature_names),  # Standard uses all
    len(feature_names),  # Ridge uses all (none exactly zero)
    n_selected           # Lasso eliminates some
]
colors_bar = ['blue', 'orange', 'red']

bars = ax2.bar(models, n_features_used, color=colors_bar, alpha=0.7)
ax2.set_ylabel('Number of Features Used', fontweight='bold')
ax2.set_title('Model Complexity (Features Used)', fontweight='bold')
ax2.set_ylim([0, len(feature_names) + 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: Performance comparison
ax3 = axes[1, 0]
models = ['Standard', 'Ridge', 'Lasso']
test_r2_values = [
    baseline['test_r2'],
    ridge_results['test_r2'],
    test_r2
]
test_rmse_values = [
    baseline['test_rmse'],
    ridge_results['test_rmse'],
    test_rmse
]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, test_r2_values, width,
                label='Test R²', alpha=0.8, color='darkgreen')
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x_pos + width/2, test_rmse_values, width,
                     label='Test RMSE', alpha=0.8, color='darkred')

ax3.set_xlabel('Model', fontweight='bold')
ax3.set_ylabel('Test R² (higher is better)', fontweight='bold', color='darkgreen')
ax3_twin.set_ylabel('Test RMSE (lower is better)', fontweight='bold', color='darkred')
ax3.set_title('Test Set Performance Comparison', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models)

# Add legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Plot 4: Coefficient error from truth
ax4 = axes[1, 1]
standard_errors = np.abs(baseline_coefs - true_coefficients)
ridge_errors = np.abs(ridge_coefs - true_coefficients)
lasso_errors = np.abs(best_lasso.coef_ - true_coefficients)

x_pos = np.arange(len(feature_names))
width = 0.25

bars1 = ax4.bar(x_pos - width, standard_errors, width,
                label='Standard', alpha=0.8, color='blue')
bars2 = ax4.bar(x_pos, ridge_errors, width,
                label='Ridge', alpha=0.8, color='orange')
bars3 = ax4.bar(x_pos + width, lasso_errors, width,
                label='Lasso', alpha=0.8, color='red')

ax4.set_xlabel('Features', fontweight='bold')
ax4.set_ylabel('Absolute Error from True Coefficient', fontweight='bold')
ax4.set_title('Coefficient Recovery Accuracy', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'F{i}' for i in range(len(feature_names))])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('04_three_way_comparison.png', dpi=100, bbox_inches='tight')
print("\nThree-way comparison saved as '04_three_way_comparison.png'")

# ============================================================================
# STEP 8: Detailed Performance Comparison Table
# ============================================================================

print("\n" + "=" * 70)
print("DETAILED PERFORMANCE COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Metric': ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 
               'Features Used', 'Mean |Coef Error|'],
    'Standard': [
        baseline['train_r2'],
        baseline['test_r2'],
        baseline['train_rmse'],
        baseline['test_rmse'],
        len(feature_names),
        np.mean(np.abs(baseline_coefs - true_coefficients))
    ],
    'Ridge': [
        ridge_results['train_r2'],
        ridge_results['test_r2'],
        ridge_results['train_rmse'],
        ridge_results['test_rmse'],
        len(feature_names),
        np.mean(np.abs(ridge_coefs - true_coefficients))
    ],
    'Lasso': [
        train_r2,
        test_r2,
        train_rmse,
        test_rmse,
        n_selected,
        np.mean(np.abs(best_lasso.coef_ - true_coefficients))
    ]
})

print("\n", comparison_df.to_string(index=False))

# Determine best model
best_test_r2_model = ['Standard', 'Ridge', 'Lasso'][np.argmax([
    baseline['test_r2'], ridge_results['test_r2'], test_r2
])]

print(f"\nBest test R²: {best_test_r2_model}")

# ============================================================================
# STEP 9: Save Lasso Results
# ============================================================================

results = {
    'model_name': 'Lasso Regression',
    'optimal_alpha': optimal_alpha,
    'coefficients': best_lasso.coef_,
    'intercept': best_lasso.intercept_,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'n_features_selected': n_selected,
    'train_predictions': y_train_pred,
    'test_predictions': y_test_pred
}

np.savez('04_lasso_regression_results.npz', **results)
print("\nResults saved to '04_lasso_regression_results.npz'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHAT WE LEARNED ABOUT LASSO REGRESSION")
print("=" * 70)

print("\n1. HOW LASSO WORKS:")
print("   - Adds L1 penalty: λ * sum(|coefficients|)")
print("   - Can set coefficients EXACTLY to zero")
print("   - Performs automatic FEATURE SELECTION")

print("\n2. FEATURE SELECTION:")
print(f"   - Started with {len(feature_names)} features")
print(f"   - Lasso selected {n_selected} features")
print(f"   - Eliminated {n_zero} features")
print(f"   - True important features: 3 (Features 0, 2, 4)")

# Check if Lasso selected the right features
noise_features = [i for i, c in enumerate(true_coefficients) if c == 0]
important_features = [i for i, c in enumerate(true_coefficients) if c != 0]

noise_eliminated = sum(1 for i in noise_features if best_lasso.coef_[i] == 0)
important_kept = sum(1 for i in important_features if best_lasso.coef_[i] != 0)

print(f"\n3. FEATURE SELECTION ACCURACY:")
print(f"   - Noise features correctly eliminated: {noise_eliminated}/5")
print(f"   - Important features correctly kept: {important_kept}/3")

print("\n4. LASSO vs. RIDGE:")
print("   Ridge: Shrinks all coefficients, keeps all features")
print("   Lasso: Eliminates features, creates sparse models")

print("\n5. WHEN TO USE LASSO:")
print("   ✓ High-dimensional data (many features)")
print("   ✓ When interpretability is important")
print("   ✓ When you believe many features are irrelevant")
print("   ✓ For automatic feature selection")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\nNow that we've seen Standard, Ridge, and Lasso individually,")
print("let's do a comprehensive side-by-side comparison!")
print("\nMove on to 05_model_comparison.py for the final analysis.")

# ============================================================================
# YOUR TURN: Experiment!
# ============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS TO TRY:")
print("=" * 70)
print("\n1. Change the alpha range in LassoCV (line 304)")
print("   - Try np.logspace(-5, 2, 100)")
print("   - Does it select different features?")
print("\n2. Look at the feature selection heatmap")
print("   - At what λ do noise features get eliminated?")
print("   - Are important features retained longer?")
print("\n3. Compare to Ridge results")
print("   - Which model has better test R²?")
print("   - Which is more interpretable?")
print("   - Which would you recommend and why?")

# ============================================================================
# DISCUSSION QUESTIONS:
# ============================================================================

"""
DISCUSSION QUESTIONS:

1. Why can Lasso set coefficients to exactly zero, but Ridge cannot?
   - Think about the shape of the L1 vs. L2 penalty
   - Draw the constraint regions if helpful

2. Did Lasso correctly identify the important features?
   - Compare selected features to true_coefficients
   - What does this tell you about Lasso's effectiveness?

3. When would you choose Lasso over Ridge?
   - Consider interpretability
   - Consider feature selection needs
   - Consider computational cost

4. What happens if λ is too large in Lasso?
   - It eliminates too many features (underfitting)
   - That's why cross-validation is crucial!

FINAL STEP:
Run 05_model_comparison.py for a comprehensive comparison
and guidance on how to choose between these approaches!
"""
