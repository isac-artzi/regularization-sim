"""
Topic 3 In-Class Activity - Part 5: Comprehensive Model Comparison
===================================================================

PURPOSE:
--------
Now that we've explored Standard Regression, Ridge, and Lasso individually,
let's bring it all together with a comprehensive comparison.

This script will help you understand:
- When to use each approach
- How to evaluate and compare models
- How to make informed modeling decisions
- How to present results clearly

DECISION FRAMEWORK:
------------------
After this analysis, you'll be able to answer:
1. Which model performs best? (Performance)
2. Which model is most interpretable? (Simplicity)
3. Which model best recovers the truth? (Accuracy)
4. Which model would you recommend? (Balance)
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

# ============================================================================
# STEP 2: Load All Model Results
# ============================================================================

print("\nLoading all model results...")

# Load the original data
data = np.load('data_for_regularization.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
true_coefficients = data['true_coefficients']

# Load model results
baseline = np.load('02_standard_regression_results.npz')
ridge_results = np.load('03_ridge_regression_results.npz')
lasso_results = np.load('04_lasso_regression_results.npz')

print("All results loaded successfully!")

# ============================================================================
# STEP 3: Create Comprehensive Comparison Table
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMANCE METRICS COMPARISON")
print("=" * 70)

# Create comprehensive comparison DataFrame
comparison_data = {
    'Model': ['Standard Regression', 'Ridge Regression', 'Lasso Regression'],
    'Regularization': ['None', 'L2 (sum of squares)', 'L1 (sum of absolutes)'],
    'Lambda (α)': [
        'N/A',
        f"{ridge_results['optimal_alpha']:.4f}",
        f"{lasso_results['optimal_alpha']:.6f}"
    ],
    'Train R²': [
        baseline['train_r2'],
        ridge_results['train_r2'],
        lasso_results['train_r2']
    ],
    'Test R²': [
        baseline['test_r2'],
        ridge_results['test_r2'],
        lasso_results['test_r2']
    ],
    'Train RMSE': [
        baseline['train_rmse'],
        ridge_results['train_rmse'],
        lasso_results['train_rmse']
    ],
    'Test RMSE': [
        baseline['test_rmse'],
        ridge_results['test_rmse'],
        lasso_results['test_rmse']
    ],
    'Features Used': [
        len(feature_names),
        len(feature_names),
        lasso_results['n_features_selected']
    ],
    'Overfit Gap': [
        baseline['train_r2'] - baseline['test_r2'],
        ridge_results['train_r2'] - ridge_results['test_r2'],
        lasso_results['train_r2'] - lasso_results['test_r2']
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n", comparison_df.to_string(index=False))

# Identify best performers
best_test_r2_idx = comparison_df['Test R²'].idxmax()
best_test_rmse_idx = comparison_df['Test RMSE'].idxmin()
least_overfit_idx = comparison_df['Overfit Gap'].idxmin()
simplest_idx = comparison_df['Features Used'].idxmin()

print("\n" + "=" * 70)
print("BEST PERFORMERS:")
print("=" * 70)
print(f"  Best Test R²:      {comparison_df.loc[best_test_r2_idx, 'Model']}")
print(f"  Lowest Test RMSE:  {comparison_df.loc[best_test_rmse_idx, 'Model']}")
print(f"  Least Overfitting: {comparison_df.loc[least_overfit_idx, 'Model']}")
print(f"  Simplest Model:    {comparison_df.loc[simplest_idx, 'Model']}")

# ============================================================================
# STEP 4: Detailed Coefficient Comparison
# ============================================================================

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON")
print("=" * 70)

# Create coefficient comparison table
coef_comparison = pd.DataFrame({
    'Feature': feature_names,
    'True': true_coefficients,
    'Standard': baseline['coefficients'],
    'Ridge': ridge_results['coefficients'],
    'Lasso': lasso_results['coefficients'],
})

# Add errors from true coefficients
coef_comparison['Std Error'] = np.abs(coef_comparison['Standard'] - coef_comparison['True'])
coef_comparison['Ridge Error'] = np.abs(coef_comparison['Ridge'] - coef_comparison['True'])
coef_comparison['Lasso Error'] = np.abs(coef_comparison['Lasso'] - coef_comparison['True'])

# Mark important features
coef_comparison['Important'] = ['Yes' if c != 0 else 'No' 
                                 for c in true_coefficients]

print("\n", coef_comparison.to_string(index=False))

# Calculate mean errors
print("\n" + "=" * 70)
print("COEFFICIENT RECOVERY ACCURACY")
print("=" * 70)

mean_errors = {
    'Standard Regression': coef_comparison['Std Error'].mean(),
    'Ridge Regression': coef_comparison['Ridge Error'].mean(),
    'Lasso Regression': coef_comparison['Lasso Error'].mean()
}

for model, error in mean_errors.items():
    print(f"  {model}: Mean Absolute Error = {error:.4f}")

best_recovery = min(mean_errors, key=mean_errors.get)
print(f"\n  Best coefficient recovery: {best_recovery}")

# ============================================================================
# STEP 5: Visualize Model Performance
# ============================================================================

print("\n" + "=" * 70)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 70)

# Create a comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color scheme for consistent model representation
model_colors = {
    'Standard': '#1f77b4',  # blue
    'Ridge': '#ff7f0e',     # orange
    'Lasso': '#2ca02c'      # green
}

# Plot 1: Coefficient Comparison (Large)
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(feature_names))
width = 0.2

ax1.bar(x_pos - 1.5*width, true_coefficients, width, 
        label='True', alpha=0.9, color='black', edgecolor='black', linewidth=1.5)
ax1.bar(x_pos - 0.5*width, baseline['coefficients'], width,
        label='Standard', alpha=0.8, color=model_colors['Standard'])
ax1.bar(x_pos + 0.5*width, ridge_results['coefficients'], width,
        label='Ridge', alpha=0.8, color=model_colors['Ridge'])
ax1.bar(x_pos + 1.5*width, lasso_results['coefficients'], width,
        label='Lasso', alpha=0.8, color=model_colors['Lasso'])

ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax1.set_title('Coefficient Comparison: How Well Did Each Model Recover the Truth?', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'F{i}' for i in range(len(feature_names))], fontsize=10)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add important feature markers
for i, is_important in enumerate(true_coefficients != 0):
    if is_important:
        ax1.axvspan(i-0.5, i+0.5, alpha=0.1, color='green')

# Plot 2: Test R² Comparison
ax2 = fig.add_subplot(gs[1, 0])
models = ['Standard', 'Ridge', 'Lasso']
r2_values = [baseline['test_r2'], ridge_results['test_r2'], lasso_results['test_r2']]
colors = [model_colors[m] for m in models]

bars = ax2.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Test R² Score', fontweight='bold')
ax2.set_title('Test Set R² (Higher is Better)', fontweight='bold')
ax2.set_ylim([min(r2_values) * 0.95, max(r2_values) * 1.05])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight best
best_idx = np.argmax(r2_values)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# Plot 3: Test RMSE Comparison
ax3 = fig.add_subplot(gs[1, 1])
rmse_values = [baseline['test_rmse'], ridge_results['test_rmse'], lasso_results['test_rmse']]

bars = ax3.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Test RMSE', fontweight='bold')
ax3.set_title('Test Set RMSE (Lower is Better)', fontweight='bold')
ax3.set_ylim([min(rmse_values) * 0.95, max(rmse_values) * 1.05])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight best
best_idx = np.argmin(rmse_values)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# Plot 4: Overfitting Analysis
ax4 = fig.add_subplot(gs[1, 2])
overfit_gaps = [
    baseline['train_r2'] - baseline['test_r2'],
    ridge_results['train_r2'] - ridge_results['test_r2'],
    lasso_results['train_r2'] - lasso_results['test_r2']
]

bars = ax4.bar(models, overfit_gaps, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Train R² - Test R²', fontweight='bold')
ax4.set_title('Overfitting Gap (Lower is Better)', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Add value labels
for bar, val in zip(bars, overfit_gaps):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight best
best_idx = np.argmin(overfit_gaps)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# Plot 5: Feature Count
ax5 = fig.add_subplot(gs[2, 0])
feature_counts = [
    len(feature_names),
    len(feature_names),
    lasso_results['n_features_selected']
]

bars = ax5.bar(models, feature_counts, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Number of Features', fontweight='bold')
ax5.set_title('Model Complexity', fontweight='bold')
ax5.set_ylim([0, len(feature_names) + 1])
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels and percentage
for bar, val in zip(bars, feature_counts):
    height = bar.get_height()
    pct = (val / len(feature_names)) * 100
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val)}\n({pct:.0f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 6: Mean Coefficient Error
ax6 = fig.add_subplot(gs[2, 1])
coef_errors = [
    coef_comparison['Std Error'].mean(),
    coef_comparison['Ridge Error'].mean(),
    coef_comparison['Lasso Error'].mean()
]

bars = ax6.bar(models, coef_errors, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Mean |Coefficient Error|', fontweight='bold')
ax6.set_title('Coefficient Recovery Error', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, coef_errors):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight best
best_idx = np.argmin(coef_errors)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# Plot 7: Predictions Scatter (all models)
ax7 = fig.add_subplot(gs[2, 2])

ax7.scatter(y_test, baseline['test_predictions'], alpha=0.5, s=30, 
           color=model_colors['Standard'], label='Standard')
ax7.scatter(y_test, ridge_results['test_predictions'], alpha=0.5, s=30, 
           color=model_colors['Ridge'], label='Ridge', marker='s')
ax7.scatter(y_test, lasso_results['test_predictions'], alpha=0.5, s=30, 
           color=model_colors['Lasso'], label='Lasso', marker='^')

# Perfect prediction line
ax7.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'k--', lw=2, label='Perfect', alpha=0.5)

ax7.set_xlabel('Actual Values', fontweight='bold')
ax7.set_ylabel('Predicted Values', fontweight='bold')
ax7.set_title('Test Predictions: All Models', fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

plt.savefig('05_comprehensive_comparison.png', dpi=100, bbox_inches='tight')
print("\nComprehensive comparison saved as '05_comprehensive_comparison.png'")

# ============================================================================
# STEP 6: Feature Selection Analysis (for Lasso)
# ============================================================================

print("\n" + "=" * 70)
print("LASSO FEATURE SELECTION ANALYSIS")
print("=" * 70)

# Identify which features Lasso selected
lasso_coefs = lasso_results['coefficients']
selected_mask = lasso_coefs != 0

print(f"\nLasso selected {sum(selected_mask)} out of {len(feature_names)} features:")

for i, (name, selected, true_coef, lasso_coef) in enumerate(
    zip(feature_names, selected_mask, true_coefficients, lasso_coefs)):
    
    is_important = true_coef != 0
    status = "✓" if selected else "✗"
    
    if is_important and selected:
        result = "CORRECT (kept important feature)"
    elif is_important and not selected:
        result = "ERROR (eliminated important feature)"
    elif not is_important and selected:
        result = "NOISE (kept noise feature)"
    elif not is_important and not selected:
        result = "CORRECT (eliminated noise)"
    
    print(f"  {status} {name}: True={true_coef:6.2f}, Lasso={lasso_coef:6.3f} - {result}")

# Calculate selection accuracy
noise_indices = [i for i in range(len(true_coefficients)) if true_coefficients[i] == 0]
important_indices = [i for i in range(len(true_coefficients)) if true_coefficients[i] != 0]

noise_eliminated = sum(1 for i in noise_indices if lasso_coefs[i] == 0)
important_kept = sum(1 for i in important_indices if lasso_coefs[i] != 0)

print(f"\nFeature Selection Accuracy:")
print(f"  Noise features eliminated: {noise_eliminated}/{len(noise_indices)} ({noise_eliminated/len(noise_indices)*100:.1f}%)")
print(f"  Important features kept: {important_kept}/{len(important_indices)} ({important_kept/len(important_indices)*100:.1f}%)")

# ============================================================================
# STEP 7: Decision Framework
# ============================================================================

print("\n" + "=" * 70)
print("DECISION FRAMEWORK: WHICH MODEL TO CHOOSE?")
print("=" * 70)

# Calculate scores for each criterion
scores = {
    'Standard': {
        'Performance': baseline['test_r2'],
        'Simplicity': 0,  # Uses all features
        'Accuracy': 1 / (1 + mean_errors['Standard Regression']),  # Inverse of error
        'Generalization': 1 / (1 + (baseline['train_r2'] - baseline['test_r2']))
    },
    'Ridge': {
        'Performance': ridge_results['test_r2'],
        'Simplicity': 0,  # Uses all features
        'Accuracy': 1 / (1 + mean_errors['Ridge Regression']),
        'Generalization': 1 / (1 + (ridge_results['train_r2'] - ridge_results['test_r2']))
    },
    'Lasso': {
        'Performance': lasso_results['test_r2'],
        'Simplicity': 1 - (lasso_results['n_features_selected'] / len(feature_names)),
        'Accuracy': 1 / (1 + mean_errors['Lasso Regression']),
        'Generalization': 1 / (1 + (lasso_results['train_r2'] - lasso_results['test_r2']))
    }
}

# Create scoring table
print("\nModel Evaluation Scorecard (normalized 0-1, higher is better):")
print("-" * 70)
print(f"{'Criterion':<20} {'Standard':<15} {'Ridge':<15} {'Lasso':<15}")
print("-" * 70)

for criterion in ['Performance', 'Simplicity', 'Accuracy', 'Generalization']:
    std_score = scores['Standard'][criterion]
    ridge_score = scores['Ridge'][criterion]
    lasso_score = scores['Lasso'][criterion]
    
    print(f"{criterion:<20} {std_score:<15.4f} {ridge_score:<15.4f} {lasso_score:<15.4f}")

# Overall recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

print("\nChoose based on your priorities:")

print("\n1. STANDARD REGRESSION if:")
print("   - You have few features (no overfitting risk)")
print("   - Features are carefully selected already")
print("   - You need unbiased coefficient estimates")

print("\n2. RIDGE REGRESSION if:")
print("   - Features are correlated (multicollinearity)")
print("   - You want to keep all features")
print("   - You prioritize prediction over interpretation")
print("   - You need stability in coefficient estimates")

print("\n3. LASSO REGRESSION if:")
print("   - You have many features (high-dimensional)")
print("   - You believe many features are irrelevant")
print("   - Interpretability is crucial")
print("   - You want automatic feature selection")
print("   - You need a sparse, simple model")

# Make data-driven recommendation
best_test_r2 = max(baseline['test_r2'], ridge_results['test_r2'], lasso_results['test_r2'])
best_model = ''
if best_test_r2 == baseline['test_r2']:
    best_model = 'Standard Regression'
elif best_test_r2 == ridge_results['test_r2']:
    best_model = 'Ridge Regression'
else:
    best_model = 'Lasso Regression'

print(f"\n" + "=" * 70)
print(f"FOR THIS DATASET: {best_model} performed best")
print("=" * 70)

if best_model == 'Lasso Regression':
    print(f"\nLasso selected {lasso_results['n_features_selected']} features and achieved:")
    print(f"  - Test R²: {lasso_results['test_r2']:.4f}")
    print(f"  - Feature selection accuracy: {(noise_eliminated + important_kept)}/{len(feature_names)}")
    print("\nThis makes it the best choice for:")
    print("  ✓ Interpretability (fewer features)")
    print("  ✓ Simplicity (sparse model)")
    print("  ✓ Performance (good test R²)")

# ============================================================================
# STEP 8: Save Final Summary
# ============================================================================

# Create a final summary report
summary_report = {
    'comparison_table': comparison_df,
    'coefficient_comparison': coef_comparison,
    'best_test_r2_model': best_model,
    'feature_selection_accuracy': {
        'noise_eliminated': noise_eliminated,
        'important_kept': important_kept,
        'total_accuracy': (noise_eliminated + important_kept) / len(feature_names)
    },
    'recommendations': {
        'standard': 'Use when: Few features, no multicollinearity, need unbiased estimates',
        'ridge': 'Use when: Multicollinearity present, keep all features, prioritize prediction',
        'lasso': 'Use when: High-dimensional, need feature selection, prioritize interpretability'
    }
}

# Save comparison table as CSV
comparison_df.to_csv('05_model_comparison.csv', index=False)
coef_comparison.to_csv('05_coefficient_comparison.csv', index=False)

print("\n" + "=" * 70)
print("FINAL OUTPUT FILES CREATED:")
print("=" * 70)
print("  - 05_comprehensive_comparison.png (visual comparison)")
print("  - 05_model_comparison.csv (performance metrics)")
print("  - 05_coefficient_comparison.csv (coefficient details)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ACTIVITY COMPLETE!")
print("=" * 70)

print("\nYou've now learned:")
print("  ✓ How to generate and prepare data for regularization")
print("  ✓ How standard linear regression performs without regularization")
print("  ✓ How Ridge (L2) regularization shrinks coefficients")
print("  ✓ How Lasso (L1) regularization performs feature selection")
print("  ✓ How to compare models systematically")
print("  ✓ How to choose the right approach for your problem")

print("\n" + "=" * 70)
print("NEXT STEPS FOR THE ASSIGNMENT:")
print("=" * 70)
print("\n1. Apply these same techniques to the real housing dataset")
print("2. Use AIC and BIC for model selection (we'll cover in class)")
print("3. Create comprehensive visualizations")
print("4. Write clear interpretations")
print("5. Make evidence-based recommendations")

print("\nRemember:")
print("  - Always standardize features before regularization")
print("  - Use cross-validation to select λ")
print("  - Consider both performance AND interpretability")
print("  - Regularization is about finding the right balance")

print("\n" + "=" * 70)

# ============================================================================
# DISCUSSION QUESTIONS FOR CLASS:
# ============================================================================

"""
CLASS DISCUSSION QUESTIONS:

1. CONCEPTUAL UNDERSTANDING:
   - Why does L1 penalty lead to exact zeros but L2 doesn't?
   - Draw the constraint regions for L1 vs L2 regularization
   - Explain bias-variance tradeoff in the context of regularization

2. PRACTICAL APPLICATION:
   - Given a new dataset, how would you decide between Ridge and Lasso?
   - What if you have 1000 features but only 100 samples?
   - How would you explain your choice to a non-technical stakeholder?

3. RESULTS INTERPRETATION:
   - Did regularization help in our synthetic example?
   - Were the selected features meaningful?
   - What would you conclude from the coefficient comparison?

4. REAL-WORLD CONSIDERATIONS:
   - When might you use Elastic Net (combination of L1 and L2)?
   - How would you handle categorical features?
   - What about non-linear relationships?

5. ASSIGNMENT PREPARATION:
   - What additional metrics might be useful for the housing dataset?
   - How will you present your findings in the video?
   - What business insights can you derive from regularization results?
"""
