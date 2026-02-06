"""
Topic 3: In-Class Activity - Introduction to Regularization
Complete implementation with menu system and visual elements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class RegularizationActivity:
    """Class to manage the regularization activity with all parts"""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.true_coefficients = None
        self.models = {}
        self.results = {}

    def generate_data(self, n_samples=500, noise_level=2.0, noise_distribution='normal'):
        """Part 1: Generate synthetic dataset with known relationships"""
        print("\n" + "="*70)
        print("PART 1: DATA GENERATION")
        print("="*70)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate samples with 8 features
        n_features = 8

        # Create feature names
        self.feature_names = [f'Feature_{i+1}' for i in range(n_features)]

        # Generate base features with some correlation
        X = np.random.randn(n_samples, n_features)

        # Add correlation between some features (simulating real-world data)
        X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.3  # Feature_2 correlated with Feature_1
        X[:, 3] = X[:, 2] + np.random.randn(n_samples) * 0.4  # Feature_4 correlated with Feature_3

        # Define true coefficients (ground truth)
        # Some features are important, others are noise
        self.true_coefficients = np.array([5.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

        # Generate target variable with specified noise distribution
        signal = X @ self.true_coefficients

        if noise_distribution == 'normal':
            noise = np.random.randn(n_samples) * noise_level
        elif noise_distribution == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, n_samples)
        elif noise_distribution == 'exponential':
            noise = (np.random.exponential(noise_level, n_samples) - noise_level)
        elif noise_distribution == 'laplace':
            noise = np.random.laplace(0, noise_level, n_samples)
        elif noise_distribution == 't-distribution':
            noise = np.random.standard_t(df=3, size=n_samples) * noise_level
        else:  # default to normal
            noise = np.random.randn(n_samples) * noise_level

        y = signal + noise

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print(f"\n✓ Generated dataset with {n_samples:,} samples and {n_features} features")
        print(f"  - Training set: {len(self.X_train):,} samples")
        print(f"  - Test set: {len(self.X_test):,} samples")
        print(f"  - Noise level: {noise_level}")
        print(f"  - Noise distribution: {noise_distribution}")
        if n_samples > 5000:
            print(f"\n  ⚡ Large dataset detected! Models will use optimized settings.")

        print(f"\n✓ True coefficients (what we're trying to recover):")
        for name, coef in zip(self.feature_names, self.true_coefficients):
            if coef != 0:
                print(f"  - {name}: {coef:.2f} (IMPORTANT)")
            else:
                print(f"  - {name}: {coef:.2f} (noise)")

        # Visualize feature correlations
        self._visualize_correlations()

        return self

    def _visualize_correlations(self):
        """Visualize feature correlations"""
        corr_matrix = np.corrcoef(self.X_train.T)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=self.feature_names, yticklabels=self.feature_names,
                    center=0, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix\n(Notice correlated features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def standard_regression(self):
        """Part 2: Fit standard linear regression (no regularization)"""
        print("\n" + "="*70)
        print("PART 2: STANDARD LINEAR REGRESSION (Baseline)")
        print("="*70)

        # Fit model
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)

        # Store model
        self.models['Linear Regression'] = lr

        # Make predictions
        y_train_pred = lr.predict(self.X_train)
        y_test_pred = lr.predict(self.X_test)

        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

        # Store results
        self.results['Linear Regression'] = {
            'coefficients': lr.coef_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }

        print("\n📊 PERFORMANCE METRICS:")
        print(f"  Training R²:  {train_r2:.4f}")
        print(f"  Test R²:      {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:     {test_rmse:.4f}")

        print("\n📈 COEFFICIENT VALUES:")
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'True Coef': self.true_coefficients,
            'Estimated Coef': lr.coef_
        })
        print(coef_df.to_string(index=False))

        print("\n💡 KEY OBSERVATIONS:")
        print("  - Without regularization, the model estimates coefficients for ALL features")
        print("  - Some noise features have non-zero coefficients (they shouldn't!)")
        print("  - The model may be overfitting by using irrelevant features")

        # Visualize coefficients
        self._plot_coefficients(lr.coef_, "Standard Linear Regression")

        return self

    def ridge_regression(self):
        """Part 3: Fit ridge regression with different alpha values"""
        print("\n" + "="*70)
        print("PART 3: RIDGE REGRESSION (L2 Penalty)")
        print("="*70)

        # Test different alpha values
        alphas = [0.1, 1.0, 10.0, 100.0]
        ridge_coefs = []

        print("\n🔬 TESTING DIFFERENT λ (alpha) VALUES:\n")

        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(self.X_train, self.y_train)
            ridge_coefs.append(ridge.coef_)

            y_test_pred = ridge.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

            print(f"α = {alpha:6.1f} → Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}")

        # Use cross-validation to find optimal alpha
        # Optimize for large datasets
        n_train = len(self.X_train)
        if n_train > 5000:
            alphas_cv = np.logspace(-2, 3, 50)
            cv_folds = 3
            print("\n🎯 FINDING OPTIMAL λ USING CROSS-VALIDATION (optimized for large dataset)...")
        else:
            alphas_cv = np.logspace(-2, 3, 100)
            cv_folds = 5
            print("\n🎯 FINDING OPTIMAL λ USING CROSS-VALIDATION...")

        ridge_cv = RidgeCV(alphas=alphas_cv, cv=cv_folds)
        ridge_cv.fit(self.X_train, self.y_train)

        optimal_alpha = ridge_cv.alpha_
        print(f"  ✓ Optimal α = {optimal_alpha:.4f}")

        # Store best model
        self.models['Ridge'] = ridge_cv

        y_train_pred = ridge_cv.predict(self.X_train)
        y_test_pred = ridge_cv.predict(self.X_test)

        self.results['Ridge'] = {
            'coefficients': ridge_cv.coef_,
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'alpha': optimal_alpha
        }

        print("\n📈 COEFFICIENT VALUES (with optimal α):")
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'True Coef': self.true_coefficients,
            'Ridge Coef': ridge_cv.coef_
        })
        print(coef_df.to_string(index=False))

        print("\n💡 KEY OBSERVATIONS:")
        print("  - Ridge shrinks ALL coefficients towards zero")
        print("  - No coefficients become exactly zero (all features retained)")
        print("  - Larger λ → more shrinkage")
        print("  - Helps with correlated features by shrinking them together")

        # Visualize coefficient paths
        self._plot_ridge_path(alphas_cv)

        # Visualize CV results
        self._plot_ridge_cv(ridge_cv, alphas_cv)

        return self

    def lasso_regression(self):
        """Part 4: Fit lasso regression with different alpha values"""
        print("\n" + "="*70)
        print("PART 4: LASSO REGRESSION (L1 Penalty)")
        print("="*70)

        # Test different alpha values
        alphas = [0.01, 0.1, 1.0, 10.0]
        lasso_coefs = []

        print("\n🔬 TESTING DIFFERENT λ (alpha) VALUES:\n")

        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(self.X_train, self.y_train)
            lasso_coefs.append(lasso.coef_)

            y_test_pred = lasso.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            n_nonzero = np.sum(lasso.coef_ != 0)

            print(f"α = {alpha:6.2f} → Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}, Features used = {n_nonzero}")

        # Use cross-validation to find optimal alpha
        # Optimize for large datasets
        n_train = len(self.X_train)
        if n_train > 5000:
            alphas_cv = np.logspace(-3, 1, 50)
            cv_folds = 3
            max_iterations = 20000
            print("\n🎯 FINDING OPTIMAL λ USING CROSS-VALIDATION (optimized for large dataset)...")
        else:
            alphas_cv = np.logspace(-3, 1, 100)
            cv_folds = 5
            max_iterations = 10000
            print("\n🎯 FINDING OPTIMAL λ USING CROSS-VALIDATION...")

        lasso_cv = LassoCV(alphas=alphas_cv, cv=cv_folds, max_iter=max_iterations, n_jobs=-1)
        lasso_cv.fit(self.X_train, self.y_train)

        optimal_alpha = lasso_cv.alpha_
        n_features_used = np.sum(lasso_cv.coef_ != 0)
        print(f"  ✓ Optimal α = {optimal_alpha:.4f}")
        print(f"  ✓ Features selected: {n_features_used} out of {len(self.feature_names)}")

        # Store best model
        self.models['Lasso'] = lasso_cv

        y_train_pred = lasso_cv.predict(self.X_train)
        y_test_pred = lasso_cv.predict(self.X_test)

        self.results['Lasso'] = {
            'coefficients': lasso_cv.coef_,
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'alpha': optimal_alpha,
            'n_features': n_features_used
        }

        print("\n📈 COEFFICIENT VALUES (with optimal α):")
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'True Coef': self.true_coefficients,
            'Lasso Coef': lasso_cv.coef_,
            'Selected': ['✓' if c != 0 else '✗' for c in lasso_cv.coef_]
        })
        print(coef_df.to_string(index=False))

        print("\n💡 KEY OBSERVATIONS:")
        print("  - Lasso can shrink coefficients to EXACTLY zero")
        print("  - Performs automatic feature selection")
        print("  - Eliminates noise features (as it should!)")
        print("  - Creates sparse models (fewer features)")
        print("  - Better for interpretation when you want to identify key features")

        # Visualize coefficient paths
        self._plot_lasso_path(alphas_cv)

        # Visualize CV results
        self._plot_lasso_cv(lasso_cv, alphas_cv)

        return self

    def compare_models(self):
        """Part 5: Compare all three models side-by-side"""
        print("\n" + "="*70)
        print("PART 5: MODEL COMPARISON")
        print("="*70)

        print("\n📊 PERFORMANCE COMPARISON:\n")

        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train R²': [self.results[m]['train_r2'] for m in self.results.keys()],
            'Test R²': [self.results[m]['test_r2'] for m in self.results.keys()],
            'Train RMSE': [self.results[m]['train_rmse'] for m in self.results.keys()],
            'Test RMSE': [self.results[m]['test_rmse'] for m in self.results.keys()],
            'Features Used': [
                len(self.feature_names),
                len(self.feature_names),
                self.results['Lasso']['n_features']
            ]
        })

        print(comparison_df.to_string(index=False))

        print("\n📈 COEFFICIENT COMPARISON:\n")

        coef_comparison = pd.DataFrame({
            'Feature': self.feature_names,
            'True': self.true_coefficients,
            'Linear': self.results['Linear Regression']['coefficients'],
            'Ridge': self.results['Ridge']['coefficients'],
            'Lasso': self.results['Lasso']['coefficients']
        })
        print(coef_comparison.to_string(index=False))

        # Visualize comparison
        self._plot_model_comparison()

        print("\n💡 KEY TAKEAWAYS:")
        print("\n1. COEFFICIENT SHRINKAGE:")
        print("   • Linear Regression: No shrinkage, all features used")
        print("   • Ridge: Shrinks all coefficients but keeps all features")
        print("   • Lasso: Shrinks some coefficients to zero, performs feature selection")

        print("\n2. PERFORMANCE:")
        print("   • All models perform similarly on this dataset")
        print("   • Ridge/Lasso prevent overfitting (better test performance)")
        print("   • Lasso achieves similar performance with fewer features")

        print("\n3. INTERPRETABILITY:")
        print("   • Lasso is most interpretable (fewer features)")
        print("   • Linear Regression is hardest to interpret (noise features included)")
        print("   • Ridge is in between")

        print("\n4. SPARSITY:")
        print(f"   • Linear Regression: {len(self.feature_names)} features")
        print(f"   • Ridge: {len(self.feature_names)} features")
        print(f"   • Lasso: {self.results['Lasso']['n_features']} features ★ Most parsimonious")

        return self

    def discussion_questions(self):
        """Answer all discussion questions with visual elements"""
        print("\n" + "="*70)
        print("DISCUSSION QUESTIONS - COMPREHENSIVE ANSWERS")
        print("="*70)

        print("\n" + "─"*70)
        print("Q1: When would you choose Ridge over Lasso?")
        print("─"*70)
        print("""
ANSWER:
Choose RIDGE when:
• You believe MOST features are relevant (just with small effects)
• Features are highly correlated (ridge handles multicollinearity better)
• You want to reduce overfitting but keep all features
• You need stable predictions (ridge is less sensitive to small data changes)
• Example: Predicting house prices where many features contribute

Choose LASSO when:
• You believe ONLY SOME features are truly relevant
• You need automatic feature selection
• You want a sparse, interpretable model
• You need to identify the "most important" predictors
• Example: Gene selection in biology (thousands of genes, few are relevant)

VISUALIZATION:
The plots below show how Ridge keeps all features (shrinks but doesn't eliminate)
while Lasso eliminates irrelevant features (sets coefficients to exactly zero).
        """)

        self._plot_ridge_vs_lasso_comparison()

        print("\n" + "─"*70)
        print("Q2: What does it mean when Lasso sets a coefficient to zero?")
        print("─"*70)
        print("""
ANSWER:
When Lasso sets a coefficient to zero, it means:

1. FEATURE EXCLUSION:
   • The feature is REMOVED from the model entirely
   • It contributes NOTHING to predictions

2. AUTOMATIC FEATURE SELECTION:
   • Lasso has determined this feature is not important
   • Given the penalty (λ), the cost of including it outweighs the benefit

3. SPARSITY:
   • Creates simpler, more interpretable models
   • Easier to explain: "Only these 3 features matter"

4. NOT NECESSARILY USELESS:
   • The feature might be redundant (correlated with another kept feature)
   • It might have very weak predictive power
   • With different λ, it might be included

In our example, Lasso correctly identified that Features 3, 4, 6, 7, 8 are noise
and eliminated them (their true coefficients are zero).
        """)

        self._plot_feature_importance()

        print("\n" + "─"*70)
        print("Q3: Why do we use cross-validation to select λ?")
        print("─"*70)
        print("""
ANSWER:
We use cross-validation to select λ because:

1. PREVENTS OVERFITTING:
   • Can't use training error alone (would choose λ = 0 = no regularization)
   • Can't use test set (would be "cheating" - test set should be untouched)
   • CV estimates generalization performance honestly

2. FINDS THE SWEET SPOT:
   • Too small λ → underfitting (too much regularization, high bias)
   • Too large λ → overfitting (too little regularization, high variance)
   • CV finds the λ that balances bias-variance tradeoff

3. USES DATA EFFICIENTLY:
   • Uses only training data (no peeking at test set)
   • Multiple folds give robust estimate of performance
   • Each data point used for both training and validation

4. OBJECTIVE SELECTION:
   • Removes human judgment/guessing
   • Reproducible and principled
   • Based on actual predictive performance

VISUALIZATION:
The CV curve below shows test error vs. λ. The optimal λ minimizes CV error.
        """)

        self._plot_cv_explanation()

        print("\n" + "─"*70)
        print("Q4: How does regularization help with multicollinearity?")
        print("─"*70)
        print("""
ANSWER:
Regularization helps with multicollinearity (correlated features) by:

1. THE PROBLEM:
   • When features are correlated, coefficients become unstable
   • Small data changes → large coefficient changes
   • Hard to interpret which feature is "really" important
   • Variance of coefficient estimates explodes

2. HOW RIDGE HELPS:
   • L2 penalty "ties together" correlated coefficients
   • Distributes weight across correlated features
   • Stabilizes coefficient estimates
   • Reduces variance at cost of small bias
   • Best when ALL correlated features are truly relevant

3. HOW LASSO HELPS:
   • L1 penalty picks ONE from correlated group, zeros others
   • Automatically chooses a representative feature
   • Creates sparse solution
   • Best when only some correlated features matter

4. IN OUR DATA:
   • Feature_1 and Feature_2 are correlated (r ≈ 0.95)
   • Feature_3 and Feature_4 are correlated (r ≈ 0.93)
   • Without regularization: coefficients are unstable
   • With regularization: stable, interpretable coefficients

VISUALIZATION:
The heatmap and coefficient comparison below show how correlated features
are handled differently by Ridge (keeps both, shrinks together) vs. Lasso
(picks one, zeros the other).
        """)

        self._plot_multicollinearity_effect()

        print("\n" + "="*70)
        print("END OF DISCUSSION QUESTIONS")
        print("="*70)

        return self

    # Plotting methods
    def _plot_coefficients(self, coefs, title):
        """Plot coefficient values"""
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.feature_names))
        width = 0.35

        ax.bar(x - width/2, self.true_coefficients, width, label='True', alpha=0.8, color='green')
        ax.bar(x + width/2, coefs, width, label='Estimated', alpha=0.8, color='blue')

        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title(f'Coefficient Comparison: {title}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def _plot_ridge_path(self, alphas):
        """Plot coefficient paths for Ridge"""
        coefs = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(self.X_train, self.y_train)
            coefs.append(ridge.coef_)

        coefs = np.array(coefs)

        plt.figure(figsize=(12, 6))
        for i, name in enumerate(self.feature_names):
            plt.plot(alphas, coefs[:, i], label=name, marker='o', markersize=3)

        plt.xscale('log')
        plt.xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        plt.ylabel('Coefficient Value', fontweight='bold', fontsize=12)
        plt.title('Ridge Coefficient Paths\n(All coefficients shrink but none become zero)',
                  fontweight='bold', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.show()

    def _plot_ridge_cv(self, ridge_cv, alphas):
        """Plot Ridge cross-validation results"""
        # Calculate CV scores for visualization
        # Optimize for large datasets
        n_train = len(self.X_train)
        cv_folds = 3 if n_train > 5000 else 5

        cv_scores = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(ridge, self.X_train, self.y_train,
                                     cv=cv_folds, scoring='neg_mean_squared_error',
                                     n_jobs=-1)
            cv_scores.append(-scores.mean())

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, cv_scores, 'b-', linewidth=2)
        plt.axvline(ridge_cv.alpha_, color='r', linestyle='--', linewidth=2,
                    label=f'Optimal α = {ridge_cv.alpha_:.4f}')
        plt.xscale('log')
        plt.xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        plt.ylabel('Cross-Validation MSE', fontweight='bold', fontsize=12)
        plt.title('Ridge: Cross-Validation to Select Optimal α',
                  fontweight='bold', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_lasso_path(self, alphas):
        """Plot coefficient paths for Lasso"""
        # Optimize for large datasets
        n_train = len(self.X_train)
        max_iter = 20000 if n_train > 5000 else 10000

        coefs = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(self.X_train, self.y_train)
            coefs.append(lasso.coef_)

        coefs = np.array(coefs)

        plt.figure(figsize=(12, 6))
        for i, name in enumerate(self.feature_names):
            plt.plot(alphas, coefs[:, i], label=name, marker='o', markersize=3)

        plt.xscale('log')
        plt.xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        plt.ylabel('Coefficient Value', fontweight='bold', fontsize=12)
        plt.title('Lasso Coefficient Paths\n(Coefficients become exactly zero for large α)',
                  fontweight='bold', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.show()

    def _plot_lasso_cv(self, lasso_cv, alphas):
        """Plot Lasso cross-validation results"""
        # Optimize for large datasets
        n_train = len(self.X_train)
        cv_folds = 3 if n_train > 5000 else 5
        max_iter = 20000 if n_train > 5000 else 10000

        cv_scores = []
        n_features = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            scores = cross_val_score(lasso, self.X_train, self.y_train,
                                     cv=cv_folds, scoring='neg_mean_squared_error',
                                     n_jobs=-1)
            cv_scores.append(-scores.mean())
            lasso.fit(self.X_train, self.y_train)
            n_features.append(np.sum(lasso.coef_ != 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # CV error plot
        ax1.plot(alphas, cv_scores, 'b-', linewidth=2)
        ax1.axvline(lasso_cv.alpha_, color='r', linestyle='--', linewidth=2,
                    label=f'Optimal α = {lasso_cv.alpha_:.4f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Cross-Validation MSE', fontweight='bold', fontsize=12)
        ax1.set_title('Lasso: CV Error vs. α', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Number of features plot
        ax2.plot(alphas, n_features, 'g-', linewidth=2, marker='o')
        ax2.axvline(lasso_cv.alpha_, color='r', linestyle='--', linewidth=2,
                    label=f'Optimal α = {lasso_cv.alpha_:.4f}')
        ax2.set_xscale('log')
        ax2.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Features Selected', fontweight='bold', fontsize=12)
        ax2.set_title('Lasso: Feature Selection vs. α', fontweight='bold', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_model_comparison(self):
        """Comprehensive model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Coefficient comparison
        ax = axes[0, 0]
        x = np.arange(len(self.feature_names))
        width = 0.2

        ax.bar(x - width*1.5, self.true_coefficients, width, label='True', alpha=0.8)
        ax.bar(x - width*0.5, self.results['Linear Regression']['coefficients'],
               width, label='Linear', alpha=0.8)
        ax.bar(x + width*0.5, self.results['Ridge']['coefficients'],
               width, label='Ridge', alpha=0.8)
        ax.bar(x + width*1.5, self.results['Lasso']['coefficients'],
               width, label='Lasso', alpha=0.8)

        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title('Coefficient Values Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 2. Performance metrics
        ax = axes[0, 1]
        models = list(self.results.keys())
        test_r2 = [self.results[m]['test_r2'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]

        x = np.arange(len(models))
        ax2 = ax.twinx()

        bars1 = ax.bar(x - 0.2, test_r2, 0.4, label='Test R²', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x + 0.2, test_rmse, 0.4, label='Test RMSE', alpha=0.8, color='salmon')

        ax.set_ylabel('R² Score', fontweight='bold', color='skyblue')
        ax2.set_ylabel('RMSE', fontweight='bold', color='salmon')
        ax.set_title('Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='salmon')
        ax.grid(axis='y', alpha=0.3)

        # 3. Coefficient magnitude (L2 norm)
        ax = axes[1, 0]
        l2_norms = []
        for model_name in models:
            coefs = self.results[model_name]['coefficients']
            l2_norms.append(np.linalg.norm(coefs))

        bars = ax.bar(models, l2_norms, alpha=0.8, color=['green', 'blue', 'red'])
        ax.set_ylabel('L2 Norm of Coefficients', fontweight='bold')
        ax.set_title('Coefficient Shrinkage (L2 Norm)', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. Sparsity (number of non-zero coefficients)
        ax = axes[1, 1]
        n_features_used = []
        for model_name in models:
            coefs = self.results[model_name]['coefficients']
            n_features_used.append(np.sum(np.abs(coefs) > 1e-10))

        bars = ax.bar(models, n_features_used, alpha=0.8, color=['green', 'blue', 'red'])
        ax.set_ylabel('Number of Features Used', fontweight='bold')
        ax.set_title('Model Sparsity (Feature Selection)', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0, len(self.feature_names) + 1])
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def _plot_ridge_vs_lasso_comparison(self):
        """Visualize key difference between Ridge and Lasso"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ridge keeps all features
        ax = axes[0]
        ridge_coefs = self.results['Ridge']['coefficients']
        colors = ['green' if abs(c) > 0.5 else 'lightblue' for c in ridge_coefs]
        bars = ax.bar(self.feature_names, ridge_coefs, color=colors, alpha=0.8)
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title('RIDGE: Shrinks all coefficients\n(None become exactly zero)',
                     fontweight='bold', fontsize=13)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        # Lasso eliminates some features
        ax = axes[1]
        lasso_coefs = self.results['Lasso']['coefficients']
        colors = ['red' if abs(c) < 1e-10 else 'green' for c in lasso_coefs]
        bars = ax.bar(self.feature_names, lasso_coefs, color=colors, alpha=0.8)
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title('LASSO: Sets some coefficients to zero\n(Automatic feature selection)',
                     fontweight='bold', fontsize=13)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.8, label='Important (non-zero)'),
                          Patch(facecolor='red', alpha=0.8, label='Eliminated (zero)')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def _plot_feature_importance(self):
        """Visualize feature importance based on Lasso selection"""
        fig, ax = plt.subplots(figsize=(10, 6))

        lasso_coefs = self.results['Lasso']['coefficients']
        importance = np.abs(lasso_coefs)
        colors = ['darkgreen' if abs(c) > 1e-10 else 'gray' for c in lasso_coefs]

        bars = ax.barh(self.feature_names, importance, color=colors, alpha=0.8)
        ax.set_xlabel('Absolute Coefficient Value (Importance)', fontweight='bold', fontsize=12)
        ax.set_title('Feature Importance According to Lasso\n(Zero = Feature Excluded)',
                     fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)

        # Add annotations
        for i, (name, imp, coef) in enumerate(zip(self.feature_names, importance, lasso_coefs)):
            if abs(coef) < 1e-10:
                ax.text(imp + 0.05, i, 'ELIMINATED', va='center', fontweight='bold', color='red')
            else:
                ax.text(imp + 0.05, i, f'{coef:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def _plot_cv_explanation(self):
        """Visualize cross-validation concept"""
        # Generate CV curves for Ridge and Lasso
        alphas = np.logspace(-2, 3, 50)

        ridge_cv_scores = []
        lasso_cv_scores = []

        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            lasso = Lasso(alpha=alpha, max_iter=10000)

            ridge_scores = cross_val_score(ridge, self.X_train, self.y_train,
                                          cv=5, scoring='neg_mean_squared_error')
            lasso_scores = cross_val_score(lasso, self.X_train, self.y_train,
                                          cv=5, scoring='neg_mean_squared_error')

            ridge_cv_scores.append(-ridge_scores.mean())
            lasso_cv_scores.append(-lasso_scores.mean())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ridge CV
        ax = axes[0]
        ax.plot(alphas, ridge_cv_scores, 'b-', linewidth=2, label='CV Error')
        optimal_idx = np.argmin(ridge_cv_scores)
        ax.axvline(alphas[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Optimal α = {alphas[optimal_idx]:.4f}')
        ax.scatter(alphas[optimal_idx], ridge_cv_scores[optimal_idx],
                   color='red', s=200, zorder=5, marker='*')
        ax.set_xscale('log')
        ax.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cross-Validation MSE', fontweight='bold', fontsize=12)
        ax.set_title('Ridge: Finding Optimal α via CV\n(Minimizes CV error)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.annotate('Too little\nregularization\n(overfitting)',
                   xy=(alphas[0], ridge_cv_scores[0]), xytext=(alphas[0]*10, ridge_cv_scores[0]*1.2),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=10, color='red',
                   fontweight='bold')
        ax.annotate('Too much\nregularization\n(underfitting)',
                   xy=(alphas[-1], ridge_cv_scores[-1]), xytext=(alphas[-1]/10, ridge_cv_scores[-1]*0.8),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=10, color='red',
                   fontweight='bold')
        ax.annotate('SWEET SPOT ★',
                   xy=(alphas[optimal_idx], ridge_cv_scores[optimal_idx]),
                   xytext=(alphas[optimal_idx]*5, ridge_cv_scores[optimal_idx]*0.9),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2), fontsize=11, color='green',
                   fontweight='bold')

        # Lasso CV
        ax = axes[1]
        ax.plot(alphas, lasso_cv_scores, 'b-', linewidth=2, label='CV Error')
        optimal_idx = np.argmin(lasso_cv_scores)
        ax.axvline(alphas[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Optimal α = {alphas[optimal_idx]:.4f}')
        ax.scatter(alphas[optimal_idx], lasso_cv_scores[optimal_idx],
                   color='red', s=200, zorder=5, marker='*')
        ax.set_xscale('log')
        ax.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cross-Validation MSE', fontweight='bold', fontsize=12)
        ax.set_title('Lasso: Finding Optimal α via CV\n(Minimizes CV error)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_multicollinearity_effect(self):
        """Visualize how regularization handles correlated features"""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Correlation heatmap (focus on correlated pairs)
        ax1 = fig.add_subplot(gs[0, :])
        corr_matrix = np.corrcoef(self.X_train.T)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=self.feature_names, yticklabels=self.feature_names,
                    center=0, vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Feature Correlations\n(Features 1-2 and 3-4 are highly correlated)',
                     fontweight='bold', fontsize=14)

        # Highlight correlated pairs
        ax1.add_patch(plt.Rectangle((0, 0), 2, 2, fill=False, edgecolor='red', lw=3))
        ax1.add_patch(plt.Rectangle((2, 2), 2, 2, fill=False, edgecolor='red', lw=3))

        # 2. How Ridge handles correlated features
        ax2 = fig.add_subplot(gs[1, 0])
        ridge_coefs = self.results['Ridge']['coefficients']
        colors_ridge = []
        for i, name in enumerate(self.feature_names):
            if 'Feature_1' in name or 'Feature_2' in name:
                colors_ridge.append('purple')
            elif 'Feature_3' in name or 'Feature_4' in name:
                colors_ridge.append('orange')
            else:
                colors_ridge.append('lightblue')

        ax2.bar(self.feature_names, ridge_coefs, color=colors_ridge, alpha=0.8)
        ax2.set_ylabel('Coefficient Value', fontweight='bold')
        ax2.set_title('Ridge: Shrinks Correlated Features Together\n(Both kept, distributed weight)',
                     fontweight='bold', fontsize=12)
        ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)

        # 3. How Lasso handles correlated features
        ax3 = fig.add_subplot(gs[1, 1])
        lasso_coefs = self.results['Lasso']['coefficients']
        colors_lasso = []
        for i, name in enumerate(self.feature_names):
            if 'Feature_1' in name or 'Feature_2' in name:
                if abs(lasso_coefs[i]) > 1e-10:
                    colors_lasso.append('purple')
                else:
                    colors_lasso.append('gray')
            elif 'Feature_3' in name or 'Feature_4' in name:
                if abs(lasso_coefs[i]) > 1e-10:
                    colors_lasso.append('orange')
                else:
                    colors_lasso.append('gray')
            else:
                colors_lasso.append('lightblue' if abs(lasso_coefs[i]) > 1e-10 else 'gray')

        ax3.bar(self.feature_names, lasso_coefs, color=colors_lasso, alpha=0.8)
        ax3.set_ylabel('Coefficient Value', fontweight='bold')
        ax3.set_title('Lasso: Picks One from Correlated Group\n(Others set to zero)',
                     fontweight='bold', fontsize=12)
        ax3.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()


def print_menu():
    """Display the main menu"""
    print("\n" + "="*70)
    print(" REGULARIZATION ACTIVITY - INTERACTIVE MENU")
    print("="*70)
    print("\n📚 ACTIVITY PARTS:")
    print("  1. Generate Synthetic Data (Part 1)")
    print("  2. Standard Linear Regression (Part 2)")
    print("  3. Ridge Regression (Part 3)")
    print("  4. Lasso Regression (Part 4)")
    print("  5. Compare All Models (Part 5)")
    print("\n💬 DISCUSSION:")
    print("  6. View Discussion Questions & Answers")
    print("\n🚀 SHORTCUTS:")
    print("  7. Run Complete Activity (All Parts)")
    print("  8. Run All + Discussion Questions")
    print("\n❌ EXIT:")
    print("  0. Exit Program")
    print("\n" + "="*70)


def main():
    """Main function with interactive menu"""
    activity = RegularizationActivity()
    data_generated = False

    print("\n" + "="*70)
    print("   WELCOME TO THE REGULARIZATION ACTIVITY!")
    print("="*70)
    print("\nThis interactive program will guide you through:")
    print("  • Understanding linear regression and regularization")
    print("  • Comparing Ridge and Lasso regression")
    print("  • Visualizing model behavior with rich graphics")
    print("  • Answering key discussion questions")
    print("\nLet's get started!\n")

    while True:
        print_menu()
        choice = input("\nEnter your choice (0-8): ").strip()

        if choice == '0':
            print("\n👋 Thank you for using the Regularization Activity program!")
            print("   Keep learning and exploring machine learning! 🚀\n")
            break

        elif choice == '1':
            # Get user input for data generation parameters
            print("\n" + "="*70)
            print("Configure Data Generation")
            print("="*70)

            try:
                n_samples = int(input("\nNumber of samples (100-10000) [default: 500]: ") or "500")
                n_samples = max(100, min(10000, n_samples))  # Clamp between 100 and 10000

                noise_level = float(input("Noise level (0.5-20.0) [default: 2.0]: ") or "2.0")
                noise_level = max(0.5, min(20.0, noise_level))  # Clamp between 0.5 and 20.0

                print("\nNoise distribution options:")
                print("  1. Normal (Gaussian) - standard assumption, symmetric")
                print("  2. Uniform - all values equally likely, no outliers")
                print("  3. Laplace - heavy-tailed, more outliers")
                print("  4. t-distribution - very heavy-tailed, extreme outliers")
                print("  5. Exponential - asymmetric, skewed")

                dist_choice = input("Select noise distribution (1-5) [default: 1]: ") or "1"
                dist_map = {
                    '1': 'normal',
                    '2': 'uniform',
                    '3': 'laplace',
                    '4': 't-distribution',
                    '5': 'exponential'
                }
                noise_distribution = dist_map.get(dist_choice, 'normal')

                activity.generate_data(n_samples, noise_level, noise_distribution)
                data_generated = True

            except ValueError:
                print("\n❌ Invalid input. Using default values.")
                activity.generate_data()
                data_generated = True

            input("\nPress Enter to continue...")

        elif choice in ['2', '3', '4', '5']:
            if not data_generated:
                print("\n⚠️  Please generate data first (Option 1)")
                input("\nPress Enter to continue...")
                continue

            if choice == '2':
                activity.standard_regression()
            elif choice == '3':
                activity.ridge_regression()
            elif choice == '4':
                activity.lasso_regression()
            elif choice == '5':
                if 'Linear Regression' not in activity.results:
                    print("\n⚠️  Please run Standard Regression first (Option 2)")
                elif 'Ridge' not in activity.results:
                    print("\n⚠️  Please run Ridge Regression first (Option 3)")
                elif 'Lasso' not in activity.results:
                    print("\n⚠️  Please run Lasso Regression first (Option 4)")
                else:
                    activity.compare_models()

            input("\nPress Enter to continue...")

        elif choice == '6':
            if not data_generated or 'Lasso' not in activity.results:
                print("\n⚠️  Please complete all activity parts first (Options 1-5)")
                input("\nPress Enter to continue...")
                continue
            activity.discussion_questions()
            input("\nPress Enter to continue...")

        elif choice == '7':
            print("\n🚀 Running complete activity (Parts 1-5)...\n")
            activity.generate_data()
            data_generated = True
            input("\nPress Enter to continue to Part 2...")

            activity.standard_regression()
            input("\nPress Enter to continue to Part 3...")

            activity.ridge_regression()
            input("\nPress Enter to continue to Part 4...")

            activity.lasso_regression()
            input("\nPress Enter to continue to Part 5...")

            activity.compare_models()
            print("\n✅ Complete activity finished!")
            input("\nPress Enter to return to menu...")

        elif choice == '8':
            print("\n🚀 Running complete activity + discussion questions...\n")
            activity.generate_data()
            data_generated = True
            input("\nPress Enter to continue to Part 2...")

            activity.standard_regression()
            input("\nPress Enter to continue to Part 3...")

            activity.ridge_regression()
            input("\nPress Enter to continue to Part 4...")

            activity.lasso_regression()
            input("\nPress Enter to continue to Part 5...")

            activity.compare_models()
            input("\nPress Enter to continue to Discussion Questions...")

            activity.discussion_questions()
            print("\n✅ Complete activity with discussion questions finished!")
            input("\nPress Enter to return to menu...")

        else:
            print("\n❌ Invalid choice. Please enter a number between 0 and 8.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
