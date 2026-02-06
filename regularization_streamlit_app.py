"""
Topic 3: Regularization Activity - Streamlit App
Interactive web-based UI for exploring Ridge and Lasso regression
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Regularization Activity",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class RegularizationActivityStreamlit:
    """Streamlit version of the regularization activity"""

    def __init__(self):
        if 'data_generated' not in st.session_state:
            st.session_state.data_generated = False
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.feature_names = None
            st.session_state.true_coefficients = None
            st.session_state.models = {}
            st.session_state.results = {}

    def generate_data(self, n_samples=500, noise_level=2.0):
        """Generate synthetic dataset"""
        np.random.seed(42)

        n_features = 8
        st.session_state.feature_names = [f'Feature_{i+1}' for i in range(n_features)]

        # Generate base features with some correlation
        X = np.random.randn(n_samples, n_features)
        X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.3
        X[:, 3] = X[:, 2] + np.random.randn(n_samples) * 0.4

        # True coefficients
        st.session_state.true_coefficients = np.array([5.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

        # Generate target
        y = X @ st.session_state.true_coefficients + np.random.randn(n_samples) * noise_level

        # Split data
        st.session_state.X_train, st.session_state.X_test, \
        st.session_state.y_train, st.session_state.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        st.session_state.data_generated = True

    def plot_correlations(self):
        """Plot feature correlation matrix"""
        corr_matrix = np.corrcoef(st.session_state.X_train.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=st.session_state.feature_names,
                    yticklabels=st.session_state.feature_names,
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        return fig

    def plot_coefficients(self, coefs, title):
        """Plot coefficient comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(st.session_state.feature_names))
        width = 0.35

        ax.bar(x - width/2, st.session_state.true_coefficients, width,
               label='True', alpha=0.8, color='green')
        ax.bar(x + width/2, coefs, width, label='Estimated', alpha=0.8, color='blue')

        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(st.session_state.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        return fig

    def fit_linear_regression(self):
        """Fit standard linear regression"""
        lr = LinearRegression()
        lr.fit(st.session_state.X_train, st.session_state.y_train)

        y_train_pred = lr.predict(st.session_state.X_train)
        y_test_pred = lr.predict(st.session_state.X_test)

        st.session_state.models['Linear Regression'] = lr
        st.session_state.results['Linear Regression'] = {
            'coefficients': lr.coef_,
            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred))
        }

    def fit_ridge_regression(self, alphas_to_test=[0.1, 1.0, 10.0, 100.0]):
        """Fit ridge regression with CV"""
        alphas_cv = np.logspace(-2, 3, 100)
        ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
        ridge_cv.fit(st.session_state.X_train, st.session_state.y_train)

        y_train_pred = ridge_cv.predict(st.session_state.X_train)
        y_test_pred = ridge_cv.predict(st.session_state.X_test)

        st.session_state.models['Ridge'] = ridge_cv
        st.session_state.results['Ridge'] = {
            'coefficients': ridge_cv.coef_,
            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
            'alpha': ridge_cv.alpha_
        }

    def fit_lasso_regression(self, alphas_to_test=[0.01, 0.1, 1.0, 10.0]):
        """Fit lasso regression with CV"""
        alphas_cv = np.logspace(-3, 1, 100)
        lasso_cv = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000)
        lasso_cv.fit(st.session_state.X_train, st.session_state.y_train)

        y_train_pred = lasso_cv.predict(st.session_state.X_train)
        y_test_pred = lasso_cv.predict(st.session_state.X_test)

        n_features_used = np.sum(lasso_cv.coef_ != 0)

        st.session_state.models['Lasso'] = lasso_cv
        st.session_state.results['Lasso'] = {
            'coefficients': lasso_cv.coef_,
            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
            'alpha': lasso_cv.alpha_,
            'n_features': n_features_used
        }

    def plot_ridge_path(self):
        """Plot ridge coefficient paths"""
        alphas = np.logspace(-2, 3, 50)
        coefs = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(st.session_state.X_train, st.session_state.y_train)
            coefs.append(ridge.coef_)

        coefs = np.array(coefs)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(st.session_state.feature_names):
            ax.plot(alphas, coefs[:, i], label=name, marker='o', markersize=3)

        ax.set_xscale('log')
        ax.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontweight='bold', fontsize=12)
        ax.set_title('Ridge Coefficient Paths', fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.tight_layout()
        return fig

    def plot_lasso_path(self):
        """Plot lasso coefficient paths"""
        alphas = np.logspace(-3, 1, 50)
        coefs = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(st.session_state.X_train, st.session_state.y_train)
            coefs.append(lasso.coef_)

        coefs = np.array(coefs)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(st.session_state.feature_names):
            ax.plot(alphas, coefs[:, i], label=name, marker='o', markersize=3)

        ax.set_xscale('log')
        ax.set_xlabel('α (Regularization Strength)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontweight='bold', fontsize=12)
        ax.set_title('Lasso Coefficient Paths', fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.tight_layout()
        return fig

    def plot_cv_curves(self):
        """Plot CV curves for Ridge and Lasso"""
        alphas = np.logspace(-2, 3, 30)

        ridge_cv_scores = []
        lasso_cv_scores = []

        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            lasso = Lasso(alpha=alpha, max_iter=10000)

            ridge_scores = cross_val_score(ridge, st.session_state.X_train,
                                          st.session_state.y_train,
                                          cv=5, scoring='neg_mean_squared_error')
            lasso_scores = cross_val_score(lasso, st.session_state.X_train,
                                          st.session_state.y_train,
                                          cv=5, scoring='neg_mean_squared_error')

            ridge_cv_scores.append(-ridge_scores.mean())
            lasso_cv_scores.append(-lasso_scores.mean())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Ridge CV
        ax1.plot(alphas, ridge_cv_scores, 'b-', linewidth=2)
        optimal_idx = np.argmin(ridge_cv_scores)
        ax1.axvline(alphas[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Optimal α = {alphas[optimal_idx]:.4f}')
        ax1.scatter(alphas[optimal_idx], ridge_cv_scores[optimal_idx],
                   color='red', s=200, zorder=5, marker='*')
        ax1.set_xscale('log')
        ax1.set_xlabel('α', fontweight='bold')
        ax1.set_ylabel('CV MSE', fontweight='bold')
        ax1.set_title('Ridge Cross-Validation', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Lasso CV
        ax2.plot(alphas, lasso_cv_scores, 'b-', linewidth=2)
        optimal_idx = np.argmin(lasso_cv_scores)
        ax2.axvline(alphas[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Optimal α = {alphas[optimal_idx]:.4f}')
        ax2.scatter(alphas[optimal_idx], lasso_cv_scores[optimal_idx],
                   color='red', s=200, zorder=5, marker='*')
        ax2.set_xscale('log')
        ax2.set_xlabel('α', fontweight='bold')
        ax2.set_ylabel('CV MSE', fontweight='bold')
        ax2.set_title('Lasso Cross-Validation', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_model_comparison(self):
        """Comprehensive model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models = list(st.session_state.results.keys())

        # 1. Coefficient comparison
        ax = axes[0, 0]
        x = np.arange(len(st.session_state.feature_names))
        width = 0.2

        ax.bar(x - width*1.5, st.session_state.true_coefficients, width,
               label='True', alpha=0.8)
        ax.bar(x - width*0.5, st.session_state.results['Linear Regression']['coefficients'],
               width, label='Linear', alpha=0.8)
        ax.bar(x + width*0.5, st.session_state.results['Ridge']['coefficients'],
               width, label='Ridge', alpha=0.8)
        ax.bar(x + width*1.5, st.session_state.results['Lasso']['coefficients'],
               width, label='Lasso', alpha=0.8)

        ax.set_ylabel('Coefficient Value', fontweight='bold')
        ax.set_title('Coefficient Values Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(st.session_state.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 2. Performance metrics
        ax = axes[0, 1]
        test_r2 = [st.session_state.results[m]['test_r2'] for m in models]
        test_rmse = [st.session_state.results[m]['test_rmse'] for m in models]

        x = np.arange(len(models))
        ax2 = ax.twinx()

        ax.bar(x - 0.2, test_r2, 0.4, label='Test R²', alpha=0.8, color='skyblue')
        ax2.bar(x + 0.2, test_rmse, 0.4, label='Test RMSE', alpha=0.8, color='salmon')

        ax.set_ylabel('R² Score', fontweight='bold', color='skyblue')
        ax2.set_ylabel('RMSE', fontweight='bold', color='salmon')
        ax.set_title('Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # 3. Coefficient magnitude (L2 norm)
        ax = axes[1, 0]
        l2_norms = []
        for model_name in models:
            coefs = st.session_state.results[model_name]['coefficients']
            l2_norms.append(np.linalg.norm(coefs))

        bars = ax.bar(models, l2_norms, alpha=0.8, color=['green', 'blue', 'red'])
        ax.set_ylabel('L2 Norm', fontweight='bold')
        ax.set_title('Coefficient Shrinkage', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. Sparsity
        ax = axes[1, 1]
        n_features_used = []
        for model_name in models:
            coefs = st.session_state.results[model_name]['coefficients']
            n_features_used.append(np.sum(np.abs(coefs) > 1e-10))

        bars = ax.bar(models, n_features_used, alpha=0.8, color=['green', 'blue', 'red'])
        ax.set_ylabel('Features Used', fontweight='bold')
        ax.set_title('Model Sparsity', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0, len(st.session_state.feature_names) + 1])
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig


def main():
    # Initialize
    activity = RegularizationActivityStreamlit()

    # Title and header
    st.title("📊 Regularization Activity: Interactive Learning")
    st.markdown("### Explore Ridge and Lasso Regression with Real-Time Visualizations")

    # Sidebar
    st.sidebar.title("🎯 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Activity Part:",
        ["🏠 Home",
         "1️⃣ Data Generation",
         "2️⃣ Linear Regression",
         "3️⃣ Ridge Regression",
         "4️⃣ Lasso Regression",
         "5️⃣ Model Comparison",
         "💬 Discussion Questions",
         "🚀 Run All"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data_generated:
        st.sidebar.success("✅ Data Generated")
        if 'Linear Regression' in st.session_state.results:
            st.sidebar.success("✅ Linear Regression")
        if 'Ridge' in st.session_state.results:
            st.sidebar.success("✅ Ridge Regression")
        if 'Lasso' in st.session_state.results:
            st.sidebar.success("✅ Lasso Regression")
    else:
        st.sidebar.warning("⏳ Generate data to start")

    # HOME PAGE
    if page == "🏠 Home":
        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ## Welcome! 👋

            This interactive application will help you understand **regularization** in machine learning:

            ### 📚 What You'll Learn:
            - How **Ridge** and **Lasso** regression work
            - The effect of regularization parameter **λ (alpha)**
            - How to use **cross-validation** to select optimal parameters
            - When to use Ridge vs. Lasso
            - How regularization helps with **multicollinearity**

            ### 🎯 Activity Structure:
            1. **Data Generation** - Create synthetic dataset
            2. **Linear Regression** - Baseline model (no regularization)
            3. **Ridge Regression** - L2 penalty (shrinks coefficients)
            4. **Lasso Regression** - L1 penalty (feature selection)
            5. **Model Comparison** - Side-by-side analysis

            ### 🚀 Getting Started:
            Choose a section from the sidebar to begin, or use **"Run All"** to execute the complete activity!
            """)

        with col2:
            st.info("""
            **💡 Tips:**

            - Start with Data Generation
            - Run each section in order
            - Examine the visualizations carefully
            - Read the discussion questions
            - Try "Run All" for the full experience
            """)

            st.success("""
            **🎓 Learning Goals:**

            - Understand regularization
            - Compare Ridge vs Lasso
            - Interpret coefficients
            - Apply cross-validation
            """)

    # PART 1: DATA GENERATION
    elif page == "1️⃣ Data Generation":
        st.header("Part 1: Data Generation")
        st.markdown("---")

        st.markdown("""
        We'll create a **synthetic dataset** with known relationships to understand how regularization works.
        This controlled environment lets us see exactly what the models are doing.
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            n_samples = st.slider("Number of samples", 100, 1000, 500, 50)
        with col2:
            noise_level = st.slider("Noise level", 0.5, 5.0, 2.0, 0.5)

        if st.button("🎲 Generate Data", type="primary"):
            with st.spinner("Generating data..."):
                activity.generate_data(n_samples, noise_level)
            st.success("✅ Data generated successfully!")

        if st.session_state.data_generated:
            st.markdown("### 📊 Dataset Information")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", n_samples)
            with col2:
                st.metric("Training Samples", len(st.session_state.X_train))
            with col3:
                st.metric("Test Samples", len(st.session_state.X_test))
            with col4:
                st.metric("Features", len(st.session_state.feature_names))

            st.markdown("### 🎯 True Coefficients")
            st.markdown("These are the **ground truth** values we're trying to recover:")

            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'True Coefficient': st.session_state.true_coefficients,
                'Type': ['Important' if c != 0 else 'Noise'
                        for c in st.session_state.true_coefficients]
            })

            st.dataframe(coef_df, use_container_width=True)

            st.markdown("### 📈 Feature Correlations")
            st.markdown("Notice that some features are **correlated** (simulating real-world data):")

            fig = activity.plot_correlations()
            st.pyplot(fig)
            plt.close()

            st.info("**Key Observations:** Features 1-2 and 3-4 are highly correlated. This will help us understand how regularization handles multicollinearity.")

    # PART 2: LINEAR REGRESSION
    elif page == "2️⃣ Linear Regression":
        st.header("Part 2: Standard Linear Regression (Baseline)")
        st.markdown("---")

        if not st.session_state.data_generated:
            st.warning("⚠️ Please generate data first (Part 1)")
            return

        st.markdown("""
        First, we'll fit a **standard linear regression** model to establish a baseline.
        This shows us what happens **WITHOUT regularization**.
        """)

        if st.button("🔄 Fit Linear Regression", type="primary"):
            with st.spinner("Training model..."):
                activity.fit_linear_regression()
            st.success("✅ Model trained successfully!")

        if 'Linear Regression' in st.session_state.results:
            results = st.session_state.results['Linear Regression']

            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Train R²", f"{results['train_r2']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            with col3:
                st.metric("Train RMSE", f"{results['train_rmse']:.4f}")
            with col4:
                st.metric("Test RMSE", f"{results['test_rmse']:.4f}")

            st.markdown("### 📈 Coefficient Comparison")

            fig = activity.plot_coefficients(results['coefficients'],
                                           "Standard Linear Regression")
            st.pyplot(fig)
            plt.close()

            st.markdown("### 📋 Detailed Coefficients")
            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'True Coef': st.session_state.true_coefficients,
                'Estimated Coef': results['coefficients'],
                'Error': np.abs(st.session_state.true_coefficients - results['coefficients'])
            })
            st.dataframe(coef_df, use_container_width=True)

            st.markdown("### 💡 Key Observations")
            st.info("""
            - Without regularization, the model estimates coefficients for **ALL features**
            - Some **noise features** have non-zero coefficients (they shouldn't!)
            - The model may be **overfitting** by using irrelevant features
            - This is our **baseline** to compare Ridge and Lasso against
            """)

    # PART 3: RIDGE REGRESSION
    elif page == "3️⃣ Ridge Regression":
        st.header("Part 3: Ridge Regression (L2 Penalty)")
        st.markdown("---")

        if not st.session_state.data_generated:
            st.warning("⚠️ Please generate data first (Part 1)")
            return

        st.markdown("""
        **Ridge regression** adds an **L2 penalty** that shrinks ALL coefficients but doesn't eliminate any.

        **Formula:** Loss = MSE + α × Σ(coefficients²)
        """)

        if st.button("🔄 Fit Ridge Regression", type="primary"):
            with st.spinner("Training Ridge model with cross-validation..."):
                activity.fit_ridge_regression()
            st.success("✅ Ridge model trained successfully!")

        if 'Ridge' in st.session_state.results:
            results = st.session_state.results['Ridge']

            st.markdown("### 🎯 Optimal Parameters")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Optimal α (lambda)", f"{results['alpha']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            with col3:
                st.metric("Features Used", len(st.session_state.feature_names))

            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Train R²", f"{results['train_r2']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            with col3:
                st.metric("Train RMSE", f"{results['train_rmse']:.4f}")
            with col4:
                st.metric("Test RMSE", f"{results['test_rmse']:.4f}")

            st.markdown("### 📈 Coefficient Paths")
            st.markdown("See how coefficients shrink as α increases:")

            fig = activity.plot_ridge_path()
            st.pyplot(fig)
            plt.close()

            st.markdown("### 📋 Coefficient Values (with optimal α)")
            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'True Coef': st.session_state.true_coefficients,
                'Ridge Coef': results['coefficients']
            })
            st.dataframe(coef_df, use_container_width=True)

            st.markdown("### 💡 Key Observations")
            st.info("""
            - Ridge **shrinks all coefficients** towards zero
            - **No coefficients become exactly zero** (all features retained)
            - Larger α → more shrinkage
            - Helps with **correlated features** by shrinking them together
            - Good when you believe **most features are relevant**
            """)

    # PART 4: LASSO REGRESSION
    elif page == "4️⃣ Lasso Regression":
        st.header("Part 4: Lasso Regression (L1 Penalty)")
        st.markdown("---")

        if not st.session_state.data_generated:
            st.warning("⚠️ Please generate data first (Part 1)")
            return

        st.markdown("""
        **Lasso regression** adds an **L1 penalty** that can shrink coefficients **ALL THE WAY to zero**,
        effectively performing **automatic feature selection**.

        **Formula:** Loss = MSE + α × Σ|coefficients|
        """)

        if st.button("🔄 Fit Lasso Regression", type="primary"):
            with st.spinner("Training Lasso model with cross-validation..."):
                activity.fit_lasso_regression()
            st.success("✅ Lasso model trained successfully!")

        if 'Lasso' in st.session_state.results:
            results = st.session_state.results['Lasso']

            st.markdown("### 🎯 Optimal Parameters")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Optimal α (lambda)", f"{results['alpha']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            with col3:
                st.metric("Features Selected",
                         f"{results['n_features']}/{len(st.session_state.feature_names)}")

            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Train R²", f"{results['train_r2']:.4f}")
            with col2:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            with col3:
                st.metric("Train RMSE", f"{results['train_rmse']:.4f}")
            with col4:
                st.metric("Test RMSE", f"{results['test_rmse']:.4f}")

            st.markdown("### 📈 Coefficient Paths")
            st.markdown("See how some coefficients become exactly zero as α increases:")

            fig = activity.plot_lasso_path()
            st.pyplot(fig)
            plt.close()

            st.markdown("### 📋 Coefficient Values (with optimal α)")
            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'True Coef': st.session_state.true_coefficients,
                'Lasso Coef': results['coefficients'],
                'Selected': ['✅' if abs(c) > 1e-10 else '❌' for c in results['coefficients']]
            })
            st.dataframe(coef_df, use_container_width=True)

            st.markdown("### 💡 Key Observations")
            st.info("""
            - Lasso can shrink coefficients to **exactly zero**
            - Performs **automatic feature selection**
            - Eliminates **noise features** (as it should!)
            - Creates **sparse models** (fewer features)
            - Better for **interpretation** when you want to identify key features
            - Good when you believe **only some features are relevant**
            """)

    # PART 5: MODEL COMPARISON
    elif page == "5️⃣ Model Comparison":
        st.header("Part 5: Model Comparison")
        st.markdown("---")

        if not st.session_state.data_generated:
            st.warning("⚠️ Please generate data first (Part 1)")
            return

        if len(st.session_state.results) < 3:
            st.warning("⚠️ Please run all three models first (Parts 2, 3, and 4)")
            return

        st.markdown("""
        Let's compare all three approaches **side-by-side** to understand the tradeoffs.
        """)

        # Performance comparison table
        st.markdown("### 📊 Performance Comparison")

        comparison_df = pd.DataFrame({
            'Model': list(st.session_state.results.keys()),
            'Train R²': [st.session_state.results[m]['train_r2']
                        for m in st.session_state.results.keys()],
            'Test R²': [st.session_state.results[m]['test_r2']
                       for m in st.session_state.results.keys()],
            'Train RMSE': [st.session_state.results[m]['train_rmse']
                          for m in st.session_state.results.keys()],
            'Test RMSE': [st.session_state.results[m]['test_rmse']
                         for m in st.session_state.results.keys()],
            'Features Used': [
                len(st.session_state.feature_names),
                len(st.session_state.feature_names),
                st.session_state.results['Lasso']['n_features']
            ]
        })

        st.dataframe(comparison_df, use_container_width=True)

        # Coefficient comparison table
        st.markdown("### 📈 Coefficient Comparison")

        coef_comparison = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'True': st.session_state.true_coefficients,
            'Linear': st.session_state.results['Linear Regression']['coefficients'],
            'Ridge': st.session_state.results['Ridge']['coefficients'],
            'Lasso': st.session_state.results['Lasso']['coefficients']
        })

        st.dataframe(coef_comparison, use_container_width=True)

        # Comprehensive visualization
        st.markdown("### 📊 Visual Comparison")

        fig = activity.plot_model_comparison()
        st.pyplot(fig)
        plt.close()

        # Cross-validation curves
        st.markdown("### 🎯 Cross-Validation Curves")

        fig = activity.plot_cv_curves()
        st.pyplot(fig)
        plt.close()

        # Key takeaways
        st.markdown("### 💡 Key Takeaways")

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            **🎯 Ridge Regression:**
            - Shrinks ALL coefficients
            - Keeps all features
            - Good for correlated features
            - Stable predictions
            - Use when most features matter
            """)

        with col2:
            st.success("""
            **🎯 Lasso Regression:**
            - Can eliminate features (zero coefficients)
            - Performs feature selection
            - Creates sparse models
            - More interpretable
            - Use when few features matter
            """)

        st.info("""
        **📌 Summary:**
        - All models perform similarly on this dataset
        - Ridge/Lasso prevent overfitting
        - Lasso achieves similar performance with fewer features
        - Lasso is most interpretable (identifies truly important features)
        """)

    # DISCUSSION QUESTIONS
    elif page == "💬 Discussion Questions":
        st.header("Discussion Questions & Answers")
        st.markdown("---")

        st.markdown("""
        Here are comprehensive answers to key questions about regularization.
        """)

        # Question 1
        with st.expander("❓ Q1: When would you choose Ridge over Lasso?", expanded=True):
            st.markdown("""
            ### Answer:

            **Choose RIDGE when:**
            - You believe **MOST features are relevant** (just with small effects)
            - Features are **highly correlated** (ridge handles multicollinearity better)
            - You want to **reduce overfitting** but keep all features
            - You need **stable predictions** (ridge is less sensitive to small data changes)
            - **Example:** Predicting house prices where many features contribute

            **Choose LASSO when:**
            - You believe **ONLY SOME features are truly relevant**
            - You need **automatic feature selection**
            - You want a **sparse, interpretable model**
            - You need to identify the "most important" predictors
            - **Example:** Gene selection in biology (thousands of genes, few are relevant)

            **Key Difference:**
            - Ridge: Keeps all features, distributes weight
            - Lasso: Eliminates features, picks the best subset
            """)

            if st.session_state.data_generated and 'Lasso' in st.session_state.results:
                st.markdown("### Visual Comparison:")

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Ridge
                ridge_coefs = st.session_state.results['Ridge']['coefficients']
                axes[0].bar(st.session_state.feature_names, ridge_coefs,
                           color='skyblue', alpha=0.8)
                axes[0].set_title("Ridge: Keeps ALL features", fontweight='bold')
                axes[0].set_ylabel("Coefficient Value")
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
                axes[0].grid(axis='y', alpha=0.3)

                # Lasso
                lasso_coefs = st.session_state.results['Lasso']['coefficients']
                colors = ['green' if abs(c) > 1e-10 else 'red' for c in lasso_coefs]
                axes[1].bar(st.session_state.feature_names, lasso_coefs,
                           color=colors, alpha=0.8)
                axes[1].set_title("Lasso: Eliminates some features", fontweight='bold')
                axes[1].set_ylabel("Coefficient Value")
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
                axes[1].grid(axis='y', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Question 2
        with st.expander("❓ Q2: What does it mean when Lasso sets a coefficient to zero?"):
            st.markdown("""
            ### Answer:

            When Lasso sets a coefficient to zero, it means:

            **1. FEATURE EXCLUSION:**
            - The feature is **REMOVED** from the model entirely
            - It contributes **NOTHING** to predictions

            **2. AUTOMATIC FEATURE SELECTION:**
            - Lasso has determined this feature is not important
            - Given the penalty (λ), the cost of including it outweighs the benefit

            **3. SPARSITY:**
            - Creates simpler, more **interpretable models**
            - Easier to explain: "Only these 3 features matter"

            **4. NOT NECESSARILY USELESS:**
            - The feature might be **redundant** (correlated with another kept feature)
            - It might have very **weak predictive power**
            - With different λ, it might be included

            **In our example:** Lasso correctly identified that Features 3, 4, 6, 7, 8 are noise
            and eliminated them (their true coefficients are zero).
            """)

        # Question 3
        with st.expander("❓ Q3: Why do we use cross-validation to select λ?"):
            st.markdown("""
            ### Answer:

            We use cross-validation to select λ because:

            **1. PREVENTS OVERFITTING:**
            - Can't use training error alone (would choose λ = 0 = no regularization)
            - Can't use test set (would be "cheating" - test set should be untouched)
            - CV estimates **generalization performance honestly**

            **2. FINDS THE SWEET SPOT:**
            - Too small λ → **underfitting** (too much regularization, high bias)
            - Too large λ → **overfitting** (too little regularization, high variance)
            - CV finds the λ that **balances bias-variance tradeoff**

            **3. USES DATA EFFICIENTLY:**
            - Uses only training data (no peeking at test set)
            - Multiple folds give **robust estimate** of performance
            - Each data point used for both training and validation

            **4. OBJECTIVE SELECTION:**
            - Removes human judgment/guessing
            - **Reproducible** and principled
            - Based on actual predictive performance

            **The CV curve shows:** Test error vs. λ. The optimal λ minimizes CV error.
            """)

        # Question 4
        with st.expander("❓ Q4: How does regularization help with multicollinearity?"):
            st.markdown("""
            ### Answer:

            Regularization helps with **multicollinearity** (correlated features) by:

            **1. THE PROBLEM:**
            - When features are correlated, coefficients become **unstable**
            - Small data changes → **large coefficient changes**
            - Hard to interpret which feature is "really" important
            - **Variance of coefficient estimates explodes**

            **2. HOW RIDGE HELPS:**
            - L2 penalty **"ties together" correlated coefficients**
            - Distributes weight across correlated features
            - **Stabilizes coefficient estimates**
            - Reduces variance at cost of small bias
            - Best when ALL correlated features are truly relevant

            **3. HOW LASSO HELPS:**
            - L1 penalty **picks ONE from correlated group**, zeros others
            - Automatically chooses a representative feature
            - Creates sparse solution
            - Best when only some correlated features matter

            **4. IN OUR DATA:**
            - Feature_1 and Feature_2 are correlated (r ≈ 0.95)
            - Feature_3 and Feature_4 are correlated (r ≈ 0.93)
            - Without regularization: coefficients are unstable
            - With regularization: stable, interpretable coefficients

            **Visual Example:** The heatmap shows correlations. Ridge keeps both and shrinks together,
            while Lasso picks one and zeros the other.
            """)

    # RUN ALL
    elif page == "🚀 Run All":
        st.header("🚀 Complete Activity")
        st.markdown("---")

        st.markdown("""
        Click the button below to run the **entire activity** from start to finish.
        This will execute all parts and display comprehensive results.
        """)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("▶️ Run Complete Activity", type="primary", use_container_width=True):
                with st.spinner("Running complete activity..."):
                    # Part 1: Generate data
                    st.markdown("### 1️⃣ Generating Data...")
                    activity.generate_data()
                    st.success("✅ Data generated")

                    # Part 2: Linear regression
                    st.markdown("### 2️⃣ Fitting Linear Regression...")
                    activity.fit_linear_regression()
                    st.success("✅ Linear regression complete")

                    # Part 3: Ridge regression
                    st.markdown("### 3️⃣ Fitting Ridge Regression...")
                    activity.fit_ridge_regression()
                    st.success("✅ Ridge regression complete")

                    # Part 4: Lasso regression
                    st.markdown("### 4️⃣ Fitting Lasso Regression...")
                    activity.fit_lasso_regression()
                    st.success("✅ Lasso regression complete")

                st.success("🎉 Complete activity finished! Scroll down for results.")

        if st.session_state.data_generated and len(st.session_state.results) == 3:
            st.markdown("---")
            st.markdown("## 📊 Complete Results")

            # Summary metrics
            st.markdown("### 🎯 Performance Summary")

            comparison_df = pd.DataFrame({
                'Model': list(st.session_state.results.keys()),
                'Test R²': [st.session_state.results[m]['test_r2']
                           for m in st.session_state.results.keys()],
                'Test RMSE': [st.session_state.results[m]['test_rmse']
                             for m in st.session_state.results.keys()],
                'Features Used': [
                    len(st.session_state.feature_names),
                    len(st.session_state.feature_names),
                    st.session_state.results['Lasso']['n_features']
                ],
                'Alpha': [
                    'N/A',
                    f"{st.session_state.results['Ridge']['alpha']:.4f}",
                    f"{st.session_state.results['Lasso']['alpha']:.4f}"
                ]
            })

            st.dataframe(comparison_df, use_container_width=True)

            # Visualizations
            st.markdown("### 📈 Comprehensive Visualizations")

            tab1, tab2, tab3 = st.tabs(["Model Comparison", "Coefficient Paths", "CV Curves"])

            with tab1:
                fig = activity.plot_model_comparison()
                st.pyplot(fig)
                plt.close()

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Ridge Paths:**")
                    fig = activity.plot_ridge_path()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("**Lasso Paths:**")
                    fig = activity.plot_lasso_path()
                    st.pyplot(fig)
                    plt.close()

            with tab3:
                fig = activity.plot_cv_curves()
                st.pyplot(fig)
                plt.close()

            # Key insights
            st.markdown("### 💡 Key Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("""
                **Linear Regression:**
                - Uses all features
                - No regularization
                - Baseline model
                """)

            with col2:
                st.info("""
                **Ridge (L2):**
                - Shrinks all coefficients
                - Keeps all features
                - Good for correlated features
                """)

            with col3:
                st.info("""
                **Lasso (L1):**
                - Can eliminate features
                - Sparse solution
                - Automatic feature selection
                """)

            st.success("""
            **✅ Conclusion:** All models perform similarly, but Lasso achieves this with
            fewer features, making it the most interpretable choice for this dataset!
            """)


if __name__ == "__main__":
    main()
