# Regularization Activity - Streamlit App Guide 🚀

An interactive web application for learning about Ridge and Lasso regression with real-time visualizations.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run regularization_streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📚 Features

### Interactive Pages:
- **🏠 Home** - Welcome page with learning objectives and getting started guide
- **1️⃣ Data Generation** - Create synthetic dataset with adjustable parameters (sample size, noise level)
- **2️⃣ Linear Regression** - Baseline model without regularization
- **3️⃣ Ridge Regression** - L2 penalty with automatic cross-validation
- **4️⃣ Lasso Regression** - L1 penalty with automatic feature selection
- **5️⃣ Model Comparison** - Comprehensive side-by-side comparison with visualizations
- **💬 Discussion Questions** - Detailed answers to key questions with visual examples
- **🚀 Run All** - Execute complete activity at once with one click

### Rich Visualizations:
- 📊 Feature correlation heatmaps
- 📈 Coefficient comparison bar charts
- 🎯 Coefficient paths showing how coefficients shrink with increasing λ
- 🔄 Cross-validation curves for optimal parameter selection
- 📉 Performance metrics comparison (R², RMSE)
- 🎨 Comprehensive model comparison dashboard
- 🔍 Feature importance and sparsity visualizations

### Interactive Features:
- ✅ Adjustable parameters (sample size, noise level)
- ✅ Real-time model training
- ✅ Progress tracking in sidebar
- ✅ Expandable sections for discussion questions
- ✅ Tabbed visualizations for easy navigation
- ✅ Metric cards showing key statistics
- ✅ Color-coded results for easy interpretation

## 📖 How to Use

### Step-by-Step Approach:
1. **Start with Home** - Read the welcome page to understand the objectives
2. **Generate Data** - Go to "Data Generation" and create your dataset
   - Adjust sample size (default: 500)
   - Adjust noise level (default: 2.0)
   - View correlation matrix
3. **Run Models in Order**:
   - Linear Regression (baseline)
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
4. **Compare Results** - View comprehensive comparison
5. **Study Questions** - Read discussion answers with examples

### Quick Approach:
1. Navigate to **🚀 Run All**
2. Click **"▶️ Run Complete Activity"**
3. Scroll through all results

## 🎓 Learning Objectives

After using this app, you will understand:

- ✅ **How regularization works** - Prevent overfitting by penalizing large coefficients
- ✅ **Ridge vs. Lasso** - L2 penalty vs. L1 penalty, when to use each
- ✅ **Cross-validation** - How to select optimal regularization parameter (λ)
- ✅ **Feature selection** - How Lasso automatically eliminates irrelevant features
- ✅ **Multicollinearity** - How regularization handles correlated features
- ✅ **Model comparison** - How to evaluate and compare different approaches

## 💡 Key Concepts

### Regularization Types:
```
Standard Regression: Minimize MSE
Ridge (L2):          Minimize MSE + α × Σ(coefficients²)
Lasso (L1):          Minimize MSE + α × Σ|coefficients|
```

### When to Use Each:
- **Ridge**: When most features are relevant, or features are highly correlated
- **Lasso**: When you want feature selection, or only some features are relevant
- **Standard**: When you have few features and no overfitting concerns

### Parameter Selection:
- **α (lambda)**: Controls regularization strength
  - α = 0: No regularization (standard regression)
  - Small α: Weak regularization
  - Large α: Strong regularization
- **Cross-validation**: Finds optimal α that balances bias-variance tradeoff

## 📊 Understanding the Visualizations

### Feature Correlation Matrix:
- Shows how features relate to each other
- Red = positive correlation
- Blue = negative correlation
- Dark colors = strong correlation

### Coefficient Paths:
- X-axis: Regularization strength (α)
- Y-axis: Coefficient values
- Ridge: All lines shrink but stay away from zero
- Lasso: Some lines hit zero (feature elimination)

### Cross-Validation Curves:
- Shows test error vs. α
- Optimal α is where error is minimized
- Helps balance underfitting vs. overfitting

### Model Comparison Dashboard:
- **Top-left**: Coefficient values across all models
- **Top-right**: Performance metrics (R², RMSE)
- **Bottom-left**: Coefficient shrinkage (L2 norm)
- **Bottom-right**: Model sparsity (features used)

## 🎯 Discussion Questions (Preview)

The app includes comprehensive answers to:

1. **When to choose Ridge over Lasso?**
   - Feature relevance considerations
   - Correlation handling
   - Interpretability vs. performance

2. **What does zero coefficient mean?**
   - Feature exclusion
   - Automatic selection
   - Sparsity benefits

3. **Why use cross-validation?**
   - Prevents overfitting
   - Finds optimal λ
   - Uses data efficiently

4. **How does regularization help multicollinearity?**
   - Stabilizes coefficients
   - Handles correlated features
   - Ridge vs. Lasso approaches

## 💻 Alternative Versions

### Command-Line Version:
If you prefer a terminal-based interface:
```bash
python regularization_activity.py
```

### Original Activity Scripts:
The original activity guide references individual scripts:
- `01_data_generation.py`
- `02_standard_regression.py`
- `03_ridge_regression.py`
- `04_lasso_regression.py`
- `05_model_comparison.py`

The Streamlit app consolidates all of these into one interactive application!

## 🔧 Troubleshooting

### App won't start:
```bash
# Make sure Streamlit is installed
pip install streamlit

# Try running with explicit Python
python -m streamlit run regularization_streamlit_app.py
```

### Import errors:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Port already in use:
```bash
# Run on a different port
streamlit run regularization_streamlit_app.py --server.port 8502
```

### Plots not displaying:
- Make sure matplotlib is installed
- Clear browser cache
- Try a different browser

## 📝 Files Overview

```
regularization_streamlit_app.py  ← Main Streamlit application
regularization_activity.py       ← Command-line version
requirements.txt                 ← Python dependencies
STREAMLIT_APP_GUIDE.md          ← This guide
Topic3_InClass_Activity_Guide.md ← Original activity instructions
README.md                        ← Original project README
```

## 🎨 UI Features

### Sidebar Navigation:
- Select different activity parts
- Track progress with status indicators
- Easy navigation between sections

### Main Content Area:
- Clear section headers
- Expandable content sections
- Interactive controls (sliders, buttons)
- Rich visualizations
- Formatted tables

### Visual Indicators:
- ✅ Success messages (green)
- ⚠️ Warning messages (yellow)
- ℹ️ Info boxes (blue)
- 📊 Metric cards
- 🎯 Status badges

## 🚀 Tips for Best Experience

1. **Use a wide browser window** - Visualizations are best viewed in landscape
2. **Follow the numbered order** - Start with Part 1 and work through sequentially
3. **Read the discussion questions** - They provide deep insights
4. **Try different parameters** - Experiment with sample size and noise
5. **Examine visualizations carefully** - Each chart tells a story
6. **Use "Run All" for overview** - Then dive into individual sections
7. **Compare coefficient values** - Notice how Ridge vs. Lasso differ

## 🤝 Connection to Assignment

This app prepares you for the full assignment by:
- Teaching core regularization concepts
- Providing visual intuition
- Demonstrating model comparison techniques
- Explaining when to use different approaches

**In the assignment**, you'll apply these techniques to:
- Real housing price data
- Calculate AIC/BIC metrics
- Create formal reports
- Make business recommendations

## 🎓 Success Checklist

After using the app, you should be able to:

- [ ] Explain how Ridge and Lasso differ
- [ ] Interpret coefficient shrinkage plots
- [ ] Use cross-validation for parameter selection
- [ ] Choose between Ridge and Lasso for different scenarios
- [ ] Understand feature selection in Lasso
- [ ] Explain how regularization helps with multicollinearity
- [ ] Compare models using multiple metrics
- [ ] Create and interpret regularization visualizations

## 📚 Additional Resources

### Documentation:
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

### Learning Materials:
- ISLR Chapter 6: Linear Model Selection and Regularization
- StatQuest: Regularization (YouTube)
- Original activity guide: `Topic3_InClass_Activity_Guide.md`

---

## 🎉 Ready to Start?

Run the app and begin your journey into regularization:

```bash
streamlit run regularization_streamlit_app.py
```

**Enjoy learning! 🚀📊**

---

## 💬 Feedback

Found the app helpful? Have suggestions?
- What features would you like to see?
- What visualizations would help understanding?
- What explanations need clarification?

Happy learning! 🎓✨
