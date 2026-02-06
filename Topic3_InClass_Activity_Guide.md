# Topic 3: In-Class Activity - Introduction to Regularization
## Linear Models and Regularization

### Activity Overview
**Duration:** 50-60 minutes  
**Objective:** Understand how ridge and lasso regression work and how they differ from standard linear regression

### Learning Goals
By the end of this activity, students will be able to:
1. Fit standard linear regression, ridge regression, and lasso regression models
2. Understand how the penalty parameter (λ) affects model coefficients
3. Compare coefficient shrinkage between ridge and lasso
4. Use cross-validation to select optimal regularization parameters
5. Interpret which features are important based on regularization results

---

## Part 1: Setup and Data Generation (10 minutes)

### What We're Doing
We'll create a synthetic dataset with known relationships to understand how regularization works. This controlled environment lets us see exactly what the models are doing.

### Steps
1. Import necessary libraries
2. Create a dataset with 8 features (some important, some noise)
3. Add some correlation between features to simulate real-world data
4. Split into training and testing sets

**Use:** `01_data_generation.py`

---

## Part 2: Standard Linear Regression Baseline (10 minutes)

### What We're Doing
First, we'll fit a standard linear regression model to establish a baseline. This will show us what happens WITHOUT regularization.

### Key Observations to Make
- Which coefficients are largest?
- Do all features appear important?
- What's the model's performance on training vs. test data?

**Use:** `02_standard_regression.py`

---

## Part 3: Ridge Regression (15 minutes)

### What We're Doing
Ridge regression adds an L2 penalty that shrinks ALL coefficients but doesn't eliminate any.

### Key Questions to Answer
1. How do coefficients change as λ increases?
2. Do all coefficients shrink at the same rate?
3. What happens to correlated features?

### Experiments to Try
- Fit ridge with λ = 0.1, 1.0, 10.0, 100.0
- Compare coefficients across different λ values
- Use cross-validation to find optimal λ

**Use:** `03_ridge_regression.py`

---

## Part 4: Lasso Regression (15 minutes)

### What We're Doing
Lasso regression adds an L1 penalty that can shrink coefficients ALL THE WAY to zero, effectively performing feature selection.

### Key Questions to Answer
1. Which coefficients become exactly zero?
2. How does this differ from ridge regression?
3. Which features does lasso consider "important"?

### Experiments to Try
- Fit lasso with λ = 0.01, 0.1, 1.0, 10.0
- Identify which features are eliminated
- Use cross-validation to find optimal λ

**Use:** `04_lasso_regression.py`

---

## Part 5: Model Comparison (10 minutes)

### What We're Doing
Compare all three approaches side-by-side to understand the tradeoffs.

### What to Compare
1. **Coefficient values** - How do they differ across models?
2. **Number of features used** - Which model is most parsimonious?
3. **Performance metrics** - R², RMSE on training and test sets
4. **Interpretability** - Which model is easiest to explain?

**Use:** `05_model_comparison.py`

---

## Discussion Questions

1. **When would you choose ridge over lasso?**
   - Hint: Think about interpretability vs. performance

2. **What does it mean when lasso sets a coefficient to zero?**
   - Hint: Consider feature importance

3. **Why do we use cross-validation to select λ?**
   - Hint: Think about overfitting

4. **How does regularization help with multicollinearity?**
   - Hint: Consider correlated features

---

## Connection to the Assignment

This in-class activity prepares you for the full assignment where you will:
- Work with a real housing price dataset
- Apply these same techniques
- Calculate AIC and BIC for model selection
- Make recommendations based on comprehensive analysis

The main differences in the assignment:
- Real data (not synthetic)
- More formal evaluation metrics (AIC, BIC)
- Requirement to explain business implications
- Video presentation of your work

---

## Tips for Success

1. **Run the code incrementally** - Don't just execute entire scripts
2. **Examine outputs carefully** - The numbers tell a story
3. **Make predictions** - Before running code, predict what will happen
4. **Compare results** - Always compare back to your baseline
5. **Ask "why"** - If something surprises you, investigate why

---

## Additional Resources

- Scikit-learn Ridge documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-learn Lasso documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- Cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html
