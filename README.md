# Topic 3 In-Class Activity: Linear Models and Regularization

## Overview
This in-class activity is designed to prepare you for the Topic 3 assignment on regularization techniques. You will learn through hands-on experimentation with synthetic data where you know the "ground truth," making it easier to understand what each technique is doing.

---

## Learning Objectives

By completing this activity, you will be able to:

1. ✅ **Understand regularization fundamentals**
   - Why we need regularization
   - How it prevents overfitting
   - The difference between L1 and L2 penalties

2. ✅ **Apply regularization techniques**
   - Fit Ridge regression models
   - Fit Lasso regression models
   - Use cross-validation to select optimal λ

3. ✅ **Compare model performance**
   - Evaluate using R², RMSE
   - Interpret coefficient shrinkage
   - Understand feature selection

4. ✅ **Make informed modeling decisions**
   - Choose appropriate techniques for different scenarios
   - Balance performance vs. interpretability
   - Prepare for the full assignment

---

## File Structure

```
Topic3_InClass_Activity_Guide.md  ← Start here! Activity instructions
01_data_generation.py             ← Step 1: Create synthetic dataset
02_standard_regression.py         ← Step 2: Baseline (no regularization)
03_ridge_regression.py            ← Step 3: L2 regularization
04_lasso_regression.py            ← Step 4: L1 regularization
05_model_comparison.py            ← Step 5: Comprehensive comparison
README.md                         ← This file
```

---

## How to Use This Activity

### Option 1: Follow Along in Class (Recommended)
1. Read `Topic3_InClass_Activity_Guide.md` for context
2. Instructor will walk through each script
3. Run code sections as demonstrated
4. Discuss results with classmates
5. Complete experiments at the end of each script

### Option 2: Self-Paced Learning
1. Start with `Topic3_InClass_Activity_Guide.md`
2. Work through scripts in order (01 → 05)
3. Read ALL comments carefully - they explain WHY, not just HOW
4. Run experiments at the end of each script
5. Answer discussion questions before moving on

---

## Estimated Time

| Script | Description | Time |
|--------|-------------|------|
| 01 | Data Generation | 10 min |
| 02 | Standard Regression | 10 min |
| 03 | Ridge Regression | 15 min |
| 04 | Lasso Regression | 15 min |
| 05 | Model Comparison | 10 min |
| **Total** | **Complete Activity** | **60 min** |

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of linear regression
- Familiarity with NumPy, pandas, matplotlib
- Basic understanding of overfitting

### Required Software
```python
# All required libraries (should be installed in your environment)
numpy
pandas
matplotlib
scikit-learn
seaborn
```

### Installation Check
Run this in a Python environment:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
print("All required libraries are installed! ✓")
```

---

## Script Descriptions

### 01_data_generation.py
**What it does:**
- Creates a controlled synthetic dataset
- 8 features: 3 important, 5 noise
- Adds correlation between features
- Splits into train/test sets
- Standardizes features

**Why it matters:**
You know the TRUE relationships, so you can evaluate whether models recover them correctly.

**Key outputs:**
- `data_for_regularization.npz` (data file)
- `01_feature_relationships.png` (visualization)

---

### 02_standard_regression.py
**What it does:**
- Fits standard linear regression (NO regularization)
- Evaluates performance on train/test sets
- Analyzes coefficient estimates
- Identifies potential overfitting

**Why it matters:**
Establishes a baseline to compare against regularized models.

**Key outputs:**
- `02_standard_regression_results.npz`
- `02_standard_regression_coefficients.png`
- `02_standard_regression_predictions.png`
- `02_standard_regression_residuals.png`

**Key observations to make:**
- Does it assign coefficients to noise features?
- Is there evidence of overfitting?
- How accurate are the coefficient estimates?

---

### 03_ridge_regression.py
**What it does:**
- Applies Ridge regression (L2 regularization)
- Tests different λ values
- Uses cross-validation to find optimal λ
- Compares to standard regression

**Why it matters:**
Ridge shrinks coefficients to reduce overfitting while keeping all features.

**Key outputs:**
- `03_ridge_regression_results.npz`
- `03_ridge_coefficient_paths.png`
- `03_ridge_performance_vs_lambda.png`
- `03_ridge_vs_standard_coefficients.png`

**Key observations to make:**
- How do coefficients shrink as λ increases?
- What's the optimal λ?
- Does it improve generalization?

---

### 04_lasso_regression.py
**What it does:**
- Applies Lasso regression (L1 regularization)
- Tests different λ values
- Performs automatic feature selection
- Uses cross-validation to find optimal λ

**Why it matters:**
Lasso can set coefficients to EXACTLY zero, eliminating unimportant features.

**Key outputs:**
- `04_lasso_regression_results.npz`
- `04_lasso_coefficient_paths.png`
- `04_lasso_feature_selection_heatmap.png`
- `04_three_way_comparison.png`

**Key observations to make:**
- Which features does Lasso eliminate?
- Does it correctly identify noise features?
- How does it compare to Ridge?

---

### 05_model_comparison.py
**What it does:**
- Comprehensive comparison of all three approaches
- Creates detailed visualizations
- Provides decision framework
- Makes recommendations

**Why it matters:**
Synthesizes everything you've learned into actionable insights.

**Key outputs:**
- `05_comprehensive_comparison.png`
- `05_model_comparison.csv`
- `05_coefficient_comparison.csv`

**Key observations to make:**
- Which model performed best overall?
- When would you choose each approach?
- How do you balance performance vs. interpretability?

---

## Key Concepts Explained

### What is Regularization?
Adding a penalty term to the loss function to discourage complex models:
- **Standard regression:** Minimize `sum(errors²)`
- **Ridge:** Minimize `sum(errors²) + λ·sum(coefficients²)`
- **Lasso:** Minimize `sum(errors²) + λ·sum(|coefficients|)`

### Why Standardize Features?
Regularization penalizes coefficient magnitude, but magnitude depends on feature scale!
- Feature in dollars: coefficient might be 0.0001
- Same feature in cents: coefficient might be 10
- Solution: Standardize all features to mean=0, std=1

### Lambda (λ) Parameter
Controls regularization strength:
- **λ = 0:** No penalty (standard regression)
- **Small λ:** Weak penalty, slight shrinkage
- **Large λ:** Strong penalty, heavy shrinkage
- **Too large λ:** Underfitting (model too simple)

### Cross-Validation for λ Selection
- Can't use training error (always prefers λ=0)
- Can't use test set (that's for final evaluation)
- Solution: Cross-validation on training set
- Finds λ that generalizes best

---

## Common Questions

### Q: Why do we create synthetic data?
**A:** Because we know the TRUE relationships! This lets us evaluate whether models correctly identify important features and recover true coefficients. With real data, we never know the ground truth.

### Q: What if my Ridge model doesn't improve over Standard?
**A:** This can happen if:
1. Standard regression isn't overfitting much
2. All features are actually important
3. The dataset is very clean
In the assignment with real data, regularization will likely help more.

### Q: Why doesn't Ridge eliminate features?
**A:** The L2 penalty (sum of squares) can shrink coefficients very small, but mathematically can't reach exactly zero. Only L1 (absolute value) penalty can hit zero exactly.

### Q: How do I choose between Ridge and Lasso?
**A:** Consider:
- **Use Ridge** if you think most features are somewhat important
- **Use Lasso** if you think many features are noise
- **Use Ridge** for prediction focus
- **Use Lasso** for interpretation focus

### Q: What's Elastic Net?
**A:** A combination of Ridge and Lasso:
- Penalty: `λ₁·sum(|coefficients|) + λ₂·sum(coefficients²)`
- Balances feature selection with stability
- We'll cover in advanced topics if time permits

---

## Troubleshooting

### Import Errors
```python
# If you get import errors, install missing packages:
pip install numpy pandas matplotlib scikit-learn seaborn
```

### Convergence Warnings (Lasso)
```python
# If Lasso doesn't converge, increase max_iter:
lasso_model = Lasso(alpha=alpha, max_iter=20000)
```

### File Not Found Errors
Make sure you:
1. Run scripts in order (01 → 05)
2. All scripts are in the same directory
3. Don't delete .npz files between scripts

### Plots Not Showing
```python
# Add at the end of scripts if needed:
plt.show()
```

---

## Connection to Assignment

This activity prepares you for the full assignment where you will:

### Similarities
- Apply Ridge and Lasso regression
- Use cross-validation for λ selection
- Compare model performance
- Interpret coefficients

### Differences
- **Real data** (housing prices, not synthetic)
- **Additional metrics** (AIC, BIC for model selection)
- **Assumption checking** (residual analysis)
- **Feature engineering** (suggesting new variables)
- **Video presentation** (explaining your work)

### Assignment Tips
1. **Use this code as a template** - adapt it for housing data
2. **Reference these visualizations** - create similar plots
3. **Follow the same workflow** - baseline → Ridge → Lasso → compare
4. **Explain in your video** - show you understand WHY, not just HOW

---

## Experiments to Try

After completing the basic activity, try these extensions:

### Beginner Experiments
1. Change noise level in script 01
   - How does it affect model performance?
   
2. Try different λ ranges
   - What happens with very small/large values?

3. Add more features
   - Do models still identify important ones?

### Intermediate Experiments
4. Change which features are important
   - Does Lasso still work correctly?

5. Add more correlation
   - How does it affect Ridge vs. Lasso?

6. Try different train/test splits
   - How stable are the results?

### Advanced Experiments
7. Implement Elastic Net
   - Combine Ridge and Lasso penalties

8. Try different metrics
   - MAE instead of RMSE

9. Visualize regularization paths
   - Plot coefficient evolution more detailed

---

## Additional Resources

### Scikit-learn Documentation
- [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

### Textbook Chapters
- ISLR Chapter 6: Linear Model Selection and Regularization
- ESL Chapter 3: Linear Methods for Regression

### Videos (if you want more explanation)
- StatQuest: Regularization (YouTube)
- 3Blue1Brown: Gradient Descent and Regularization

---

## Success Checklist

Before moving to the assignment, make sure you can:

- [ ] Explain why regularization helps prevent overfitting
- [ ] Describe the difference between L1 and L2 penalties
- [ ] Interpret coefficient shrinkage plots
- [ ] Use cross-validation to select λ
- [ ] Compare models using multiple metrics
- [ ] Explain when to use Ridge vs. Lasso
- [ ] Create clear visualizations
- [ ] Write interpretations of results

---

## Getting Help

### During Class
1. Ask instructor to explain confusing concepts
2. Discuss with classmates
3. Work through examples together

### After Class
1. Review script comments carefully
2. Try the experiments
3. Post questions in discussion forum
4. Attend office hours
5. Form study groups

---

## Final Notes

**This is a LEARNING activity, not graded!**
- Experiment freely
- Make mistakes
- Ask questions
- Understand concepts deeply

**The goal is preparation for success on the assignment**
- Where accuracy matters
- Where you'll be graded
- Where you'll demonstrate mastery

Good luck! 🚀

---

## Feedback

Found this activity helpful? Have suggestions for improvement?
Let your instructor know!

- What worked well?
- What was confusing?
- What would you add/change?

Your feedback helps improve the activity for future students.
