# Lead Scoring ML

A machine learning system that scores and ranks sales leads by their 
likelihood to convert, helping sales teams prioritize outreach and 
maximize conversion capture under limited calling capacity.

## What it does
- Trains and compares three classifiers — Logistic Regression, 
  Random Forest, and XGBoost — on historical lead data
- Ranks leads by conversion probability (best model ROC-AUC ≈ 0.91)
- Optimizes the decision threshold around business cost, not a 
  default 0.5 cutoff
- Reports lift and precision@K so sales can act on the top-N leads

## Model comparison

| Model | Test Accuracy | ROC-AUC | Recall (class 1) | Precision (class 1) |
|---|---|---|---|---|
| Logistic Regression | 0.83 | 0.903 | 0.75 | 0.81 |
| Random Forest | 0.83 | 0.908 | 0.74 | 0.83 |
| XGBoost | 0.84 | 0.910 | 0.78 | 0.81 |

All three models perform comparably, with XGBoost edging ahead on 
ROC-AUC and recall. The tight spread across models suggests the 
signal in the data is strong and the choice of algorithm matters 
less than threshold selection.

## Results (best model — XGBoost)

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.910 |
| CV ROC-AUC | 0.913 |
| Recall (converters) @ 0.35 | 0.85 |
| Precision @ 0.35 | 0.78 |

At a 0.35 threshold, the model captures 85% of all converters while 
keeping precision high enough that reps trust the "hot lead" flag. 
The threshold was lowered from the default 0.5 because a missed lead 
costs far more than a wasted call in this domain.
