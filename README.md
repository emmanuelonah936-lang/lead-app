# Lead Scoring ML

A machine learning system that scores and ranks sales leads by their 
likelihood to convert, helping sales teams prioritize outreach and 
maximize conversion capture under limited calling capacity.

## What it does
- Trains and tunes classifiers on historical lead data
- Ranks leads by conversion probability (ROC-AUC ≈ 0.92)
- Optimizes the decision threshold around business cost, not a 
  default 0.5 cutoff
- Reports lift and precision@K so sales can act on the top-N leads

## Results
| Metric | Value |
|---|---|
| Test ROC-AUC | 0.920 |
| CV ROC-AUC | 0.913 |
| Recall (converters) @ 0.35 | 0.85 |
| Precision @ 0.35 | 0.78 |

At a 0.35 threshold, the model captures 85% of all converters while 
keeping precision high enough that reps trust the "hot lead" flag.
