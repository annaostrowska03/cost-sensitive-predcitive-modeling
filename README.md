# Cost-Sensitive Predictive Modeling for Targeted Marketing

## Project Overview
The objective of this project is to develop a high-performance, cost-effective predictive model to identify customers most likely to convert on a marketing offer. 

In modern marketing, sending unsolicited offers to uninterested clients creates "Marketing Fatigue" and degrades brand value. This project approaches customer attention as a finite resource and introduces two main constraints:
1. **Data Investment Cost:** Every feature (variable) used in the model carries an acquisition cost. The model must perform aggressive feature selection to use only the most valuable data points.
2. **Targeting Efficiency:** The campaign is limited to at most 1,000 customers. The model must rank customers by conversion probability and only target profitable cases.

## Team:
* [Anna Ostrowska](https://github.com/annaostrowska03)
* [Gabriela Majstrak](https://github.com/GabrielaMajstrak)
* [Igor Lechoszest](https://github.com/IgorLechoszest)

## Alternative Optimization Pipelines
Additional standalone approaches are available in separate scripts:

1. `python -m src.run_optimization_threshold_search`
   Approach: tuned + calibrated `HistGradientBoosting` with threshold optimization directly for profit.
   Output files: `{TEAM_NAME}_threshold_obs.txt`, `{TEAM_NAME}_threshold_vars.txt`

2. `python -m src.run_optimization_ensemble`
   Approach: weighted soft-voting ensemble (`HGB + RandomForest + LogisticRegression`) with CV-based threshold search.
   Output files: `{TEAM_NAME}_ensemble_obs.txt`, `{TEAM_NAME}_ensemble_vars.txt`

3. `python -m src.run_optimization_interactions`
   Approach: free interaction features (`PolynomialFeatures`) and model selection under profit metric.
   Output files: `{TEAM_NAME}_interactions_obs.txt`, `{TEAM_NAME}_interactions_vars.txt`

You can set a team prefix before running:
`$env:TEAM_NAME="123456_98765_98764"` (PowerShell)

*This project was created as part of the Advanced Machine Learning course (Data Science @ WUT)
