# Assignment 3 — Advanced Statistics (Transportation)

## Task
Randomly mask 5% of `maintenance_cost_USD`, split the dataset into 75% training / 25% testing, compare Min-Max, Standard and Robust scaling combined with Mean, Median and KNN (k=5) imputation, and avoid data leakage.

## Files
- `Assignment_3.ipynb`: Google Colab notebook with reproducible Python analysis.
- `Assignment_03.py`: standalone Python code.
- `results/imputation_scaling_results.csv`: all nine evaluated combinations.
- `results/scaling_statistics.csv`: descriptive statistics of scaled target.

## Running
Upload `pavement_transportation_master_dataset(4).csv` to Google Colab and run `Assignment_3.ipynb`. The dataset is not duplicated here; the repository already contains a pavement dataset under a different filename, so verify that the uploaded CSV matches the analysis dataset before comparing outputs.

## Interpretation
Median imputation achieved the lowest MAE across all three scaling methods. Robust scaling + KNN achieved the lowest RMSE and relatively low bias. Mean and median results are unchanged by affine scaling after inverse transformation. The comparison uses only 30 masked test observations, so the ranking may vary with random seed.

## Method
The train/test split precedes masking and preprocessing. Scalers and imputers are fitted using training data only. Evaluation uses only artificially masked test values, in original USD units.

Course reference: https://github.com/agolroo/Advanced-Statistics
