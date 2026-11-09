# Assignment 03 — Scaling and Imputation of maintenance_cost_USD
# Dataset: pavement_transportation_master_dataset(4).csv
# Course methods: https://github.com/agolroo/Advanced-Statistics
#
# Leakage-safe design:
# 1) Split first (75/25).
# 2) Artificially mask 5% of the target separately in train and test.
# 3) Fit each scaler and imputer on training data only.
# 4) Transform the test set without fitting on it.
# 5) Evaluate only the artificially masked test observations.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone

DATA_PATH = "/content/pavement_transportation_master_dataset(4).csv"
RANDOM_STATE = 42
TARGET = "maintenance_cost_USD"

NUMERIC_FEATURES = [
    "survey_year", "lanes", "length", "speed_limit_kmh",
    "section_age_years", "AADT_veh_day", "truck_percent",
    "annual_rainfall_mm", "mean_temp_C", "freeze_thaw_cycles",
    "subgrade_CBR_percent", "pavement_thickness_mm",
    "density_percent", "binder_content_percent", "IRI_m_per_km",
    "rut_depth_mm", "cracking_percent", "FWD_deflection_mm",
    "skid_resistance", "avg_operating_speed_kmh", "crash_count"
]
FEATURES = NUMERIC_FEATURES + [TARGET]

# 1. Load the dataset and select quantitative variables.
df = pd.read_csv(DATA_PATH)
data = df[FEATURES].copy()

# 2. Split before masking, scaling, or imputation.
train_raw, test_raw = train_test_split(
    data, test_size=0.25, random_state=RANDOM_STATE
)
train_raw = train_raw.copy()
test_raw = test_raw.copy()

train_truth = train_raw[TARGET].copy()
test_truth = test_raw[TARGET].copy()

# 3. Randomly hide 5% of target values within each split.
rng = np.random.default_rng(RANDOM_STATE)
n_train_mask = round(0.05 * len(train_raw))
n_test_mask = round(0.05 * len(test_raw))

train_mask_idx = rng.choice(train_raw.index.to_numpy(),
                            n_train_mask, replace=False)
test_mask_idx = rng.choice(test_raw.index.to_numpy(),
                           n_test_mask, replace=False)

train_work = train_raw.copy()
test_work = test_raw.copy()
train_work.loc[train_mask_idx, TARGET] = np.nan
test_work.loc[test_mask_idx, TARGET] = np.nan

scalers = {
    "Min-Max": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler()
}

results = []

# 4. Run all 3 x 3 scaling/imputation combinations.
for scaling_name, scaler_template in scalers.items():
    scaler = clone(scaler_template)

    # IMPORTANT: fit only on training data.
    scaler.fit(train_work[FEATURES])

    train_scaled = pd.DataFrame(
        scaler.transform(train_work[FEATURES]),
        index=train_work.index, columns=FEATURES
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_work[FEATURES]),
        index=test_work.index, columns=FEATURES
    )

    target_position = FEATURES.index(TARGET)

    # Inverse transformation for the target column.
    if scaling_name == "Min-Max":
        center = scaler.data_min_[target_position]
        scale = scaler.scale_[target_position]
    elif scaling_name == "Standard":
        center = scaler.mean_[target_position]
        scale = scaler.scale_[target_position]
    else:
        center = scaler.center_[target_position]
        scale = scaler.scale_[target_position]

    def inverse_target(values):
        if scaling_name == "Min-Max":
            return np.asarray(values) / scale + center
        return np.asarray(values) * scale + center

    observed_train_target = train_scaled.loc[
        ~train_scaled[TARGET].isna(), TARGET
    ]

    for imputation_name in ["Mean", "Median", "KNN (k=5)"]:
        train_imputed = train_scaled.copy()
        test_imputed = test_scaled.copy()

        if imputation_name == "Mean":
            # Mean estimated from observed TRAIN target values only.
            fill_value = observed_train_target.mean()
            train_imputed[TARGET] = train_imputed[TARGET].fillna(fill_value)
            test_imputed[TARGET] = test_imputed[TARGET].fillna(fill_value)

        elif imputation_name == "Median":
            # Median estimated from observed TRAIN target values only.
            fill_value = observed_train_target.median()
            train_imputed[TARGET] = train_imputed[TARGET].fillna(fill_value)
            test_imputed[TARGET] = test_imputed[TARGET].fillna(fill_value)

        else:
            # KNN learns neighbours from TRAIN only.
            knn = KNNImputer(n_neighbors=5, weights="distance")
            train_array = knn.fit_transform(train_scaled)
            test_array = knn.transform(test_scaled)
            train_imputed = pd.DataFrame(
                train_array, index=train_scaled.index, columns=FEATURES
            )
            test_imputed = pd.DataFrame(
                test_array, index=test_scaled.index, columns=FEATURES
            )

        # Evaluate only hidden values in TEST, after inverse scaling.
        y_pred_scaled = test_imputed.loc[test_mask_idx, TARGET].to_numpy()
        y_pred = inverse_target(y_pred_scaled)
        y_true = test_truth.loc[test_mask_idx].to_numpy()

        results.append({
            "Scaling": scaling_name,
            "Imputation": imputation_name,
            "N_test_masked": len(y_true),
            "MAE_USD": mean_absolute_error(y_true, y_pred),
            "RMSE_USD": np.sqrt(mean_squared_error(y_true, y_pred)),
            "Bias_USD": np.mean(y_pred - y_true)
        })

results_df = pd.DataFrame(results).sort_values(
    ["MAE_USD", "RMSE_USD"]
)
print("\nComparison of 9 pipelines:")
print(results_df.round(2).to_string(index=False))

# 5. Save the comparison table.
results_df.to_csv("assignment03_results.csv", index=False)

# 6. Optional: plot MAE comparison.
plot_df = results_df.copy()
plot_df["Combination"] = (
    plot_df["Scaling"] + " + " + plot_df["Imputation"]
)
plot_df = plot_df.sort_values("MAE_USD")

plt.figure(figsize=(10, 5))
plt.barh(plot_df["Combination"], plot_df["MAE_USD"])
plt.xlabel("MAE (USD)")
plt.title("Test-set imputation error: all scaling/imputation combinations")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig("assignment03_mae_comparison.png", dpi=180)
plt.show()
