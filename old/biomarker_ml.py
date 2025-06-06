# 1) Import the functions
from biomarker_metabo import (
    load_mwtab,
    preprocess_df,
    load_labels,
    run_pca,
    filter_invariant_features,
    run_univariate,
    run_ml
)

# 2) Point to your mwTab file
mwfile = 'ST002091_AN003415.txt'   # or full path if needed

# 3) Load & reshape the raw data
df_raw = load_mwtab(mwfile)
print("Raw shape:", df_raw.shape)

# 4) Preprocess (drop NaNs/∞, PQN, log2, impute)
df = preprocess_df(df_raw)
print("After preprocess:", df.shape)


# 5) Load your Case/Control labels
ph = load_labels(mwfile, df)
print("Class counts:\n", ph.value_counts())

# 6) Run PCA
pcs, var = run_pca(df)
print(f"PC1 variance: {var[0]*100:.1f}%")

# 7. Drop invariant features
df = filter_invariant_features(df, ph)
print("Shape after filtering invariants:", df.shape)

# 7) Univariate t-tests + FDR
uni = run_univariate(df, ph)
print(uni.head(5).to_string(index=False))

# 8) Logistic regression CV + coefficients
aucs, coefs = run_ml(df, ph)
print("ROC AUC per fold:", aucs, "mean:", aucs.mean())
print("Top predictors:\n", coefs.head(5).to_string(index=False))