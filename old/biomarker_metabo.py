#!/usr/bin/env python3
# biomarker_metabo.py

import sys
import mwtab                      # parses the mwTab text file
import pandas as pd               # data wrangling
import numpy as np
import matplotlib.pyplot as plt   # plotting
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ────────────────────────────────────────────────
# 1 · Helper: preprocess DataFrame
# ────────────────────────────────────────────────
def preprocess_df(df):
    # 1. Replace ±∞ with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Drop any feature that's 100% NaN up front
    df = df.dropna(axis=1, how='all')

    # 3. Drop extreme‐TIC samples by IQR
    tic = df.sum(axis=1)
    q1, q3 = np.percentile(tic, [25,75])
    iqr = q3 - q1
    mask = (tic >= q1 - 1.5*iqr) & (tic <= q3 + 1.5*iqr)
    df = df.loc[mask]

    # 4. Probabilistic Quotient Normalisation
    ref   = df.median() + 1e-12
    quot  = df.div(ref, axis=1).median(axis=1)
    df     = df.div(quot, axis=0)

    # 5. Log2 transform
    df     = np.log2(df + 1e-9)

    # 6. Drop any feature that became all‐NaN after sample drop
    df     = df.dropna(axis=1, how='all')

    # 7. Median‐impute every remaining NaN
    df     = df.fillna(df.median())

    # 8. Final check (should pass)
    assert not df.isna().any().any(), "Still got NaNs in preprocessing!"

    return df

# ────────────────────────────────────────────────
def load_mwtab(mwtab_file: str) -> pd.DataFrame:
    """
    Load an mwTab text file and return a samples×features DataFrame
    by always reading the 'Data' block.
    """
    # 1. Parse the file
    mw = next(mwtab.read_files(mwtab_file))
    ms = mw['MS_METABOLITE_DATA']

    # 2. Grab the 'Data' block
    data_block = ms['Data']

    # 3a. If it's a dict-of-dicts, orient='index'
    if isinstance(data_block, dict):
        df = pd.DataFrame.from_dict(data_block, orient='index')

    # 3b. If it's a list of per-metabolite dicts, index by 'Metabolite' then transpose
    elif isinstance(data_block, list):
        df_feat = pd.DataFrame(data_block).set_index('Metabolite')
        df      = df_feat.T

    else:
        raise ValueError(f"Unexpected type for MS_METABOLITE_DATA['Data']: {type(data_block)}")

    # 4. Force everything numeric (non-numbers → NaN)
    return df.apply(pd.to_numeric, errors='coerce')
# ────────────────────────────────────────────────
# 3 · Extract case/control labels
# ────────────────────────────────────────────────
def load_labels(mwtab_file, df, id_col='Sample ID', pheno_col='Group'):
    mw = next(mwtab.read_files(mwtab_file))
    ssf = pd.DataFrame(mw['SUBJECT_SAMPLE_FACTORS'])
    factors = pd.json_normalize(ssf['Factors'])
    ssf = ssf.drop(columns='Factors').join(factors)
    pheno = ssf.set_index(id_col)[pheno_col].reindex(df.index)
    return pheno

# ────────────────────────────────────────────────
# 4 · PCA helper
# ────────────────────────────────────────────────
def run_pca(df):
    X = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    return pcs, pca.explained_variance_ratio_

def filter_invariant_features(df, pheno):
    """
    Remove any feature (column) that has zero variance in Case or Control.
    """
    case_df    = df[pheno == 'Case']
    control_df = df[pheno == 'Control']

    # Count unique values per feature in each group
    case_uniques    = case_df.nunique()
    control_uniques = control_df.nunique()

    # Keep only those that vary in both groups
    varying_feats = case_uniques[case_uniques > 1].index\
                    .intersection(control_uniques[control_uniques > 1].index)

    dropped = df.shape[1] - len(varying_feats)
    print(f"[filter] Dropping {dropped} invariant features")
    return df[varying_feats]

# ────────────────────────────────────────────────
# 5 · Univariate t-tests + FDR
# ────────────────────────────────────────────────
def run_univariate(df, pheno):
    case_df = df[pheno=='Case']
    ctrl_df = df[pheno=='Control']
    rows = []
    for feat in df.columns:
        x, y = case_df[feat], ctrl_df[feat]
        stat, p = ttest_ind(x, y, nan_policy='omit', equal_var=False)
        rows.append((feat, stat, p, x.mean()-y.mean()))
    res = pd.DataFrame(rows, columns=['Metabolite','t_stat','p_value','mean_diff'])
    res['q_value'] = multipletests(res['p_value'], method='fdr_bh')[1]
    return res.sort_values('q_value').reset_index(drop=True)

# ────────────────────────────────────────────────
# 6 · Logistic Regression ML
# ────────────────────────────────────────────────
def run_ml(df, pheno):
    X = df.values
    y = (pheno=='Case').astype(int).values
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',   LogisticRegression(penalty='l2',solver='liblinear',random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe.fit(X, y)
    coefs = pd.DataFrame({
        'Metabolite': df.columns,
        'Coefficient': pipe.named_steps['clf'].coef_[0]
    }).assign(abs_coef=lambda d: np.abs(d['Coefficient'])
    ).sort_values('abs_coef', ascending=False)
    return aucs, coefs

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python metabo_pipeline.py <mwtab_file.txt>")
        sys.exit(1)

    mwfile = sys.argv[1]
    print("[1] Loading data…")
    df_raw = load_mwtab(mwfile)

    print(f"[2] Raw shape: {df_raw.shape}")
    print("[3] Preprocessing…")
    df = preprocess_df(df_raw)
    print(f"[4] After preprocess: {df.shape}")

    print("[5] Loading labels…")
    ph = load_labels(mwfile, df)
    print(ph.value_counts())

    print("[6] Running PCA…")
    pcs, var = run_pca(df)
    print(f"PC1 variance: {var[0]*100:.1f}%")

    print("[7] Univariate stats…")
    uni = run_univariate(df, ph)
    print(uni.head(5).to_string(index=False))

    print("[8] Running ML…")
    aucs, coefs = run_ml(df, ph)
    print("ROC AUC per fold:", np.round(aucs,3), "mean:", np.round(aucs.mean(),3))
    print("Top predictors:\n", coefs.head(5).to_string(index=False))