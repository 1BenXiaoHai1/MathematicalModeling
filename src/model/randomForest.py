# -*- coding: utf-8 -*-
"""
RF full features + TopK(20/30/40/50, then step by 10) until matching full performance.
Exports: importance CSVs, all models (Pipeline), reports, results table, and predictions.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
CSV_PATH = r"feature(2).csv"  # 改成你的路径
OUT_DIR  = Path("rf_perfile_topk_outputs6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ESTIMATORS = 300
RANDOM_STATE = 42
TEST_SIZE = 0.2

INITIAL_TOPK = [10 , 20, 30, 40, 50, 60, 70]
STEP = 10
MAX_K_CAP = 4000  # 安全上限

# ---------------- Helpers ----------------
def pick_label_col(df: pd.DataFrame) -> str:
    preferred = ["label_cls","cls","label","fault_type","fault","status","state","class","target","y","diagnosis","category"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        s = df[c].dropna()
        if not np.issubdtype(s.dtype, np.number) and 2 <= s.nunique() <= 20:
            return c
    return df.columns[-1]

def build_preprocessor(numeric_cols, categorical_cols):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric_cols),
                              ("cat", cat, categorical_cols)])

def train_and_eval(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    rep = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1m, rep, cm, y_pred

def save_model_and_columns(pipe, path_pkl, numeric_cols, categorical_cols, path_json):
    joblib.dump(pipe, path_pkl)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"numeric_cols": list(numeric_cols),
                   "categorical_cols": list(categorical_cols)}, f, ensure_ascii=False, indent=2)

# ---------------- Load ----------------
df = pd.read_csv(CSV_PATH)
label_col = pick_label_col(df)
y = df[label_col]

# 去掉 label + 前9列基础信息
base_info_cols = df.columns[:3]
drop_cols = [label_col] + list(base_info_cols)
X = df.drop(columns=drop_cols)

numeric_cols = [c for c in X.columns if np.issubdtype(df[c].dropna().dtype, np.number)]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---------------- Full model ----------------
pre_full = build_preprocessor(numeric_cols, categorical_cols)
rf_full = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
    max_depth=None, min_samples_split=2
)
pipe_full = Pipeline([("pre", pre_full), ("clf", rf_full)])

acc_full, f1_full, rep_full, cm_full, y_pred_full = train_and_eval(pipe_full, X_train, y_train, X_test, y_test)

(Path(OUT_DIR/"report_full.txt")).write_text(rep_full, encoding="utf-8")
save_model_and_columns(pipe_full, OUT_DIR/"model_full.pkl", numeric_cols, categorical_cols, OUT_DIR/"features_full.json")

# 保存预测值
pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_full})
pred_df.to_csv(OUT_DIR/"predictions_full.csv", index=False, encoding="utf-8-sig")

# ---------------- Importance (transformed) ----------------
feature_names = numeric_cols.copy()
ohe_names = []
if len(categorical_cols) > 0:
    ohe = pipe_full.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + ohe_names

importances = pipe_full.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})\
            .sort_values("importance", ascending=False).reset_index(drop=True)
imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
imp_df["cum_importance"] = imp_df["importance_norm"].cumsum()
imp_df.to_csv(OUT_DIR/"importance_transformed.csv", index=False, encoding="utf-8-sig")

# ---------------- Aggregate to original columns ----------------
agg = {}
for i, col in enumerate(numeric_cols):
    agg[col] = agg.get(col, 0.0) + float(importances[i])
offset = len(numeric_cols)
for j, name in enumerate(ohe_names):
    for orig_col in categorical_cols:
        if name.startswith(orig_col + "_"):
            agg[orig_col] = agg.get(orig_col, 0.0) + float(importances[offset + j])
            break

agg_df = pd.DataFrame(sorted(agg.items(), key=lambda x: x[1], reverse=True),
                      columns=["original_feature","importance"])
agg_df["importance_norm"] = agg_df["importance"] / agg_df["importance"].sum()
agg_df["cum_importance"] = agg_df["importance_norm"].cumsum()
agg_df.to_csv(OUT_DIR/"importance_aggregated_original.csv", index=False, encoding="utf-8-sig")

# ---------------- TopK loop until matching full ----------------
def build_selected_pipe_from_transformed_toplist(top_list, numeric_cols, categorical_cols):
    sel_numeric = [c for c in top_list if c in numeric_cols]
    sel_categorical = []
    for c in categorical_cols:
        if any(f.startswith(c + "_") for f in top_list):
            sel_categorical.append(c)
    if len(sel_numeric) == 0 and len(sel_categorical) == 0:
        raise RuntimeError(f"Top list 映射为空：top_list中前缀未命中任何原始列，样例：{top_list[:5]}")
    pre_sel = build_preprocessor(sel_numeric, sel_categorical)
    rf_sel = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1,
        max_depth=None, min_samples_split=2
    )
    return Pipeline([("pre", pre_sel), ("clf", rf_sel)]), sel_numeric, sel_categorical

rows = [{
    "scheme":"Full features", "k_or_thr":"-",
    "kept_features": f"{len(numeric_cols)} num + {len(categorical_cols)} cat",
    "accuracy":acc_full, "macro_f1":f1_full,
    "model_file":"model_full.pkl", "report":"report_full.txt",
    "feature_list":"features_full.json"
}]

acc_full_r = round(acc_full, 4)
f1_full_r  = round(f1_full, 4)

tested_K = set()
first_match_k = None

K_list = INITIAL_TOPK[:]
cap = len(feature_names)

print(">>> INITIAL_TOPK =", K_list, "| cap =", cap)

while True:
    progressed = False
    for K in K_list:
        if K in tested_K or K > cap:
            continue
        progressed = True
        tested_K.add(K)

        topK_feats = imp_df.head(K)["feature"].tolist()
        print(f"[TopK] 训练 K={K}，样本特征数={len(topK_feats)}")

        pipe_sel, sel_num, sel_cat = build_selected_pipe_from_transformed_toplist(
            topK_feats, numeric_cols, categorical_cols
        )
        print(f"[TopK] K={K} -> 原始列映射：{len(sel_num)} 数值 + {len(sel_cat)} 类别")

        acc, f1m, rep, cm, y_pred = train_and_eval(pipe_sel, X_train, y_train, X_test, y_test)

        tag = f"top{K}"
        (OUT_DIR/f"report_{tag}.txt").write_text(rep, encoding="utf-8")
        joblib.dump(pipe_sel, OUT_DIR / f"model_{tag}.pkl")
        with open(OUT_DIR / f"features_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"numeric_cols": sel_num, "categorical_cols": sel_cat},
                      f, ensure_ascii=False, indent=2)

        # 保存预测值
        pred_top_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        pred_top_df.to_csv(OUT_DIR/f"predictions_{tag}.csv", index=False, encoding="utf-8-sig")

        print(f"[TopK] 已保存 {tag}: model_{tag}.pkl / features_{tag}.json / report_{tag}.txt / predictions_{tag}.csv")
        print(f"[TopK] K={K} -> acc={acc:.4f}, f1={f1m:.4f}")

        rows.append({
            "scheme":"TopK","k_or_thr":tag,
            "kept_features": f"{len(sel_num)} num + {len(sel_cat)} cat",
            "accuracy":acc, "macro_f1":f1m,
            "model_file": f"model_{tag}.pkl",
            "report": f"report_{tag}.txt",
            "feature_list": f"features_{tag}.json"
        })

        if round(acc,4) == acc_full_r and round(f1m,4) == f1_full_r and first_match_k is None:
            first_match_k = K
            print(f"[TopK] 首次匹配全特征：K={K}")

    if first_match_k is not None:
        break
    if not progressed:
        break
    next_K = (K_list[-1] // STEP + 1) * STEP
    if next_K > cap:
        break
    K_list.append(next_K)
    print("扩展 K_list ->", K_list)

# 保存结果表
res_df = pd.DataFrame(rows)
res_df.to_csv(OUT_DIR/"results_topk.csv", index=False, encoding="utf-8-sig")

print("Full:", f"acc={acc_full:.4f}, f1={f1_full:.4f}")
print("First matching TopK:", first_match_k)
print("Artifacts saved to:", OUT_DIR.resolve())
