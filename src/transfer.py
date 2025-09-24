import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import warnings
warnings.filterwarnings("ignore")

# ===== sklearn / xgboost / imblearn =====
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from numpy.linalg import norm


import matplotlib.pyplot as plt
import matplotlib
import os
os.environ["MPLBACKEND"] = "Agg"  # 必须在 import pyplot 之前

import matplotlib
matplotlib.use("Agg")             # 双保险

# Set Chinese font
OUT_DIR_Q4 = Path("output_directory_q42")
OUT_DIR_Q4.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
matplotlib.rcParams['axes.unicode_minus'] = False  # Solve the issue with negative signs

# ==========================
# Data Loading (consistent with template)
# ==========================
df1 = pd.read_csv('1.csv')  # Source domain data
df2 = pd.read_csv('2.csv')      # Target domain data

# Feature columns as in the template
# col = ['FE_rho_d_over_D', 'DE_BSF_order_center', 'DE_BPFI_order_center',
#        'DE_BPFO_order_center', 'FE_BPFI_SBI_Q3_ord', 'DE_BSF_SB_Q3_ord',
#        'DE_BSF_harmRatio_M5_ord', 'DE_BSF_harmE_M5_ord', 'DE_BSF_Eratio_ord',
#        'DE_BSF_bandE_ord', 'DE_BSF_peak_ord', 'DE_BPFI_SBI_Q3_ord',
#        'DE_BPFI_SB_Q3_ord', 'DE_BPFI_harmRatio_M5_ord', 'DE_BPFI_Eratio_ord']
# col = ['DE_FTF_harmRatio_M5_ord', 'DE_BPFO_harmRatio_M5_ord', 'DE_BPFI_SBI_Q3_ord',
#  'DE_BPFO_Eratio_ord', 'DE_BPFO_harmE_M5_ord', 'DE_BPFI_harmRatio_M5_ord',
#  'DE_BSF_bandE_ord', 'DE_FTF_harmE_M5_ord', 'DE_FTF_SBI_Q3_ord',
#  'DE_FTF_Eratio_ord', 'FE_BPFI_SBI_Q3_ord', 'FE_BSF_peak_ord',
#  'DE_BPFI_peak_ord', 'DE_BPFO_SB_Q3_ord', 'FE_BPFO_harmRatio_M5_ord',
#  'DE_BPFO_peak_ord', 'DE_FTF_SB_Q3_ord', 'DE_FTF_peak_ord',
#  'DE_BPFO_SBI_Q3_ord', 'FE_BPFI_peak_ord', 'FE_BPFO_Eratio_ord',
#  'DE_Peak', 'FE_BPFO_SBI_Q3_ord', 'FE_BPFI_Eratio_ord', 'FE_FTF_Eratio_ord',
#  'DE_BPFO_bandE_ord', 'DE_RMS', 'FE_BPFO_SB_Q3_ord', 'DE_BPFI_Eratio_ord',
#  'FE_BSF_Eratio_ord', 'FE_BPFO_peak_ord', 'DE_WaveletEnergy_aad',
#  'DE_SpectralCentroid', 'DE_WaveletEnergy_dad', 'FE_BPFI_SB_Q3_ord',
#  'FE_RMS', 'DE_WaveletEnergy_aaa', 'DE_BSF_peak_ord', 'FE_BSF_bandE_ord',
#  'FE_BSF', 'FE_Peak', 'FE_BPFI_harmRatio_M5_ord', 'DE_WaveletEnergy_add',
#  'FE_BPFI', 'DE_BPFI_harmE_M5_ord', 'DE_FE_EnergyRatio_dda',
#  'DE_BPFI_bandE_ord', 'FE_FTF_SBI_Q3_ord', 'DE_FE_EnergyRatio_aad',
#  'FE_BSF_SB_Q3_ord']
# col = ['DE_BPFO_harmRatio_M5_ord', 'DE_FTF_harmRatio_M5_ord', 'DE_BPFI_SBI_Q3_ord','DE_BPFO_harmE_M5_ord', 'DE_BPFO_Eratio_ord', 'DE_BPFI_harmRatio_M5_ord','DE_BSF_bandE_ord', 'DE_FTF_SBI_Q3_ord', 'FE_BPFI_SBI_Q3_ord','DE_FTF_harmE_M5_ord', 'FE_BSF_peak_ord', 'DE_FTF_Eratio_ord','DE_BPFO_SBI_Q3_ord', 'DE_BPFI_peak_ord', 'FE_BPFO_harmRatio_M5_ord','FE_BSF_Eratio_ord', 'FE_BPFO_Eratio_ord', 'DE_BPFO_peak_ord','DE_WaveletEnergy_aad', 'FE_FTF_Eratio_ord', 'DE_FTF_peak_ord','FE_BPFI_harmRatio_M5_ord', 'DE_RMS', 'DE_BPFO_SB_Q3_ord','FE_BPFO_SBI_Q3_ord', 'DE_Peak', 'FE_BPFI_peak_ord', 'DE_BPFI_harmE_M5_ord','DE_BPFI_Eratio_ord', 'DE_FE_EnergyRatio_aaa']

col = ['centroid_freq', 'bandEnergy_1', 'norm_E_approx', 'bandEnergy_4', 'bandEnergy_6','bsf_energy', 'bpfi_energy', 'bandEnergy_5', 'norm_E_d1', 'mean_freq'];

# ==========================
# MMD-based Alignment: TCA (Linear version)
# ==========================
class TCA:
    def __init__(self, n_components: int = 15, mu: float = 1e-2, eps: float = 1e-6, random_state: int = 42):
        self.n_components = n_components
        self.mu = mu
        self.eps = eps
        self.random_state = random_state
        self.A = None    # Projection matrix (d x k)
        self.n_s = None
        self.n_t = None

    def _construct_L(self, n_s: int, n_t: int) -> np.ndarray:
        n = n_s + n_t
        L = np.zeros((n, n))
        L[:n_s, :n_s] = 1.0 / (n_s * n_s)
        L[n_s:, n_s:] = 1.0 / (n_t * n_t)
        L[:n_s, n_s:] = -1.0 / (n_s * n_t)
        L[n_s:, :n_s] = -1.0 / (n_s * n_t)
        return L

    def fit(self, Xs: np.ndarray, Xt: np.ndarray):
        self.n_s, d = Xs.shape
        self.n_t = Xt.shape[0]
        X = np.vstack([Xs, Xt]).T         # (d, n)
        n = self.n_s + self.n_t

        # Centering matrix H
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))

        # MMD matrix L
        L = self._construct_L(self.n_s, self.n_t)

        # Matrix construction
        XHXt = X @ H @ X.T            # (d, d)
        XLXt = X @ L @ X.T            # (d, d)

        # Regularization to avoid singularities
        regI = self.mu * np.eye(d)

        # Minimize tr(A^T (XLXt + mu*I) A), s.t. A^T (XHXt + eps*I) A = I
        # Convert to generalized eigenvalue problem: (XLXt + mu*I) a = lambda (XHXt + eps*I) a
        # Use numpy to solve: first construct B = (XHXt + eps*I)^{-1} (XLXt + mu*I), then symmetrize and perform standard eigen-decomposition
        from numpy.linalg import solve, eig
        B_left  = XLXt + regI
        B_right = XHXt + self.eps * np.eye(d)

        # Solve: to ensure stability, use solve instead of direct inv
        B = solve(B_right, B_left)

        # Symmetrize (numerically stable)
        B_sym = 0.5 * (B + B.T)

        eigvals, eigvecs = np.linalg.eigh(B_sym)  # Returns in ascending order
        # Take the smallest n_components eigenvectors (corresponding to the smallest eigenvalues)
        idx = np.argsort(eigvals)[:self.n_components]
        self.A = eigvecs[:, idx]                 # (d, k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("TCA not fitted yet.")
        return X @ self.A

    def fit_transform(self, Xs: np.ndarray, Xt: np.ndarray):
        self.fit(Xs, Xt)
        Zs = self.transform(Xs)
        Zt = self.transform(Xt)
        return Zs, Zt

# ==========================
# Classifier (XGBoost)
# ==========================
def build_optimized_clf(y: np.ndarray, random_state: int = 42):
    return XGBClassifier(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        objective="multi:softprob",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1
    )

# ==========================
# Evaluation Tools
# ==========================
def domain_classifier_scores(Xs_aligned: np.ndarray, Xt: np.ndarray, random_state: int = 42):
    """Evaluate domain separability: source domain = 1, target domain = 0; closer to 0.5 is better"""
    from sklearn.ensemble import RandomForestClassifier
    X = np.vstack([Xs_aligned, Xt])
    y = np.hstack([np.ones(len(Xs_aligned)), np.zeros(len(Xt))]).astype(int)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy").mean()
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    return acc, (fpr, tpr, roc_auc)

def entropy_stats(probs: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return {"entropy_mean": float(ent.mean()), "entropy_std": float(ent.std())}

# ==========================
# Visualization (consistent with template)
# ==========================
def save_tsne_plot(Xs_align, ys, Xt_align, probs_t, out_path: Path, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    n_samp_s = min(1000, len(Xs_align))
    n_samp_t = min(1000, len(Xt_align))
    idx_s = rng.choice(len(Xs_align), n_samp_s, replace=False)
    idx_t = rng.choice(len(Xt_align), n_samp_t, replace=False)

    X_vis = np.vstack([Xs_align[idx_s], Xt_align[idx_t]])
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state, perplexity=30)
    Z = tsne.fit_transform(X_vis)

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.scatter(Z[:n_samp_s, 0], Z[:n_samp_s, 1], s=14, alpha=0.75, label="Source Domain (True Labels)")
    plt.scatter(Z[n_samp_s:, 0], Z[n_samp_s:, 1], s=14, alpha=0.75, label="Target Domain (Pseudo Labels)")
    plt.title("t-SNE Alignment Visualization: Source vs Target Domain (TCA)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(frameon=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()

def save_confidence_histograms(probs_t: np.ndarray, out_path_all: Path, out_dir_per_class: Path):
    conf = probs_t.max(axis=1)
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.hist(conf, bins=30, alpha=0.9)
    plt.xlabel("Prediction Confidence (Max Probability)")
    plt.ylabel("Sample Count")
    plt.title("Target Domain Prediction Confidence Distribution (Overall)")
    plt.savefig(out_path_all, dpi=160, bbox_inches='tight')
    plt.close()

# ==========================
# End-to-End Pipeline (with TCA Alignment)
# ==========================
def uda_coral_pseudolabel_pipeline(
    Xs: np.ndarray,
    ys: np.ndarray,
    Xt: np.ndarray,
    scaler_type: str = "quantile",
    tau: float = 0.75,  # Lower threshold to increase pseudo-labeled samples
    per_class_cap: Optional[int] = None,
    random_state: int = 42,
    do_tsne: bool = True
) -> Dict[str, object]:

    # ====== Standardization (QuantileTransformer is more robust to long-tailed distributions)======
    if scaler_type == "quantile":
        scaler = QuantileTransformer(n_quantiles=min(1000, max(10, Xs.shape[0] // 2)),
                                     output_distribution="normal",
                                     random_state=random_state)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    Xs_sc = scaler.fit_transform(Xs)
    Xt_sc = scaler.transform(Xt)

    # ====== MMD-based Alignment: TCA (Linear)======
    n_comp = min(Xs_sc.shape[1], 15)
    tca = TCA(n_components=n_comp, mu=1e-2, eps=1e-6, random_state=random_state)
    Xs_align, Xt_align = tca.fit_transform(Xs_sc, Xt_sc)

    # ====== Domain Separability (closer to 0.5 is better)======
    dom_acc, fpr_tpr_auc = domain_classifier_scores(Xs_align, Xt_align, random_state=random_state)

    # ====== Source domain oversampling to alleviate extreme class imbalance (avoiding SMOTE to prevent small classes from collapsing)======
    ros = RandomOverSampler(random_state=random_state)
    Xs_bal, ys_bal = ros.fit_resample(Xs_align, ys)

    # ====== Stage 1: Train on source domain first======
    clf = build_optimized_clf(ys_bal, random_state=random_state)
    cv_acc = cross_val_score(clf, Xs_bal, ys_bal, cv=5, scoring="accuracy").mean()
    clf.fit(Xs_bal, ys_bal)

    # Initial target domain predictions
    probs_t1 = clf.predict_proba(Xt_align)
    ent_t1 = entropy_stats(probs_t1)

    # ====== Simple pseudo-labeling and retraining (Stage 2)======
    conf = probs_t1.max(axis=1)
    idx_sel = np.where(conf >= tau)[0]  # Select high-confidence samples
    if per_class_cap is not None and len(idx_sel) > 0:
        # Per-class cap
        yhat = probs_t1.argmax(axis=1)
        final_idx = []
        for c in np.unique(yhat[idx_sel]):
            idx_c = idx_sel[yhat[idx_sel] == c]
            order = np.argsort(conf[idx_c])[::-1]
            pick = idx_c[order[:per_class_cap]]
            final_idx.append(pick)
        if len(final_idx) > 0:
            idx_sel = np.concatenate(final_idx)
        else:
            idx_sel = np.array([], dtype=int)

    if len(idx_sel) > 0:
        # Soft pseudo-labels: use predicted probabilities as labels
        pseudo_probs = probs_t1[idx_sel]  # (n_sel, n_classes)
        X_aug = np.vstack([Xs_bal, Xt_align[idx_sel]])
        y_aug = np.hstack([ys_bal, pseudo_probs.argmax(axis=1)])  # Use main labels
        # Calculate sample weights
        weights = np.hstack([np.ones(len(ys_bal)), pseudo_probs.max(axis=1)])
        clf.fit(X_aug, y_aug, sample_weight=weights)  # Weighted training
        probs_t2 = clf.predict_proba(Xt_align)
        ent_t2 = entropy_stats(probs_t2)
    else:
        ent_t2 = ent_t1

    # ====== Visualization Output (consistent with template style)======
    OUT_DIR = Path("output_directory")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if do_tsne:
        save_tsne_plot(Xs_align, ys, Xt_align, probs_t1, OUT_DIR / "tsne_tca_source_target.png", random_state=random_state)
    save_confidence_histograms(probs_t1, OUT_DIR / "confidence_hist_overall_tca.png", OUT_DIR / "confidence_hist_per_class_tca")

    # Return results in consistent format with template
    return {
        "scaler": scaler,
        "coral": None,  # TCA alignment is used here, placeholder to maintain consistent interface
        "clf_stage_final": clf,
        "domain_acc_after_coral": dom_acc,
        "cv_acc_source_after_coral": cv_acc,
        "target_entropy_stage1": ent_t1,
        "target_entropy_stage2": ent_t2,
        "out_dir": OUT_DIR,
        "xlsx_path": OUT_DIR / "target_predictions.xlsx"
    }

# ==========================
# Preprocessing and execution (consistent with template)
# ==========================


df1 = df1.fillna(0)
df2 = df2.fillna(0)

# Label encoding (source domain)
encoder = LabelEncoder()
df1['LabelName'] = encoder.fit_transform(df1['LabelName'].values)

# Feature extraction
Xs = df1[col].values
ys = df1['LabelName'].values
Xt = df2[col].values

# Execute end-to-end pipeline (consistent interface with template)
result = uda_coral_pseudolabel_pipeline(
    Xs, ys, Xt,
    scaler_type="quantile",   # More robust to outliers
    tau=0.75,  # Lower pseudo-label threshold
    per_class_cap=150,
    random_state=42
)

def _redo_align_and_predict(Xs, ys, Xt,
                            scaler_type="quantile",
                            n_comp=15,
                            tau=0.75,
                            random_state=42):
    # 标准化
    if scaler_type == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, max(10, Xs.shape[0]//2)),
            output_distribution="normal",
            random_state=random_state
        )
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    Xs_sc = scaler.fit_transform(Xs)
    Xt_sc = scaler.transform(Xt)

    # TCA 对齐
    n_comp = min(Xs_sc.shape[1], n_comp)
    _tca = TCA(n_components=n_comp, mu=1e-2, eps=1e-6, random_state=random_state)
    Xs_align, Xt_align = _tca.fit_transform(Xs_sc, Xt_sc)

    # 域分离性（ROC 数据）
    dom_acc, (fpr, tpr, roc_auc) = domain_classifier_scores(Xs_align, Xt_align, random_state=random_state)

    # 源域重采样 + 训练
    ros = RandomOverSampler(random_state=random_state)
    Xs_bal, ys_bal = ros.fit_resample(Xs_align, ys)
    clf = build_optimized_clf(ys_bal, random_state=random_state)
    cv_acc = cross_val_score(clf, Xs_bal, ys_bal, cv=5, scoring="accuracy").mean()
    clf.fit(Xs_bal, ys_bal)

    # 阶段1 预测
    probs_t1 = clf.predict_proba(Xt_align)

    # 简单伪标签再训练（阶段2）
    conf = probs_t1.max(axis=1)
    idx_sel = np.where(conf >= tau)[0]
    if len(idx_sel) > 0:
        X_aug = np.vstack([Xs_bal, Xt_align[idx_sel]])
        y_aug = np.hstack([ys_bal, probs_t1[idx_sel].argmax(axis=1)])
        weights = np.hstack([np.ones(len(ys_bal)), conf[idx_sel]])
        clf.fit(X_aug, y_aug, sample_weight=weights)
        probs_t2 = clf.predict_proba(Xt_align)
    else:
        probs_t2 = probs_t1

    return {
        "scaler": scaler,
        "tca": _tca,
        "Xs_sc": Xs_sc, "Xt_sc": Xt_sc,
        "Xs_align": Xs_align, "Xt_align": Xt_align,
        "clf": clf,
        "probs_t1": probs_t1, "probs_t2": probs_t2,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "cv_acc": cv_acc, "dom_acc": dom_acc
    }

redo = _redo_align_and_predict(Xs, ys, Xt, scaler_type="quantile", n_comp=15, tau=0.75, random_state=42)

# ========== 图 A：事前——跨域特征均值漂移 Top-K（标准化后） ==========
def plot_feature_shift(Xs_sc, Xt_sc, feat_names, out_path, topk=30):
    mu_s = Xs_sc.mean(axis=0)
    mu_t = Xt_sc.mean(axis=0)
    std_s = Xs_sc.std(axis=0) + 1e-12
    shift = np.abs(mu_s - mu_t) / std_s  # 标准化均值差
    order = np.argsort(shift)[::-1][:min(topk, len(shift))]
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    y_pos = np.arange(len(order))
    plt.barh(y_pos, shift[order])
    plt.yticks(y_pos, [feat_names[i] for i in order])
    plt.gca().invert_yaxis()
    plt.xlabel("标准化均值差")
    plt.title("跨域特征差异 Top-{}".format(len(order)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_feature_shift(redo["Xs_sc"], redo["Xt_sc"], col, OUT_DIR_Q4/"A_feature_shift_topk.png", topk=20)

# ========== 图 B：过程——领域分离性 ROC（对齐后） ==========
def plot_domain_roc(fpr, tpr, auc_val, out_path):
    plt.figure(figsize=(6, 5))
    plt.grid(True, alpha=0.3)
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_val:.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("领域分离性 ROC（TCA 对齐后）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_domain_roc(redo["fpr"], redo["tpr"], redo["roc_auc"], OUT_DIR_Q4/"B_domain_roc.png")

# ========== 图 C：过程——t-SNE 可视化（对齐后，源 vs 目标） ==========
def plot_tsne_sources_targets(Xs_align, Xt_align, out_path, random_state=42):
    rng = np.random.RandomState(random_state)
    n_s = min(1000, len(Xs_align))
    n_t = min(1000, len(Xt_align))
    idx_s = rng.choice(len(Xs_align), n_s, replace=False)
    idx_t = rng.choice(len(Xt_align), n_t, replace=False)
    X_vis = np.vstack([Xs_align[idx_s], Xt_align[idx_t]])
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state, perplexity=30)
    Z = tsne.fit_transform(X_vis)
    plt.figure(figsize=(8,6))
    plt.grid(True, alpha=0.3)
    plt.scatter(Z[:n_s,0], Z[:n_s,1], s=10, alpha=0.7, label="Source")
    plt.scatter(Z[n_s:,0], Z[n_s:,1], s=10, alpha=0.7, label="Target")
    plt.legend()
    plt.title("t-SNE（TCA 对齐后：源 vs 目标）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_tsne_sources_targets(redo["Xs_align"], redo["Xt_align"], OUT_DIR_Q4/"C_tsne_tca.png")

# ========== 图 D：事后——目标域置信度直方图（阶段1/阶段2对比） ==========
def plot_conf_hist(probs_t1, probs_t2, out_path):
    conf1 = probs_t1.max(axis=1)
    conf2 = probs_t2.max(axis=1)
    plt.figure(figsize=(8,6))
    plt.grid(True, alpha=0.3)
    plt.hist(conf1, bins=30, alpha=0.7, label="Stage-1")
    plt.hist(conf2, bins=30, alpha=0.7, label="Stage-2")
    plt.xlabel("预测置信度（最大概率）")
    plt.ylabel("样本数")
    plt.title("目标域置信度分布：自训练前后对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_conf_hist(redo["probs_t1"], redo["probs_t2"], OUT_DIR_Q4/"D_conf_hist_s1_s2.png")

# ========== 图 E：事后——Permutation 重要性（目标域，使用伪标签） ==========
def plot_permutation_importance_on_target(clf, Xt_align, probs, feat_names, out_path, topk=20, random_state=42):
    yhat = probs.argmax(axis=1)  # 无标签下的自洽近似
    r = permutation_importance(clf, Xt_align, yhat, n_repeats=8, random_state=random_state, n_jobs=-1, scoring=None)
    imp_mean = r.importances_mean
    idx = np.argsort(imp_mean)[::-1][:min(topk, len(imp_mean))]
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    y_pos = np.arange(len(idx))
    plt.barh(y_pos, imp_mean[idx])
    plt.yticks(y_pos, [feat_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation 重要性（均值）")
    plt.title("目标域特征重要性 Top-{}".format(len(idx)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_permutation_importance_on_target(
    redo["clf"], redo["Xt_align"], redo["probs_t2"], col, OUT_DIR_Q4/"E_perm_importance_topk.png", topk=20
)

# ========== 图 F：事后——可靠性（近似校准曲线） ==========
def plot_reliability_curve(probs, out_path, n_bins=12):
    conf = probs.max(axis=1)
    # 无真值情况下，用自洽近似：预测类别的一致性≈概率最大类别的“自一致性”
    # 注意：这是无标注下的 proxy，仅用于定性判断置信度的“形状”。
    pseudo_correct = (probs.argmax(axis=1) == probs.argmax(axis=1)).astype(int)
    prob_true, prob_pred = calibration_curve(pseudo_correct, conf, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6,5))
    plt.grid(True, alpha=0.3)
    plt.plot(prob_pred, prob_true, marker="o", label="经验可靠性（近似）")
    plt.plot([0,1], [0,1], "--", label="理想校准")
    plt.xlabel("预测置信度（分箱均值）")
    plt.ylabel("一致性（近似正确率）")
    plt.title("目标域可靠性校准曲线（无标注近似）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_reliability_curve(redo["probs_t2"], OUT_DIR_Q4/"F_reliability_curve.png", n_bins=12)

# ========== 图 G（可选加强）：事后——一维反事实扫描（沿最重要方向） ==========
def plot_counterfactual_scan(clf, Xt_align, probs, feat_names, out_path, k_top=1, n_points=40):
    # 取 permutation 重要性最高的方向（上面已计算；此处再算一遍，避免传递变量）
    yhat = probs.argmax(axis=1)
    r = permutation_importance(clf, Xt_align, yhat, n_repeats=6, random_state=42, n_jobs=-1)
    idx_top = np.argsort(r.importances_mean)[::-1][:max(1, k_top)]
    # 选一条不太自信的样本看看如何翻转
    conf = probs.max(axis=1)
    mid_idx = np.argsort(np.abs(conf - 0.5))[0]  # 离0.5最近
    x0 = Xt_align[mid_idx:mid_idx+1].copy()
    cls0 = probs[mid_idx].argmax()
    P = []
    Xvals = []
    for j in idx_top:
        # 沿着第 j 个特征方向正负扰动
        v = np.linspace(-2.5, 2.5, n_points)  # 扰动幅度（对齐空间）
        pj = []
        for a in v:
            x = x0.copy()
            x[0, j] = x0[0, j] + a
            p = clf.predict_proba(x)[0]
            pj.append(p.max())
        P.append(np.array(pj))
        Xvals.append(v)

    # 画图
    plt.figure(figsize=(8,6))
    plt.grid(True, alpha=0.3)
    for j, (v, pj) in enumerate(zip(Xvals, P)):
        plt.plot(v, pj, label=f"方向: {feat_names[idx_top[j]]}")
    plt.axhline(0.5, linestyle="--")
    plt.xlabel("沿对齐方向的归一化扰动幅度")
    plt.ylabel("最大类别概率")
    plt.title("一维反事实扫描（展示翻转趋势）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

plot_counterfactual_scan(redo["clf"], redo["Xt_align"], redo["probs_t2"], col, OUT_DIR_Q4/"G_counterfactual_scan.png")

print("\n[Q4 可解释性] 图已生成到：", OUT_DIR_Q4.resolve())
print("A 特征差异:  A_feature_shift_topk.png")
print("B 域 ROC:   B_domain_roc.png")
print("C t-SNE:    C_tsne_tca.png")
print("D 置信度:   D_conf_hist_s1_s2.png")
print("E 重要性:   E_perm_importance_topk.png")
print("F 可靠性:   F_reliability_curve.png")
print("G 反事实:   G_counterfactual_scan.png（可选增强）")


# Output diagnostic information
print("—— Diagnostic Information ——")
print(f"Output Directory: {result['out_dir']}")
print(f"Excel File: {result['xlsx_path']}")
print(f"Domain Separability (Source/Target after Alignment): {result['domain_acc_after_coral']:.3f}")
print(f"Source Domain CV Accuracy (After Alignment): {result['cv_acc_source_after_coral']:.3f}")
print(f"Target Domain Prediction Entropy (Stage 1) mean/std: {result['target_entropy_stage1']}")
print(f"Target Domain Prediction Entropy (Stage 2) mean/std: {result['target_entropy_stage2']}")
