import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from ml_engine import DataMedicEngine

st.set_page_config(page_title="DataMedic - AI Data Quality Platform", layout="wide")
st.title("🩺 DataMedic: Interactive Human-in-the-Loop Data Diagnostics")
st.markdown(
    "Upload any classification CSV dataset, review AI-flagged issues row-by-row, **check or uncheck individual fixes**, "
    "and visually verify how your repairs impact feature distributions and held-out model performance."
)

# 1. User CSV Upload
st.sidebar.header("1. Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded_file:
    st.info("👈 Please upload a CSV dataset in the sidebar to begin diagnostics.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write(f"### 📄 Dataset Preview (`{uploaded_file.name}` — {len(df)} rows, {len(df.columns)} columns)")
st.dataframe(df.head(5), use_container_width=True)

# 2. Configure Target & Sensitivity
st.sidebar.header("2. Configure Target & Columns")
target_col = st.sidebar.selectbox("Select Target Column (to predict):", df.columns, index=len(df.columns) - 1)
available_ignore = [c for c in df.columns if c != target_col]
ignore_cols = st.sidebar.multiselect("Ignore ID / Irrelevant Columns (Optional):", available_ignore)

st.sidebar.header("3. ML Detection Sensitivity")
conf_thresh = st.sidebar.slider("Label Error Confidence Threshold:", 0.55, 0.95, 0.75, 0.05)
outlier_rate = st.sidebar.slider("Expected Outlier Proportion:", 0.01, 0.15, 0.03, 0.01)

# Clean rows missing the target and verify valid classification target
df = df.dropna(subset=[target_col]).reset_index(drop=True)
class_counts = df[target_col].value_counts()

if len(class_counts) < 2 or class_counts.min() < 2:
    st.error(f"Target column '{target_col}' must have at least 2 classes with at least 2 records per class.")
    st.stop()

# Strict Train/Test Split BEFORE Cleaning to prevent Data Leakage
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[target_col])
st.caption(f"🔒 **Experiment Guardrail:** Split into **{len(train_df)} Training rows** (inspected & cleaned) and **{len(test_df)} Held-Out Test rows** (locked for evaluation).")

# Run DataMedic Engine on Training Split Only
engine = DataMedicEngine(train_df, target_col=target_col, drop_cols=ignore_cols)
leakage_df = engine.detect_data_leakage()

leaked_cols = list(leakage_df["Feature"].values) if not leakage_df.empty else []
non_leaked_cols = [c for c in engine.X_encoded.columns if c not in leaked_cols]

label_issues_df = engine.detect_label_errors(clean_feature_cols=non_leaked_cols, confidence_threshold=conf_thresh)
outlier_issues_df = engine.detect_outliers(clean_feature_cols=non_leaked_cols, contamination=outlier_rate)
dup_issues_df = engine.detect_duplicates(clean_feature_cols=non_leaked_cols)

# Insert 'Apply_Fix' boolean column at the front of each non-empty table
for issue_df in [leakage_df, label_issues_df, outlier_issues_df, dup_issues_df]:
    if not issue_df.empty and "Apply_Fix" not in issue_df.columns:
        issue_df.insert(0, "Apply_Fix", True)

# Diagnostic Metric Cards
st.subheader("🔍 2. Review & Approve Fixes Row-by-Row")
st.markdown("Click inside any tab below to **uncheck** specific rows or features you do **not** want DataMedic to modify.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚨 Leaked Features", len(leakage_df))
c2.metric("🏷️ Suspected Label Errors", len(label_issues_df))
c3.metric("📈 Multivariate Outliers", len(outlier_issues_df))
c4.metric("👯 Near-Duplicates", len(dup_issues_df))

tab1, tab2, tab3, tab4 = st.tabs(["🚨 Data Leakage", "🏷️ Label Errors", "📈 Outliers", "👯 Near-Duplicates"])

with tab1:
    if not leakage_df.empty:
        edited_leakage = st.data_editor(
            leakage_df,
            column_config={"Apply_Fix": st.column_config.CheckboxColumn("Drop Feature?", default=True)},
            disabled=[c for c in leakage_df.columns if c != "Apply_Fix"],
            hide_index=True,
            use_container_width=True,
            key="editor_leakage"
        )
    else:
        st.info("No data leakage detected.")
        edited_leakage = pd.DataFrame()

with tab2:
    if not label_issues_df.empty:
        edited_labels = st.data_editor(
            label_issues_df,
            column_config={"Apply_Fix": st.column_config.CheckboxColumn("Relabel Row?", default=True)},
            disabled=[c for c in label_issues_df.columns if c != "Apply_Fix"],
            hide_index=True,
            use_container_width=True,
            key="editor_labels"
        )
    else:
        st.info("No label errors detected.")
        edited_labels = pd.DataFrame()

with tab3:
    if not outlier_issues_df.empty:
        edited_outliers = st.data_editor(
            outlier_issues_df,
            column_config={"Apply_Fix": st.column_config.CheckboxColumn("Remove Outlier?", default=True)},
            disabled=[c for c in outlier_issues_df.columns if c != "Apply_Fix"],
            hide_index=True,
            use_container_width=True,
            key="editor_outliers"
        )
    else:
        st.info("No outliers detected.")
        edited_outliers = pd.DataFrame()

with tab4:
    if not dup_issues_df.empty:
        edited_dups = st.data_editor(
            dup_issues_df,
            column_config={"Apply_Fix": st.column_config.CheckboxColumn("Drop Duplicate?", default=True)},
            disabled=[c for c in dup_issues_df.columns if c != "Apply_Fix"],
            hide_index=True,
            use_container_width=True,
            key="editor_dups"
        )
    else:
        st.info("No duplicates detected.")
        edited_dups = pd.DataFrame()

st.divider()

# Extract only the rows where the user left 'Apply_Fix' checked (True)
approved_leaks = list(edited_leakage[edited_leakage["Apply_Fix"]]["Feature"].values) if not edited_leakage.empty else []
approved_labels = edited_labels[edited_labels["Apply_Fix"]] if not edited_labels.empty else pd.DataFrame()
approved_outliers = edited_outliers[edited_outliers["Apply_Fix"]] if not edited_outliers.empty else pd.DataFrame()
approved_dups = edited_dups[edited_dups["Apply_Fix"]] if not edited_dups.empty else pd.DataFrame()

st.subheader("🧪 3. Execute Benchmark & Visual Comparison")
st.write(
    f"**Currently Approved for Repair:** "
    f"`{len(approved_leaks)}` Leaked Features to Drop | "
    f"`{len(approved_labels)}` Labels to Flip | "
    f"`{len(approved_outliers)}` Outliers to Remove | "
    f"`{len(approved_dups)}` Duplicates to Drop"
)

def prepare_eval_matrices(train_data, test_data, feature_list, target):
    """Safely encodes arbitrary user train & test splits together for Random Forest benchmarking."""
    combined = pd.concat([train_data[feature_list], test_data[feature_list]], axis=0)
    proc = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        if pd.api.types.is_numeric_dtype(combined[col]):
            proc[col] = combined[col].fillna(combined[col].median() if not combined[col].dropna().empty else 0)
        else:
            proc[col] = LabelEncoder().fit_transform(combined[col].astype(str).fillna("MISSING"))
    X_tr = proc.loc[train_data.index].fillna(0)
    X_te = proc.loc[test_data.index].fillna(0)

    le_y = LabelEncoder()
    le_y.fit(pd.concat([train_data[target], test_data[target]]).astype(str))
    y_tr = le_y.transform(train_data[target].astype(str))
    y_te = le_y.transform(test_data[target].astype(str))
    return X_tr, X_te, y_tr, y_te, le_y.classes_

if st.button("🚀 Apply Checked Fixes & Run Benchmark", type="primary"):
    eval_features = [c for c in engine.X_encoded.columns if c not in approved_leaks]

    if not eval_features:
        st.error("Cannot run benchmark: All available features were dropped. Please uncheck at least one feature.")
        st.stop()

    # 1. Baseline Model on Original Uncleaned Training Data
    X_tr_base, X_te_base, y_tr_base, y_te_base, class_labels = prepare_eval_matrices(train_df, test_df, eval_features, target_col)
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_tr_base, y_tr_base)
    preds_base = clf_base.predict(X_te_base)
    acc_base = accuracy_score(y_te_base, preds_base)
    f1_base = f1_score(y_te_base, preds_base, average="weighted")
    cm_base = confusion_matrix(y_te_base, preds_base)

    # 2. Apply ONLY User-Checked Repairs to Training Data
    cleaned_train = train_df.copy()
    if approved_leaks:
        cleaned_train = cleaned_train.drop(columns=approved_leaks, errors="ignore")

    relabeled_indices = []
    if not approved_labels.empty:
        for _, row in approved_labels.iterrows():
            orig_dtype = cleaned_train[target_col].dtype
            new_val = row["Suggested_Label"]
            if pd.api.types.is_integer_dtype(orig_dtype):
                new_val = int(float(new_val))
            elif pd.api.types.is_float_dtype(orig_dtype):
                new_val = float(new_val)
            cleaned_train.loc[row["Row_Index"], target_col] = new_val
            relabeled_indices.append(int(row["Row_Index"]))

    rows_to_drop = set()
    if not approved_outliers.empty:
        rows_to_drop.update(approved_outliers["Row_Index"].astype(int).values)
    if not approved_dups.empty:
        rows_to_drop.update(approved_dups["Row_Index"].astype(int).values)

    cleaned_train = cleaned_train.drop(index=list(rows_to_drop), errors="ignore")

    # 3. Train Model on Cleaned Training Data
    X_tr_clean, X_te_clean, y_tr_clean, y_te_clean, _ = prepare_eval_matrices(cleaned_train, test_df, eval_features, target_col)
    clf_clean = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_clean.fit(X_tr_clean, y_tr_clean)
    preds_clean = clf_clean.predict(X_te_clean)
    acc_clean = accuracy_score(y_te_clean, preds_clean)
    f1_clean = f1_score(y_te_clean, preds_clean, average="weighted")
    cm_clean = confusion_matrix(y_te_clean, preds_clean)

    # Build Detailed Row-Level Audit Report Markdown
    report_md = f"""# 🩺 DataMedic Human-in-the-Loop Audit Report
**Dataset:** `{uploaded_file.name}`  
**Target Column:** `{target_col}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Downstream Model Performance (Held-Out Test Set: {len(test_df)} rows)
| Metric | Baseline (Noisy Train) | Cleaned (User-Approved) | Delta |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | {acc_base*100:.2f}% | {acc_clean*100:.2f}% | {(acc_clean - acc_base)*100:+.2f}% |
| **Weighted F1-Score** | {f1_base*100:.2f}% | {f1_clean*100:.2f}% | {(f1_clean - f1_base)*100:+.2f}% |

## 2. User-Approved Repairs Log
- **Original Training Rows:** {len(train_df)} -> **Final Cleaned Training Rows:** {len(cleaned_train)}
- **Leaked Features Dropped ({len(approved_leaks)}):** {', '.join(approved_leaks) if approved_leaks else 'None'}
- **Relabeled Row Indices ({len(relabeled_indices)}):** {relabeled_indices if relabeled_indices else 'None'}
- **Removed Outlier/Duplicate Row Indices ({len(rows_to_drop)}):** {sorted(list(rows_to_drop)) if rows_to_drop else 'None'}
"""

    st.session_state["benchmark_results"] = {
        "acc_base": acc_base,
        "acc_clean": acc_clean,
        "f1_base": f1_base,
        "f1_clean": f1_clean,
        "cm_base": cm_base,
        "cm_clean": cm_clean,
        "class_labels": class_labels,
        "raw_train_df": train_df,
        "cleaned_train_df": cleaned_train,
        "eval_features": eval_features,
        "cleaned_csv": cleaned_train.to_csv(index=False).encode("utf-8"),
        "report_md": report_md.encode("utf-8"),
        "summary_msg": f"Benchmark Complete! Applied {len(relabeled_indices)} approved relabelings and removed {len(rows_to_drop)} approved outlier/duplicate rows."
    }

# Display Persistent Results, Visual Charts & Export Buttons
if "benchmark_results" in st.session_state:
    res = st.session_state["benchmark_results"]
    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline Test Accuracy", f"{res['acc_base']*100:.2f}%")
    r2.metric("Cleaned Test Accuracy", f"{res['acc_clean']*100:.2f}%", delta=f"{(res['acc_clean'] - res['acc_base'])*100:+.2f}%")
    r3.metric("Cleaned Weighted F1", f"{res['f1_clean']*100:.2f}%", delta=f"{(res['f1_clean'] - res['f1_base'])*100:+.2f}%")
    st.success(res["summary_msg"])

    # --- VISUAL DIAGNOSTICS SECTION ---
    st.subheader("📊 4. Visual Impact Analysis (Before vs. After Cleaning)")
    viz_tab1, viz_tab2 = st.tabs(["🎯 Confusion Matrix Comparison (Test Set)", "📉 Feature Distribution Shift (Training Set)"])

    with viz_tab1:
        st.caption("Compare class-level predictions on the untouched test set before and after cleaning the training data.")
        fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        sns.heatmap(
            res["cm_base"], annot=True, fmt="d", cmap="Blues",
            xticklabels=res["class_labels"], yticklabels=res["class_labels"], ax=ax1, cbar=False
        )
        ax1.set_title(f"Baseline Model (Acc: {res['acc_base']*100:.1f}%)")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        sns.heatmap(
            res["cm_clean"], annot=True, fmt="d", cmap="Greens",
            xticklabels=res["class_labels"], yticklabels=res["class_labels"], ax=ax2, cbar=False
        )
        ax2.set_title(f"Cleaned DataMedic Model (Acc: {res['acc_clean']*100:.1f}%)")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with viz_tab2:
        raw_tr = res["raw_train_df"]
        clean_tr = res["cleaned_train_df"]
        num_cols = [c for c in res["eval_features"] if c in clean_tr.columns and pd.api.types.is_numeric_dtype(clean_tr[c])]

        if num_cols:
            selected_feat = st.selectbox("Select Numeric Feature to Inspect Distribution:", num_cols)
            fig_dist, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.2))

            # Side-by-side Boxplot to show outlier removal
            box_data = pd.DataFrame({
                "Value": pd.concat([raw_tr[selected_feat], clean_tr[selected_feat]], ignore_index=True),
                "Stage": (["Before Cleaning"] * len(raw_tr)) + (["After Cleaning"] * len(clean_tr))
            })
            sns.boxplot(data=box_data, x="Stage", y="Value", hue="Stage", palette=["#e74c3c", "#2ecc71"], ax=ax_box, legend=False)
            ax_box.set_title(f"Outlier Spread: '{selected_feat}'")

            # Overlapping Histogram / KDE to show density shift
            sns.histplot(raw_tr[selected_feat].dropna(), color="#e74c3c", label="Before Cleaning", kde=True, stat="density", alpha=0.4, ax=ax_hist)
            sns.histplot(clean_tr[selected_feat].dropna(), color="#2ecc71", label="After Cleaning", kde=True, stat="density", alpha=0.5, ax=ax_hist)
            ax_hist.set_title(f"Density Distribution: '{selected_feat}'")
            ax_hist.legend()

            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)

        else:
            st.info("No numeric features available for distribution plotting.")

    st.divider()
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label="📥 Download Cleaned Training Dataset (CSV)",
        data=res["cleaned_csv"],
        file_name=f"cleaned_{uploaded_file.name}",
        mime="text/csv",
        use_container_width=True
    )
    dl2.download_button(
        label="📄 Download Audit & Benchmark Report (.md)",
        data=res["report_md"],
        file_name="DataMedic_Audit_Report.md",
        mime="text/markdown",
        use_container_width=True
    )