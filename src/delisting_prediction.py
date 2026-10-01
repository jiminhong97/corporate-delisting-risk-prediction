"""Corporate delisting risk prediction using financial ratios.

This script reorganizes the analysis pipeline used in the original course/research
project into a single runnable Python file.

Expected input:
    data/상장폐지_예측데이터셋(2021~2023).csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_PATH = Path("data/상장폐지_예측데이터셋(2021~2023).csv")


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors for all columns."""
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i)
        for i in range(df.shape[1])
    ]
    return vif


def fill_na_by_size(
    x_raw_part: pd.DataFrame,
    size_part: pd.Series,
    group_medians: pd.DataFrame,
    global_median: pd.Series,
    num_cols,
) -> pd.DataFrame:
    """Fill missing values with size-group medians and a global fallback."""
    x_filled = x_raw_part.copy()

    for group, row_median in group_medians.iterrows():
        mask = size_part == group
        if mask.sum() == 0:
            continue

        x_filled.loc[mask, num_cols] = (
            x_filled.loc[mask, num_cols].fillna(row_median)
        )

    x_filled = x_filled.fillna(global_median)
    return x_filled


def evaluate(y_true, y_pred, label: str) -> None:
    """Print major classification metrics."""
    print(f"\n=== {label} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("F2:", fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> None:
    # 1. 데이터 읽기
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}\n"
            "data/README.md를 참고하여 CSV 파일을 배치해 주세요."
        )

    df = pd.read_csv(DATA_PATH)

    # 2. 기업규모 카테고리 정리
    df["기업규모"] = (
        df["기업규모"]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan})
    )

    print("기업규모 고유값:")
    print(df["기업규모"].unique())
    print("\n기업규모별 개수:")
    print(df["기업규모"].value_counts(dropna=False))

    # 3. 식별자 제거 + 타깃/독립변수/기업규모 분리
    df_model = df.drop(
        columns=["업체코드", "종목코드", "종목명", "상장폐지일"],
        errors="ignore",
    )

    y = df_model["상장폐지여부"]
    size = df_model["기업규모"]
    x = df_model.drop(columns=["상장폐지여부", "기업규모"])

    # 4. 문자열 -> 숫자 변환
    for col in x.columns:
        x[col] = (
            x[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        x[col] = pd.to_numeric(x[col], errors="coerce")

    # 결측 Indicator는 대체 전 데이터를 기준으로 생성
    x_raw = x.copy()
    missing_features = pd.DataFrame(index=x.index)

    for col in x_raw.columns:
        if x_raw[col].isna().sum() >= 5:
            missing_features[col + "_missing"] = x_raw[col].isna().astype(int)

    print("\nIndicator 생성된 변수 개수:", missing_features.shape[1])
    print("일부:", list(missing_features.columns[:10]))

    # 5. Train/Test 분할
    (
        x_train_raw,
        x_test_raw,
        y_train,
        y_test,
        size_train,
        size_test,
        miss_train,
        miss_test,
    ) = train_test_split(
        x_raw,
        y,
        size,
        missing_features,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 6. 기업규모별 중앙값으로 결측치 대체
    train_temp = x_train_raw.copy()
    train_temp["기업규모"] = size_train

    num_cols = x_train_raw.columns

    group_medians = (
        train_temp
        .groupby("기업규모")[num_cols]
        .median(numeric_only=True)
    )
    global_median = x_train_raw.median(numeric_only=True)

    x_train = fill_na_by_size(
        x_train_raw,
        size_train,
        group_medians,
        global_median,
        num_cols,
    )
    x_test = fill_na_by_size(
        x_test_raw,
        size_test,
        group_medians,
        global_median,
        num_cols,
    )

    print("\n대체 후 train 결측 개수:", x_train.isna().sum().sum())
    print("대체 후 test 결측 개수:", x_test.isna().sum().sum())

    # 7. Missing Indicator 결합
    x_train_final = pd.concat([x_train, miss_train], axis=1)
    x_test_final = pd.concat([x_test, miss_test], axis=1)

    print("\n최종 train shape:", x_train_final.shape)
    print("최종 test shape:", x_test_final.shape)

    # 8. VIF 기반 다중공선성 통제
    numeric_cols_vif = [
        col for col in x_train_final.columns
        if not col.endswith("_missing")
    ]

    x_train_numeric = x_train_final[numeric_cols_vif]
    vif_df = calculate_vif(x_train_numeric)
    high_vif_features = vif_df[vif_df["VIF"] >= 10]["feature"].tolist()

    x_train_final = x_train_final.drop(
        columns=high_vif_features,
        errors="ignore",
    )
    x_test_final = x_test_final.drop(
        columns=high_vif_features,
        errors="ignore",
    )

    print("\n제거된 고-VIF 변수 목록:")
    print(high_vif_features)
    print("현재 남아 있는 변수 개수:", x_train_final.shape[1])

    # 9. 표준화
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_final)
    x_test_scaled = scaler.transform(x_test_final)

    # 10. Random Oversampling
    ros = RandomOverSampler(random_state=42)
    x_train_bal, y_train_bal = ros.fit_resample(x_train_scaled, y_train)

    print("\n원본 y_train 분포:", y_train.value_counts().to_dict())
    print("Oversampling 후:", y_train_bal.value_counts().to_dict())

    # 11. Logistic Regression
    log_clf = LogisticRegression(max_iter=2000, solver="liblinear")
    log_clf.fit(x_train_bal, y_train_bal)

    # 12. 기본 Threshold = 0.5
    y_pred = log_clf.predict(x_test_scaled)
    y_proba = log_clf.predict_proba(x_test_scaled)[:, 1]

    evaluate(y_test, y_pred, "Threshold = 0.5 기준")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Threshold = 0.5)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", auc)

    # 13. F2-score 기준 Threshold 탐색
    thresholds = np.arange(0.01, 1.00, 0.01)

    precision_list = []
    recall_list = []
    f2_list = []

    best_t = 0.5
    best_f2 = -1.0

    for threshold in thresholds:
        y_tmp = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_tmp, zero_division=0)
        recall = recall_score(y_test, y_tmp, zero_division=0)
        f2 = fbeta_score(y_test, y_tmp, beta=2, zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f2_list.append(f2)

        if f2 > best_f2:
            best_f2 = f2
            best_t = threshold

    print(
        f"\nF2 기준 최적 threshold: {best_t:.2f}, "
        f"최대 F2: {best_f2:.4f}"
    )

    # 14. 탐색된 Threshold 적용
    y_pred_opt = (y_proba >= best_t).astype(int)

    evaluate(
        y_test,
        y_pred_opt,
        f"Optimal Threshold = {best_t:.2f} 적용 결과",
    )

    cm_opt = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Optimal Threshold = {best_t:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # 15. 변수 중요도: 계수 + Odds Ratio
    coef = log_clf.coef_[0]
    features = x_train_final.columns

    coef_df = pd.DataFrame(
        {
            "feature": features,
            "coef": coef,
            "odds_ratio": np.exp(coef),
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    top_n = 20
    top_coef = coef_df.head(top_n).sort_values("coef")

    plt.figure(figsize=(8, 10))
    sns.barplot(
        data=top_coef,
        x="coef",
        y="feature",
        hue="feature",
        legend=False,
        palette="coolwarm",
        edgecolor="black",
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("로지스틱 회귀 변수 중요도 (계수 기준 TOP 20)", fontsize=14)
    plt.xlabel("계수 (coef)", fontsize=12)
    plt.ylabel("변수명 (feature)", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n계수 절대값 기준 상위 20개 변수:")
    print(coef_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
