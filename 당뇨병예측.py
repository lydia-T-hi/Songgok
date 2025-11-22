# app.py
import streamlit as st
import pandas as pd
import numpy as np
import platform
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 깃허브 리눅스 기준
if platform.system() == 'Linux':
    fontname = './NanumGothic.ttf'
    font_files = fm.findSystemFonts(fontpaths=fontname)
    fm.fontManager.addfont(fontname)
    fm._load_fontmanager(try_read_cache=False)
    rc('font', family='NanumGothic')

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="당뇨병 예측 분류 대시보드",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 당뇨병 예측 분류 모델 Streamlit 대시보드")
st.markdown("CSV 파일을 업로드하면 데이터 탐색과 모델 성능 비교까지 한 번에 진행합니다.")

# -----------------------------
# 유틸 함수
# -----------------------------
def basic_info(df: pd.DataFrame):
    info = {
        "행 개수": df.shape[0],
        "열 개수": df.shape[1],
        "결측치 총합": int(df.isna().sum().sum()),
        "중복 행 개수": int(df.duplicated().sum())
    }
    return info

def get_metrics(y_true, y_pred, y_proba, model_name="model"):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    }

def plot_numeric(df, col, chart_type):
    fig, ax = plt.subplots()
    if chart_type == "히스토그램":
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"{col} 분포(히스토그램)")
    elif chart_type == "박스플롯":
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f"{col} 분포(박스플롯)")
    st.pyplot(fig)

def plot_correlation(df):
    # 숫자형 컬럼만
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("상관계수 히트맵")
    st.pyplot(fig)

def plot_metric_bar(metrics_df, metric_name):
    fig, ax = plt.subplots()
    ax.bar(metrics_df["model"], metrics_df[metric_name])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"모델별 {metric_name} 비교")
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(metrics_df["model"], rotation=20)
    st.pyplot(fig)

# -----------------------------
# 사이드바: 파일 업로드
# -----------------------------
st.sidebar.header("1) 데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

# 파일 없을 때 안내
if uploaded is None:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
    st.stop()

# -----------------------------
# 데이터 로딩
# -----------------------------
df = pd.read_csv(uploaded)

# -----------------------------
# 2) 데이터 기본 정보
# -----------------------------
st.header("2) 데이터 기본 정보")

col_a, col_b = st.columns([1,2])

with col_a:
    info = basic_info(df)
    st.subheader("데이터 요약")
    st.write(info)

    st.subheader("컬럼 / 타입")
    st.dataframe(pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    }))

with col_b:
    st.subheader("기술통계(숫자형)")
    st.dataframe(df.describe())

    st.subheader("결측치 현황")
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss > 0].to_frame("missing_count"))

st.subheader("미리보기")
st.dataframe(df.head(20))

# -----------------------------
# 3) 주요 지표 시각화
# -----------------------------
st.header("3) 주요 지표 시각화")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("숫자형 컬럼 시각화")
    if len(num_cols) == 0:
        st.warning("숫자형 컬럼이 없습니다.")
    else:
        selected_num = st.selectbox("컬럼 선택", num_cols, key="num_col")
        chart_type = st.radio("차트 유형", ["히스토그램", "박스플롯"], horizontal=True)
        plot_numeric(df, selected_num, chart_type)

with vis_col2:
    st.subheader("상관관계 시각화")
    if len(num_cols) >= 2:
        if st.button("상관계수 히트맵 그리기"):
            plot_correlation(df)
    else:
        st.warning("상관관계를 그릴 숫자형 컬럼이 2개 이상 필요합니다.")

# -----------------------------
# 4) 모델 학습 & 성능 비교
# -----------------------------
st.header("4) 모델 학습 및 성능 비교")

st.markdown("타깃(정답) 컬럼을 선택하고 여러 모델 성능을 비교합니다.")

# 타깃 컬럼 추정: Outcome 있으면 기본값, 없으면 마지막 컬럼
default_target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
target_col = st.selectbox("타깃 컬럼 선택", df.columns, index=list(df.columns).index(default_target))

# 피처/타깃 분리
X = df.drop(columns=[target_col])
y = df[target_col]

# 숫자형만 사용(노트북 흐름과 동일)
X = X.select_dtypes(include=np.number)

if X.shape[1] == 0:
    st.error("피처로 사용할 숫자형 컬럼이 없습니다. 숫자형 컬럼을 포함한 데이터를 업로드해 주세요.")
    st.stop()

# 학습 옵션
st.subheader("학습 옵션")
opt1, opt2, opt3 = st.columns(3)
with opt1:
    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.2, step=0.05)
with opt2:
    random_state = st.number_input("random_state", 0, 9999, 42)
with opt3:
    scale_on = st.checkbox("표준화 사용(권장)", value=True)

# 모델 선택
st.subheader("모델 선택")
model_choices = st.multiselect(
    "비교할 모델 선택",
    ["Logistic Regression", "Random Forest", "SVC (SVM)", "KNN"],
    default=["Logistic Regression", "Random Forest", "SVC (SVM)"]
)

# 학습 실행 버튼
if st.button("모델 학습 및 평가 실행"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {}
    if "Logistic Regression" in model_choices:
        models["Logistic Regression"] = LogisticRegression(max_iter=2000)
    if "Random Forest" in model_choices:
        models["Random Forest"] = RandomForestClassifier(n_estimators=300, random_state=random_state)
    if "SVC (SVM)" in model_choices:
        models["SVC (SVM)"] = SVC(probability=True, random_state=random_state)
    if "KNN" in model_choices:
        models["KNN"] = KNeighborsClassifier(n_neighbors=5)

    results = []

    for name, model in models.items():
        if scale_on and name in ["Logistic Regression", "SVC (SVM)", "KNN"]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            pipe = Pipeline([("model", model)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # proba 계산 (roc_auc용)
        y_proba = None
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            # decision_function을 0~1로 변환
            dec = pipe.decision_function(X_test)
            y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

        results.append(get_metrics(y_test, y_pred, y_proba, name))

        # 개별 모델 상세 결과(expander)
        with st.expander(f"🔍 {name} 상세 결과 보기"):
            st.write("혼동행렬")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("분류 리포트")
            st.text(classification_report(y_test, y_pred, zero_division=0))

    metrics_df = pd.DataFrame(results)

    st.subheader("모델 성능 비교 표")
    st.dataframe(metrics_df.set_index("model").style.format("{:.3f}"))

    st.subheader("모델 성능 비교 그래프")
    metric_to_plot = st.selectbox(
        "그래프로 비교할 지표 선택",
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    plot_metric_bar(metrics_df, metric_to_plot)

else:
    st.info("위 옵션을 선택한 뒤 **모델 학습 및 평가 실행** 버튼을 눌러주세요.")
