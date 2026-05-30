from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parent.parent / "data"
CLASSIFIED_CSV = BASE_DIR / "kurly_reviews_classified.csv"
LLM_CSV = BASE_DIR / "kurly_reviews_llm_validation.csv"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

KURLY_PURPLE = "#5F0080"
KURLY_PURPLE_DARK = "#3d0054"
KURLY_PURPLE_LIGHT = "#f7f1fa"
KURLY_PURPLE_BORDER = "#ead9f0"

AXES = ["맛", "품질", "신선도", "배송"]
CATEGORIES = ["사과", "딸기", "토마토", "소고기등심", "생연어", "계란"]
CATEGORY_GRID_ORDER = ["사과", "딸기", "토마토", "계란", "소고기등심", "생연어"]
CATEGORY_GROUP_LABEL = {
    "사과": "과일",
    "딸기": "과일",
    "토마토": "채소",
    "계란": "계란",
    "소고기등심": "육류",
    "생연어": "수산",
}
NEGATIVE = "부정"
POSITIVE = "긍정"

CATEGORY_IMPORTANCE = {
    "계란": 1.00,
    "딸기": 0.90,
    "생연어": 0.85,
    "토마토": 0.80,
    "소고기등심": 0.70,
    "사과": 0.60,
}

NEGATIVE_KEYWORDS = {
    "파손/포장": ["깨짐", "깨져", "깨졌", "파손", "터짐", "흘러", "찌그러", "포장"],
    "신선도": ["비린", "냄새", "상한", "상했", "무른", "시들", "물러", "변색"],
    "품질": ["흠집", "작아", "작은", "크기", "상태", "질겨", "기름", "마블링"],
    "맛": ["맛없", "싱겁", "시큼", "달지", "당도", "질기", "느끼"],
    "배송": ["배송", "녹아", "늦", "누락", "완충"],
}

STOPWORDS = {
    "그리고", "하지만", "너무", "정말", "진짜", "그냥", "이번", "항상", "조금",
    "구매", "주문", "상품", "제품", "리뷰", "먹어", "먹고", "먹는", "받았",
    "있어요", "합니다", "해서", "보다", "같아요", "좋아요", "괜찮", "컬리",
    "이거", "저는", "다시", "많이", "매우", "정도", "때문", "처음",
    "사과", "딸기", "토마토", "소고기등심", "생연어", "계란", "달걀", "일반", "스테비아", "방울",
}


st.set_page_config(
    page_title="의사결정 중심 VOC 운영 대시보드",
    page_icon=None,
    layout="wide",
)

st.markdown(
    f"""
<style>
:root {{
    --kurly-purple: {KURLY_PURPLE};
    --kurly-purple-dark: {KURLY_PURPLE_DARK};
    --kurly-purple-light: {KURLY_PURPLE_LIGHT};
    --kurly-purple-border: {KURLY_PURPLE_BORDER};
}}
.stApp h1, .stApp h2, .stApp h3 {{
    color: var(--kurly-purple-dark);
}}
.block-container {{
    max-width: 1320px;
    padding-top: 1.4rem;
    padding-bottom: 2.4rem;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: var(--kurly-purple);
    border-bottom-color: var(--kurly-purple);
}}
section[data-testid="stSidebar"] {{
    border-right: 1px solid var(--kurly-purple-border);
}}
.keyword-card-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 4px 0 10px;
}}
.keyword-card {{
    min-height: 76px;
    border: 1px solid var(--kurly-purple-border);
    border-radius: 8px;
    background: #fff;
    padding: 12px;
}}
.keyword-card-label {{
    color: var(--kurly-purple-dark);
    font-size: 15px;
    font-weight: 700;
    line-height: 1.2;
    word-break: keep-all;
}}
.keyword-card-value {{
    color: #6b5575;
    font-size: 12px;
    margin-top: 8px;
}}
</style>
""",
    unsafe_allow_html=True,
)


def csv_fingerprint() -> tuple[tuple[str, float | None, int | None], ...]:
    paths = (CLASSIFIED_CSV, LLM_CSV)
    fingerprint = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            fingerprint.append((path.name, stat.st_mtime, stat.st_size))
        else:
            fingerprint.append((path.name, None, None))
    return tuple(fingerprint)


@st.cache_data
def load_data(_fingerprint: tuple[tuple[str, float | None, int | None], ...]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = pd.read_csv(CLASSIFIED_CSV, encoding="utf-8-sig")
    df["registered_at"] = pd.to_datetime(df["registered_at"], errors="coerce")
    for axis in AXES:
        df[f"axis_{axis}"] = df[f"axis_{axis}"].fillna(0).astype(int)

    df_llm = None
    if LLM_CSV.exists():
        df_llm = pd.read_csv(LLM_CSV, encoding="utf-8-sig")
    return df, df_llm


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filter")
        categories = st.multiselect("카테고리", CATEGORIES, default=CATEGORIES)
        subtypes = st.multiselect(
            "서브타입",
            sorted(df["subtype"].dropna().astype(str).unique()),
            default=sorted(df["subtype"].dropna().astype(str).unique()),
        )
        sentiments = st.multiselect(
            "감정",
            sorted(df["sentiment"].dropna().astype(str).unique()),
            default=sorted(df["sentiment"].dropna().astype(str).unique()),
        )

        min_date = df["registered_at"].min()
        max_date = df["registered_at"].max()
        date_range = None
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "리뷰 등록일",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

    mask = (
        df["category"].isin(categories)
        & df["subtype"].astype(str).isin(subtypes)
        & df["sentiment"].isin(sentiments)
    )
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        date_mask = df["registered_at"].between(start, end + pd.Timedelta(days=1))
        is_full_date_range = start.date() == min_date.date() and end.date() == max_date.date()
        if is_full_date_range:
            date_mask |= df["registered_at"].isna()
        mask &= date_mask
    return df[mask].copy()


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    neg_ratio = df.groupby("category")["sentiment"].apply(lambda x: (x == NEGATIVE).mean())
    review_share = df.groupby("category").size() / len(df)
    importance = pd.Series(CATEGORY_IMPORTANCE)
    raw_score = (neg_ratio * review_share * importance).fillna(0)
    max_score = raw_score.max()
    priority_index = raw_score / max_score * 100 if max_score > 0 else raw_score

    rows = []
    neg_df = df[df["sentiment"] == NEGATIVE]
    for category in priority_index.sort_values(ascending=False).index:
        cat_neg = neg_df[neg_df["category"] == category]
        axis_pct = {
            axis: cat_neg[f"axis_{axis}"].mean() * 100 if len(cat_neg) else 0
            for axis in AXES
        }
        top_axes = sorted(axis_pct.items(), key=lambda item: -item[1])[:2]
        score = float(priority_index.loc[category])
        rows.append(
            {
                "category": category,
                "Priority Index": round(score, 1),
                "tier": "P1" if score >= 80 else ("P2" if score >= 50 else "P3"),
                "negative_rate": round(neg_ratio.loc[category] * 100, 1),
                "review_share": round(review_share.loc[category] * 100, 1),
                "importance": CATEGORY_IMPORTANCE.get(category, np.nan),
                "negative_reviews": int(len(cat_neg)),
                "top_negative_axes": " / ".join([f"{axis} {pct:.1f}%" for axis, pct in top_axes]),
            }
        )
    return pd.DataFrame(rows)


def axis_heatmap(df: pd.DataFrame, negative_only: bool = False) -> pd.DataFrame:
    source = df[df["sentiment"] == NEGATIVE] if negative_only else df
    rows = []
    for category in CATEGORIES:
        cat = source[source["category"] == category]
        rows.append(
            {
                "category": category,
                **{
                    axis: round(cat[f"axis_{axis}"].mean() * 100, 1) if len(cat) else 0
                    for axis in AXES
                },
            }
        )
    return pd.DataFrame(rows).set_index("category")


def voc_axis_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    neg_df = df[df["sentiment"] == NEGATIVE]
    rows = []
    for axis in AXES:
        count = int(neg_df[f"axis_{axis}"].sum()) if len(neg_df) else 0
        rate = count / len(neg_df) * 100 if len(neg_df) else 0
        rows.append(
            {
                "VOC 축": axis,
                "부정 리뷰 수": count,
                "부정 리뷰 내 비중": round(rate, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("부정 리뷰 수", ascending=False)


def keyword_counts(
    texts: pd.Series,
    top_n: int = 50,
    extra_stopwords: set[str] | None = None,
) -> Counter:
    stopwords = STOPWORDS | (extra_stopwords or set())
    domain_terms = {term for term in stopwords if len(term) >= 2}
    words: list[str] = []
    for text in texts.dropna().astype(str):
        words.extend(re.findall(r"[가-힣A-Za-z]{2,}", text))
    return Counter(
        word
        for word in words
        if word not in stopwords and not any(word.startswith(term) for term in domain_terms)
    )


def llm_metrics(df: pd.DataFrame, df_llm: pd.DataFrame | None) -> pd.DataFrame:
    if df_llm is None or df_llm.empty:
        return pd.DataFrame()

    base = df.reset_index().rename(columns={"index": "sample_idx"})
    cols = ["sample_idx", "llm_sentiment"] + [f"llm_axis_{axis}" for axis in AXES]
    compare = base.merge(df_llm[cols], on="sample_idx", how="inner")
    rows = []
    for axis in AXES:
        true = compare[f"axis_{axis}"].fillna(0).astype(int)
        pred = compare[f"llm_axis_{axis}"].fillna(0).astype(int)
        tn = int(((true == 0) & (pred == 0)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        fn = int(((true == 1) & (pred == 0)).sum())
        tp = int(((true == 1) & (pred == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        rows.append(
            {
                "axis": axis,
                "Accuracy": round(accuracy, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
            }
        )
    return pd.DataFrame(rows)


def plot_priority(priority: pd.DataFrame) -> go.Figure:
    plot_df = priority.sort_values("Priority Index", ascending=True)
    colors = {"P1": KURLY_PURPLE, "P2": "#9b5fb5", "P3": "#adb5bd"}
    fig = go.Figure(
        go.Bar(
            x=plot_df["Priority Index"],
            y=plot_df["category"],
            orientation="h",
            marker_color=[colors.get(tier, "#495057") for tier in plot_df["tier"]],
            text=plot_df["Priority Index"],
        )
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="Priority Index",
        height=340,
        margin=dict(l=20, r=40, t=55, b=30),
        xaxis_title="상대 우선순위 점수",
        yaxis_title=None,
        plot_bgcolor="white",
    )
    return fig


def plot_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=[
                [0.0, "#fbf7fd"],
                [0.45, "#d8b7e6"],
                [0.75, "#9b5fb5"],
                [1.0, KURLY_PURPLE],
            ],
            text=matrix.values,
            texttemplate="%{text:.1f}%",
            hovertemplate="카테고리=%{y}<br>축=%{x}<br>언급률=%{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
    )
    return fig


def category_axis_risk(df: pd.DataFrame, category: str) -> pd.DataFrame:
    cat_neg = df[(df["category"] == category) & (df["sentiment"] == NEGATIVE)]
    rows = []
    for axis in AXES:
        rows.append(
            {
                "VOC 축": axis,
                "언급률": round(cat_neg[f"axis_{axis}"].mean() * 100, 1) if len(cat_neg) else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_category_axis_risk(df: pd.DataFrame, category: str) -> go.Figure:
    plot_df = category_axis_risk(df, category).sort_values("언급률", ascending=True)
    group_label = CATEGORY_GROUP_LABEL.get(category, "기타")
    fig = go.Figure(
        go.Bar(
            x=plot_df["언급률"],
            y=plot_df["VOC 축"],
            orientation="h",
            marker_color=KURLY_PURPLE,
            text=plot_df["언급률"],
        )
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=f"{category} ({group_label})",
        height=245,
        margin=dict(l=18, r=38, t=45, b=25),
        xaxis_title="부정 리뷰 내 언급률",
        yaxis_title=None,
        plot_bgcolor="white",
        xaxis=dict(range=[0, max(55, float(plot_df["언급률"].max()) + 10)]),
    )
    return fig


def tomato_subtype_axis_table(df: pd.DataFrame) -> pd.DataFrame:
    tomato = df[df["category"] == "토마토"]
    if tomato.empty:
        return pd.DataFrame()

    subtype_order = ["일반", "방울", "스테비아", "방울+스테비아"]
    pivot = tomato.groupby("subtype")[[f"axis_{axis}" for axis in AXES]].mean() * 100
    pivot.columns = AXES
    pivot = pivot.reindex([subtype for subtype in subtype_order if subtype in pivot.index])
    return pivot.round(1)


def plot_tomato_subtype_axis(df: pd.DataFrame) -> go.Figure:
    matrix = tomato_subtype_axis_table(df)
    fig = go.Figure()
    colors = {
        "맛": KURLY_PURPLE,
        "품질": "#9b5fb5",
        "신선도": "#2f9e44",
        "배송": "#868e96",
    }
    for axis in AXES:
        fig.add_bar(
            x=matrix.index,
            y=matrix[axis],
            name=axis,
            marker_color=colors.get(axis, KURLY_PURPLE),
            text=matrix[axis],
        )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="토마토 subtype별 평가축 언급률",
        height=340,
        margin=dict(l=20, r=20, t=55, b=35),
        yaxis_title="언급률",
        xaxis_title=None,
        barmode="group",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def priority_display_table(priority: pd.DataFrame) -> pd.DataFrame:
    if priority.empty:
        return pd.DataFrame()
    display = priority.rename(
        columns={
            "category": "카테고리",
            "Priority Index": "상대 우선순위 점수",
            "tier": "등급",
            "negative_rate": "부정률",
            "review_share": "리뷰 비중",
            "importance": "운영 중요도",
            "negative_reviews": "부정 리뷰 수",
            "top_negative_axes": "주요 불만 축",
        }
    )
    return display[["카테고리", "상대 우선순위 점수", "등급", "부정률", "리뷰 비중", "주요 불만 축"]]


def plot_f1(metrics_df: pd.DataFrame) -> go.Figure:
    plot_df = metrics_df.sort_values("F1", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_df["F1"],
            y=plot_df["axis"],
            orientation="h",
            marker_color=KURLY_PURPLE,
            text=plot_df["F1"],
        )
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="F1 요약",
        height=300,
        margin=dict(l=20, r=40, t=55, b=30),
        xaxis_title="F1",
        yaxis_title=None,
        plot_bgcolor="white",
        xaxis=dict(range=[0, 1.05]),
    )
    return fig


def draw_wordcloud(texts: pd.Series, extra_stopwords: set[str] | None = None) -> None:
    counts = keyword_counts(texts, top_n=80, extra_stopwords=extra_stopwords)
    if not counts:
        st.info("워드클라우드를 생성할 텍스트가 없습니다.")
        return

    wc = WordCloud(
        font_path=FONT_PATH,
        width=1000,
        height=520,
        background_color="white",
        colormap="Dark2",
        max_words=80,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(dict(counts))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def build_action_memo(priority: pd.DataFrame) -> list[str]:
    if priority.empty:
        return ["선택된 조건에서 우선순위 산출이 불가합니다."]

    top = priority.iloc[0]
    memo = [f"{top['category']} P1 관리", f"핵심 축: {top['top_negative_axes']}"]
    if top["category"] == "계란":
        memo.extend(["포장/완충재 기준 점검", "파손 리뷰 우선 확인", "배송 취약 SKU 분리 관리"])
    elif top["category"] == "생연어":
        memo.extend(["신선도 클레임 우선 확인", "비린내/변질 표현 모니터링", "콜드체인 기준 재점검"])
    elif top["category"] == "토마토":
        memo.extend(["당도 메시지 강화", "스테비아 콘텐츠 보강", "품질 편차 리뷰 확인"])
    elif top["category"] in ["사과", "딸기"]:
        memo.extend(["당도/크기 기준 명시", "흠집 검수 기준 강화", "품질 편차 상품 점검"])
    else:
        memo.extend(["맛/품질 정보 보강", "조리 가이드 개선", "부위 설명 보완"])
    return memo


def top_keywords_table(
    texts: pd.Series,
    top_n: int = 10,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    counts = keyword_counts(texts, top_n=top_n, extra_stopwords=extra_stopwords)
    return pd.DataFrame(counts.most_common(top_n), columns=["keyword", "count"])


def tomato_subtype_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    tomato = df[df["category"] == "토마토"]
    if tomato.empty:
        return pd.DataFrame()

    base = tomato[tomato["subtype"] == "일반"]
    base_taste_rate = base["axis_맛"].mean() * 100 if len(base) else tomato["axis_맛"].mean() * 100
    rows = []
    for subtype, group in tomato.groupby("subtype"):
        review_share = len(group) / len(tomato) * 100 if len(tomato) else 0
        pos_rate = (group["sentiment"] == POSITIVE).mean() * 100 if len(group) else 0
        neg_rate = (group["sentiment"] == NEGATIVE).mean() * 100 if len(group) else 0
        taste_rate = group["axis_맛"].mean() * 100 if len(group) else 0
        freshness_rate = group["axis_신선도"].mean() * 100 if len(group) else 0
        taste_diff = taste_rate - base_taste_rate
        rows.append(
            {
                "subtype": subtype,
                "리뷰 수": f"{len(group):,}건",
                "리뷰 비중": f"{review_share:.1f}%",
                "긍정률": f"{pos_rate:.1f}%",
                "부정률": f"{neg_rate:.1f}%",
                "맛 언급률": f"{taste_rate:.1f}%",
                "일반 대비 맛 차이": f"{taste_diff:+.1f}%p",
                "신선도 언급률": f"{freshness_rate:.1f}%",
            }
        )
    order = ["일반", "방울", "스테비아", "방울+스테비아"]
    table = pd.DataFrame(rows)
    table["sort_order"] = table["subtype"].map({name: idx for idx, name in enumerate(order)})
    table = table.sort_values("sort_order").drop(columns="sort_order")
    metric_order = ["리뷰 수", "리뷰 비중", "긍정률", "부정률", "맛 언급률", "일반 대비 맛 차이", "신선도 언급률"]
    transposed = table.set_index("subtype")[metric_order].T.reset_index().rename(columns={"index": "지표"})
    return transposed


def product_score_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    total_reviews = len(df)
    neg_df = df[df["sentiment"] == NEGATIVE]
    for (product_name, category, subtype), group in df.groupby(["product_name", "category", "subtype"], dropna=False):
        product_neg = neg_df[neg_df["product_name"] == product_name]
        review_count = len(group)
        negative_rate = (group["sentiment"] == NEGATIVE).mean() * 100
        volume_weight = review_count / total_reviews * 100
        raw_score = negative_rate * 0.75 + volume_weight * 0.25
        product_no = str(group["product_no"].iloc[0]).split(".")[0]
        axis_pct = {
            axis: product_neg[f"axis_{axis}"].mean() * 100 if len(product_neg) else 0
            for axis in AXES
        }
        top_axis, top_axis_rate = max(axis_pct.items(), key=lambda item: item[1])
        rows.append(
            {
                "product_no": product_no,
                "product_name": product_name,
                "category": category,
                "subtype": subtype,
                "reviews": review_count,
                "negative_rate": round(negative_rate, 1),
                "risk_score_raw": raw_score,
                "top_risk_axis": top_axis if top_axis_rate > 0 else "-",
                "negative_reviews": int(len(product_neg)),
            }
        )

    table = pd.DataFrame(rows)
    max_score = table["risk_score_raw"].max()
    table["operation_score"] = (
        (table["risk_score_raw"] / max_score * 100).round(1) if max_score > 0 else 0
    )
    return table.sort_values(["operation_score", "negative_reviews"], ascending=False).drop(columns="risk_score_raw")


def product_risk_diagnosis(product_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_neg = product_df[product_df["sentiment"] == NEGATIVE]
    axis_rows = []
    for axis in AXES:
        axis_rows.append(
            {
                "risk_axis": axis,
                "negative_axis_rate": round(product_neg[f"axis_{axis}"].mean() * 100, 1)
                if len(product_neg)
                else 0,
            }
        )
    keyword_df = voc_axis_risk_summary(product_df)
    return pd.DataFrame(axis_rows).sort_values("negative_axis_rate", ascending=False), keyword_df


def product_sentiment_top3(product_df: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    sentiment_text = product_df[product_df["sentiment"] == sentiment]["review_text"]
    product_terms = set(product_df["category"].dropna().astype(str))
    product_terms |= set(product_df["subtype"].dropna().astype(str))
    return top_keywords_table(sentiment_text, top_n=3, extra_stopwords=product_terms)


def render_keyword_metrics(keyword_df: pd.DataFrame, empty_message: str) -> None:
    if keyword_df.empty:
        st.info(empty_message)
        return

    cards = ['<div class="keyword-card-grid">']
    for _, row in keyword_df.iterrows():
        keyword = html.escape(str(row["keyword"]))
        count = int(row["count"])
        cards.append(
            f"""
<div class="keyword-card">
  <div class="keyword-card-label">{keyword}</div>
  <div class="keyword-card-value">{count:,}회 언급</div>
</div>
"""
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def product_url(product_df: pd.DataFrame) -> str | None:
    if product_df.empty or "product_no" not in product_df.columns:
        return None
    product_no = str(product_df["product_no"].iloc[0]).split(".")[0]
    if not product_no or product_no.lower() == "nan":
        return None
    return f"https://www.kurly.com/goods/{product_no}"


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    st.markdown(
        """
<style>
.voc-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 10px;
}
.voc-kpi-card {
    border: 1px solid #ead9f0;
    border-left: 4px solid var(--accent);
    background: linear-gradient(180deg, #fff 0%, #fdf9ff 100%);
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(95, 0, 128, 0.06);
}
.voc-kpi-label {
    color: #6b5575;
    font-size: 13px;
    margin-bottom: 4px;
}
.voc-kpi-value {
    color: #3d0054;
    font-size: 26px;
    font-weight: 760;
    line-height: 1.1;
}
@media (max-width: 900px) {
    .voc-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
        unsafe_allow_html=True,
    )
    card_html = ['<div class="voc-kpi-grid">']
    for label, value, color in cards:
        card_html.append(
            f"""
<div class="voc-kpi-card" style="--accent:{color}">
  <div class="voc-kpi-label">{label}</div>
  <div class="voc-kpi-value">{value}</div>
</div>
"""
        )
    card_html.append("</div>")
    st.markdown("".join(card_html), unsafe_allow_html=True)


data_fingerprint = csv_fingerprint()
df_all, df_llm_all = load_data(data_fingerprint)

with st.sidebar:
    st.caption("CSV 기준")
    for file_name, modified_at, size in data_fingerprint:
        if modified_at is None:
            st.caption(f"- {file_name}: 없음")
        else:
            updated = pd.to_datetime(modified_at, unit="s").strftime("%Y-%m-%d %H:%M")
            st.caption(f"- {file_name}: {updated}, {size:,} bytes")
df = apply_filters(df_all)
priority = compute_priority(df)

st.title("[마켓컬리 VOC 운영 대시보드]")
st.caption("리뷰 데이터 기반 고객 경험 진단 및 운영 우선순위 관리")

if df.empty:
    st.warning("선택한 조건에 해당하는 리뷰가 없습니다.")
    st.stop()

total_reviews = len(df)
neg_rate = (df["sentiment"] == NEGATIVE).mean() * 100
pos_rate = (df["sentiment"] == POSITIVE).mean() * 100
top_risk = priority.iloc[0]["category"] if not priority.empty else "-"
top_tier = priority.iloc[0]["tier"] if not priority.empty else "-"

tab1, tab2, tab3 = st.tabs(
    ["전사 운영 리스크 현황", "고객 경험 심층 진단", "상품별 상세 점검"]
)

with tab1:
    st.subheader("전사 운영 리스크 현황")
    st.markdown("#### 핵심 운영 지표")
    render_kpi_cards(
        [
            ("전체 리뷰", f"{total_reviews:,}", KURLY_PURPLE),
            ("부정률", f"{neg_rate:.1f}%", KURLY_PURPLE),
            ("긍정률", f"{pos_rate:.1f}%", "#2b8a3e"),
            (f"최우선 관리 카테고리 ({top_tier})", str(top_risk), KURLY_PURPLE),
        ]
    )

    st.markdown("#### Priority Index (운영 우선순위 지수)")
    st.plotly_chart(plot_priority(priority), use_container_width=True)
    st.dataframe(
        priority_display_table(priority),
        use_container_width=True,
        hide_index=True,
        height=245,
        column_config={
            "카테고리": st.column_config.TextColumn("카테고리", width="small"),
            "상대 우선순위 점수": st.column_config.NumberColumn("상대 우선순위 점수", width="medium"),
            "등급": st.column_config.TextColumn("등급", width="small"),
            "부정률": st.column_config.NumberColumn("부정률", format="%.1f%%", width="small"),
            "리뷰 비중": st.column_config.NumberColumn("리뷰 비중", format="%.1f%%", width="small"),
            "주요 불만 축": st.column_config.TextColumn("주요 불만 축", width="large"),
        },
    )

    st.markdown("#### 카테고리별 불만 축 진단")
    for row_start in range(0, len(CATEGORY_GRID_ORDER), 3):
        cols = st.columns(3, gap="large")
        for col, category in zip(cols, CATEGORY_GRID_ORDER[row_start : row_start + 3]):
            with col:
                st.plotly_chart(plot_category_axis_risk(df, category), use_container_width=True)

    with st.expander("최우선 관리 카테고리 및 Action Memo", expanded=False):
        st.markdown(f"**최우선 관리 카테고리:** {top_risk} ({top_tier})")
        st.markdown("**우선 조치:**")
        for item in build_action_memo(priority):
            st.markdown(f"- {item}")

with tab2:
    st.subheader("고객 경험 심층 진단")
    st.markdown("#### 카테고리별 경험 히트맵")
    heatmap_left, heatmap_right = st.columns(2, gap="large")
    with heatmap_left:
        st.plotly_chart(
            plot_heatmap(axis_heatmap(df), "카테고리별 경험 히트맵 - 룰기반 전체 리뷰 기준"),
            use_container_width=True,
        )
    with heatmap_right:
        st.plotly_chart(
            plot_heatmap(axis_heatmap(df, negative_only=True), "카테고리별 경험 히트맵 - 룰기반 부정 리뷰 기준"),
            use_container_width=True,
        )
    st.caption("히트맵은 LLM 기반이 아니라 노트북과 동일한 rule-based `axis_맛/품질/신선도/배송` 라벨 기준입니다. 단, 대시보드 사이드바 필터가 적용된 데이터로 계산됩니다.")

    st.markdown("#### 핵심 경험 요인 분석")
    pos_col, neg_col = st.columns(2, gap="large")
    with pos_col:
        st.markdown("##### 긍정 경험 요인")
        positive_keywords = top_keywords_table(df[df["sentiment"] == POSITIVE]["review_text"], top_n=10)
        st.dataframe(positive_keywords, use_container_width=True, hide_index=True, height=275)
    with neg_col:
        st.markdown("##### 부정 경험 요인")
        negative_keywords = top_keywords_table(df[df["sentiment"] == NEGATIVE]["review_text"], top_n=10)
        st.dataframe(negative_keywords, use_container_width=True, hide_index=True, height=275)

    tomato_metrics = tomato_subtype_metric_table(df)
    if not tomato_metrics.empty:
        st.markdown("#### 토마토 subtype 정량 진단")
        tomato_left, tomato_right = st.columns([0.46, 0.54], gap="large")
        with tomato_left:
            st.plotly_chart(plot_tomato_subtype_axis(df), use_container_width=True)
        with tomato_right:
            st.dataframe(
                tomato_metrics,
                use_container_width=True,
                hide_index=True,
                height=340,
                column_config={
                    "지표": st.column_config.TextColumn("지표", width="medium"),
                    "일반": st.column_config.TextColumn("일반", width="small"),
                    "방울": st.column_config.TextColumn("방울", width="small"),
                    "스테비아": st.column_config.TextColumn("스테비아", width="small"),
                    "방울+스테비아": st.column_config.TextColumn("방울+스테비아", width="medium"),
                },
            )
        st.caption("토마토 subtype 진단은 CSV의 감성 라벨과 `axis_맛/신선도` 라벨을 집계한 정량 지표입니다. `일반 대비 맛 차이`가 높을수록 맛/당도 중심 소구의 근거가 강합니다.")

    st.markdown("#### AI 분석 신뢰도 요약")
    metrics_df = llm_metrics(df_all, df_llm_all)
    if metrics_df.empty:
        st.info("LLM 검증 데이터가 없어 신뢰도 지표를 표시할 수 없습니다.")
    else:
        left, right = st.columns([0.45, 0.55], gap="large")
        with left:
            st.plotly_chart(plot_f1(metrics_df), use_container_width=True)
        with right:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True, height=300)

    with st.expander("Rule-based 한계 및 LLM 보완 설명", expanded=True):
        st.markdown(
            "- Rule-based 분류는 키워드 사전에 없는 표현, 반어, 복합 문장을 놓칠 수 있습니다.\n"
            "- LLM 검증은 자동 분류가 놓친 맥락을 확인하는 품질 점검 장치입니다.\n"
            "- F1이 낮은 축은 키워드 사전 보강 또는 분류 프롬프트 개선 대상으로 봅니다."
        )

with tab3:
    st.subheader("상품별 상세 점검")
    st.markdown("#### 상품 검색 및 필터")
    product_table = product_score_table(df)
    if product_table.empty:
        st.info("선택한 조건에서 상품별 점검 데이터를 만들 수 없습니다.")
        st.stop()

    search = st.text_input("상품명 검색", placeholder="상품명 일부를 입력하세요")
    category_filter = st.multiselect(
        "상품 카테고리",
        sorted(product_table["category"].dropna().unique()),
        default=sorted(product_table["category"].dropna().unique()),
    )

    filtered_products = product_table[product_table["category"].isin(category_filter)].copy()
    if search:
        filtered_products = filtered_products[
            filtered_products["product_name"].astype(str).str.contains(search, case=False, na=False)
        ]

    st.markdown("#### 상품별 운영 점수표")
    st.dataframe(filtered_products.head(50), use_container_width=True, hide_index=True, height=260)

    filtered_review_df = df[df["product_name"].isin(filtered_products["product_name"])]

    if filtered_products.empty:
        st.warning("검색 조건에 맞는 상품이 없습니다.")
        st.stop()

    all_products_option = "선택 카테고리 내 전체 상품"
    product_options = [all_products_option] + filtered_products["product_name"].tolist()
    selected_product = st.selectbox(
        "상세 점검 상품",
        product_options,
        index=0,
        help="'선택 카테고리 내 전체 상품'은 위 상품 카테고리/검색 조건에 맞는 모든 상품을 함께 보여줍니다. 특정 상품을 선택하면 해당 상품만 상세 점검합니다.",
    )
    if selected_product == all_products_option:
        product_df = filtered_review_df.copy()
        st.caption("현재 대표 리뷰와 리스크 진단은 위 상품 카테고리/검색 조건이 먼저 적용된 전체 상품 기준입니다.")
    else:
        product_df = df[df["product_name"] == selected_product].copy()
        st.caption(f"현재 대표 리뷰와 리스크 진단은 선택 상품 `{selected_product}` 기준입니다.")

    keyword_scope = "선택 상품군" if selected_product == all_products_option else "선택 상품"
    st.markdown(f"#### {keyword_scope} 핵심 경험 키워드")
    neg_keyword_col, pos_keyword_col = st.columns(2, gap="large")
    with neg_keyword_col:
        st.markdown("##### 부정 리스크 TOP3")
        selected_negative_top3 = product_sentiment_top3(product_df, NEGATIVE)
        render_keyword_metrics(selected_negative_top3, f"{keyword_scope}의 부정 리스크 키워드가 없습니다.")
    with pos_keyword_col:
        st.markdown("##### 긍정 구매 포인트 TOP3")
        selected_positive_top3 = product_sentiment_top3(product_df, POSITIVE)
        render_keyword_metrics(selected_positive_top3, f"{keyword_scope}의 긍정 구매 포인트 키워드가 없습니다.")

    st.markdown("#### 리스크 원인 진단")
    axis_diag, keyword_diag = product_risk_diagnosis(product_df)
    diag_left, diag_right = st.columns(2, gap="large")
    with diag_left:
        st.dataframe(axis_diag, use_container_width=True, hide_index=True, height=178)
    with diag_right:
        st.dataframe(keyword_diag, use_container_width=True, hide_index=True, height=178)

    st.markdown("#### 대표 고객 리뷰")
    sentiment_options = ["전체"] + sorted(product_df["sentiment"].dropna().astype(str).unique())
    selected_sentiment = st.selectbox(
        "대표 리뷰 감성 필터",
        sentiment_options,
        index=sentiment_options.index(NEGATIVE) if NEGATIVE in sentiment_options else 0,
    )
    review_df = product_df.copy()
    if selected_sentiment != "전체":
        review_df = review_df[review_df["sentiment"] == selected_sentiment]

    representative = review_df.sort_values("registered_at", ascending=False)
    representative = representative.assign(
        상품_바로가기=representative["product_no"].astype(str).str.split(".").str[0].map(
            lambda product_no: f"https://www.kurly.com/goods/{product_no}"
        )
    )
    review_cols = ["registered_at", "category", "subtype", "sentiment", "review_text", "상품_바로가기"]
    st.dataframe(
        representative[review_cols].head(20),
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config={
            "registered_at": st.column_config.DatetimeColumn("등록일"),
            "category": "카테고리",
            "subtype": "서브타입",
            "sentiment": "감성",
            "review_text": st.column_config.TextColumn("대표 고객 리뷰", width="large"),
            "상품_바로가기": st.column_config.LinkColumn("상품 바로가기", display_text="상품 페이지 보기"),
        },
    )
