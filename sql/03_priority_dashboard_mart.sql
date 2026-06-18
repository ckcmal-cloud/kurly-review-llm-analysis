-- =========================================================================
-- 03_priority_dashboard_mart.sql
-- 목적: Python 원본 Priority Index 공식을 SQL로 정확히 재현합니다.
--       카테고리 단위로 집계하며, 0~100 정규화 및 P1/P2/P3 Tier 분류를 포함합니다.
-- 유래: 01_kurly_review_analysis_output.ipynb Line 5050~5096
-- =========================================================================

-- =====================================================================
-- [핵심 수정] Python 원본과의 불일치 해소
--
-- 수정 전(기존 SQL):
--   Priority Index = negative_ratio × complaint_volume × avg_complaint_length
--   → 집계 단위: 카테고리 × 불만유형 (2개 레벨)
--   → 정규화: 없음 (절대값 출력)
--   → Tier 분류: 없음
--
-- 수정 후(현재 SQL):
--   Priority Index = neg_ratio × review_share × importance
--   → 집계 단위: 카테고리 (1개 레벨) ← Python과 동일
--   → 정규화: raw_score / raw_score.max() × 100 ← Python과 동일
--   → Tier 분류: P1(≥80), P2(≥50), P3(<50) ← Python과 동일
--
-- Python 원본 공식 (Notebook Line 5073~5078):
--   neg_ratio    = df.groupby('category')['sentiment'].apply(
--                      lambda x: (x == '부정').mean())
--   review_share = df.groupby('category').size() / len(df)
--   importance   = {'계란':1.00, '딸기':0.90, '생연어':0.85,
--                   '토마토':0.80, '소고기등심':0.70, '사과':0.60}
--   raw_score      = neg_ratio × review_share × importance
--   priority_index = (raw_score / raw_score.max() × 100).round(1)
--   tier = 'P1' if idx >= 80 else ('P2' if idx >= 50 else 'P3')
-- =====================================================================

CREATE TABLE mart_voc_priority AS
WITH

-- ① importance 가중치 딕셔너리 (Python Notebook Line 5050~5057 재현)
-- Python: importance = {'계란':1.00, '딸기':0.90, '생연어':0.85,
--                       '토마토':0.80, '소고기등심':0.70, '사과':0.60}
importance_weights AS (
    SELECT '계란'       AS category, 1.00 AS importance UNION ALL
    SELECT '딸기'       AS category, 0.90 AS importance UNION ALL
    SELECT '생연어'     AS category, 0.85 AS importance UNION ALL
    SELECT '토마토'     AS category, 0.80 AS importance UNION ALL
    SELECT '소고기등심' AS category, 0.70 AS importance UNION ALL
    SELECT '사과'       AS category, 0.60 AS importance
),

-- ② 전체 리뷰 수 (review_share 분모: len(df))
-- Python: review_share = df.groupby('category').size() / len(df)
total_review_count AS (
    SELECT COUNT(1) AS total_count
    FROM kurly_reviews_classified
),

-- ③ 카테고리 단위 핵심 지표 집계
-- Python: neg_ratio    = 카테고리 내 부정 리뷰 비율  (.mean())
-- Python: review_share = 카테고리가 전체 리뷰에서 차지하는 비중
category_metrics AS (
    SELECT
        r.category,

        -- neg_ratio: 해당 카테고리 내 부정 리뷰 비율
        -- Python: (x == '부정').mean()
        ROUND(
            AVG(CASE WHEN r.sentiment = '부정' THEN 1.0 ELSE 0.0 END),
            6
        ) AS neg_ratio,

        -- review_share: 카테고리가 전체 리뷰에서 차지하는 비중
        -- Python: df.groupby('category').size() / len(df)
        ROUND(
            COUNT(1) * 1.0 / t.total_count,
            6
        ) AS review_share,

        -- 부정 리뷰 수 (대시보드 참고용)
        SUM(CASE WHEN r.sentiment = '부정' THEN 1 ELSE 0 END) AS neg_count,

        -- 전체 리뷰 수 (대시보드 참고용)
        COUNT(1) AS total_reviews

    FROM kurly_reviews_classified r
    CROSS JOIN total_review_count t
    GROUP BY r.category, t.total_count
),

-- ④ raw_score 연산
-- Python: raw_score = neg_ratio × review_share × importance_s
raw_scores AS (
    SELECT
        m.category,
        m.neg_ratio,
        m.review_share,
        w.importance,
        m.neg_count,
        m.total_reviews,
        ROUND(m.neg_ratio * m.review_share * w.importance, 8) AS raw_score
    FROM category_metrics m
    -- importance 가중치가 없는 카테고리는 제외 (Python의 .mul() 동작: NaN → 제외)
    INNER JOIN importance_weights w
        ON m.category = w.category
),

-- ⑤ 최대값 추출 (정규화 분모)
-- Python: raw_score.max()
max_raw AS (
    SELECT MAX(raw_score) AS max_score
    FROM raw_scores
)

-- ⑥ 최종 Priority Index 산출 및 Tier 분류
-- Python: priority_index = (raw_score / raw_score.max() * 100).round(1)
-- Python: tier = 'P1' if idx >= 80 else ('P2' if idx >= 50 else 'P3')
SELECT
    s.category,

    -- 부정률 (%) — Python 출력: '부정률': neg_ratio * 100
    ROUND(s.neg_ratio * 100, 2)    AS neg_ratio_pct,

    -- 리뷰 비중 (%) — Python 출력: '리뷰 비중': review_share * 100
    ROUND(s.review_share * 100, 2) AS review_share_pct,

    -- importance 가중치 — Python 원본 딕셔너리값
    s.importance,

    -- 부정 리뷰 수 — Python 출력: '부정 리뷰 수'
    s.neg_count,

    -- 전체 리뷰 수 (참고용)
    s.total_reviews,

    -- raw_score (정규화 전 점수)
    s.raw_score,

    -- Priority Index: 0~100 정규화
    -- Python: (raw_score / raw_score.max() * 100).round(1)
    ROUND(s.raw_score / m.max_score * 100, 1) AS priority_index,

    -- Tier 분류
    -- Python: 'P1' if idx >= 80 else ('P2' if idx >= 50 else 'P3')
    CASE
        WHEN ROUND(s.raw_score / m.max_score * 100, 1) >= 80 THEN 'P1'
        WHEN ROUND(s.raw_score / m.max_score * 100, 1) >= 50 THEN 'P2'
        ELSE 'P3'
    END AS tier

FROM raw_scores s
CROSS JOIN max_raw m
ORDER BY priority_index DESC;
