-- 목적: 리뷰 데이터의 감성(긍정/부정/중립) 비율 요약,부정 리뷰 내에서 구체적인 고객 불만 유형(배송/신선도/품질/맛) 집계
-- 참고: 01_kurly_review_analysis_output.ipynb의 "감성 분포" 및 "4축 분류(axis_keywords)" 로직

-- 1. 전체 리뷰 감성 분포 요약 (Sentiment Summary)
CREATE TABLE voc_sentiment_summary AS
SELECT 
    sentiment,
    COUNT(1) AS review_count,
    ROUND(COUNT(1) * 100.0 / SUM(COUNT(1)) OVER (), 2) AS sentiment_ratio_pct
FROM kurly_reviews_classified
GROUP BY sentiment;


-- 2. 부정(Negative) 리뷰 대상 불만 유형 집계 (Complaint Summary)
-- Python axis_keywords 딕셔너리 전체 키워드를 LIKE 패턴으로 이식:
-- axis_맛      : 31개 키워드 / axis_품질    : 30개 키워드 / axis_신선도  : 26개 키워드 / axis_배송    : 28개 키워드

CREATE TABLE voc_complaint_summary AS
WITH negative_reviews AS (
    -- 부정 리뷰만 필터링
    SELECT 
        product_no AS review_id,
        category,
        review_text,
        LENGTH(review_text) AS review_length
    FROM kurly_reviews_classified
    WHERE sentiment = '부정'
),

-- 4축 독립 플래그 생성 (Python axis_keywords 딕셔너리 전체 반영)
axis_flags AS (
    SELECT 
        review_id,
        category,
        review_text,
        review_length,

        -- 축1: axis_맛 (Python axis_keywords['맛'], 31개 키워드)
        -- 맛있, 맛나, 달, 달콤, 달달, 새콤, 시원, 시큼, 쓴, 짭, 짜, 고소,
        -- 풍미, 향, 감칠맛, 식감, 쫀득, 부드럽, 질김, 질겨, 꿀맛, 싱겁, 담백,
        -- 당도, 산미, 향긋, 기름기, 마블링, 육즙, 쫄깃, 느끼
        CASE WHEN
            review_text LIKE '%맛있%' OR review_text LIKE '%맛나%' OR review_text LIKE '%달콤%'
            OR review_text LIKE '%달달%' OR review_text LIKE '%새콤%' OR review_text LIKE '%시원%'
            OR review_text LIKE '%시큼%' OR review_text LIKE '%고소%' OR review_text LIKE '%풍미%'
            OR review_text LIKE '%감칠맛%' OR review_text LIKE '%식감%' OR review_text LIKE '%쫀득%'
            OR review_text LIKE '%부드럽%' OR review_text LIKE '%질김%' OR review_text LIKE '%질겨%'
            OR review_text LIKE '%꿀맛%' OR review_text LIKE '%싱겁%' OR review_text LIKE '%담백%'
            OR review_text LIKE '%당도%' OR review_text LIKE '%산미%' OR review_text LIKE '%향긋%'
            OR review_text LIKE '%기름기%' OR review_text LIKE '%마블링%' OR review_text LIKE '%육즙%'
            OR review_text LIKE '%쫄깃%' OR review_text LIKE '%느끼%'
            OR review_text LIKE '%쓴%'   OR review_text LIKE '%짭%'  OR review_text LIKE '%짜%'
            OR review_text LIKE '%향%'   OR review_text LIKE '%달%'
        THEN 1 ELSE 0 END AS axis_맛,

        -- 축2: axis_품질 (Python axis_keywords['품질'], 30개 키워드)
        -- 크기, 크다, 작다, 작은, 큰, 실하, 알차, 등급, 1+, 1++, 한우,
        -- 국산, 원산지, 흠집, 상처, 멍, 벌레, 이물, 곰팡이, 가격, 저렴,
        -- 비싸, 가성비, 값어치, 품질, 퀄리티, 모양, 색, 색감, 때깔
        CASE WHEN
            review_text LIKE '%크기%'  OR review_text LIKE '%크다%'  OR review_text LIKE '%작다%'
            OR review_text LIKE '%작은%'  OR review_text LIKE '%실하%'  OR review_text LIKE '%알차%'
            OR review_text LIKE '%등급%'  OR review_text LIKE '%1+%'    OR review_text LIKE '%한우%'
            OR review_text LIKE '%국산%'  OR review_text LIKE '%원산지%' OR review_text LIKE '%흠집%'
            OR review_text LIKE '%상처%'  OR review_text LIKE '%멍%'    OR review_text LIKE '%벌레%'
            OR review_text LIKE '%이물%'  OR review_text LIKE '%곰팡이%' OR review_text LIKE '%가격%'
            OR review_text LIKE '%저렴%'  OR review_text LIKE '%비싸%'  OR review_text LIKE '%가성비%'
            OR review_text LIKE '%값어치%' OR review_text LIKE '%품질%'  OR review_text LIKE '%퀄리티%'
            OR review_text LIKE '%모양%'  OR review_text LIKE '%색감%'  OR review_text LIKE '%때깔%'
            OR review_text LIKE '%색%'   OR review_text LIKE '%큰%'
        THEN 1 ELSE 0 END AS axis_품질,

        -- 축3: axis_신선도 (Python axis_keywords['신선도'], 26개 키워드)
        -- 신선, 싱싱, 무르, 물렁, 짓무, 상했, 상한, 변질, 쉰, 썩, 곰팡,
        -- 시들, 유통기한, 유통, 임박, 날짜, 생기, 팔팔, 살아있, 비린, 비림,
        -- 비린내, 상함, 신선도, 생생, 살아
        CASE WHEN
            review_text LIKE '%신선%'    OR review_text LIKE '%싱싱%'    OR review_text LIKE '%무르%'
            OR review_text LIKE '%물렁%'    OR review_text LIKE '%짓무%'    OR review_text LIKE '%상했%'
            OR review_text LIKE '%상한%'    OR review_text LIKE '%변질%'    OR review_text LIKE '%쉰%'
            OR review_text LIKE '%썩%'     OR review_text LIKE '%곰팡%'    OR review_text LIKE '%시들%'
            OR review_text LIKE '%유통기한%' OR review_text LIKE '%유통%'   OR review_text LIKE '%임박%'
            OR review_text LIKE '%날짜%'   OR review_text LIKE '%생기%'    OR review_text LIKE '%팔팔%'
            OR review_text LIKE '%살아있%'  OR review_text LIKE '%비린%'   OR review_text LIKE '%비림%'
            OR review_text LIKE '%비린내%'  OR review_text LIKE '%상함%'   OR review_text LIKE '%신선도%'
            OR review_text LIKE '%생생%'   OR review_text LIKE '%살아%'
        THEN 1 ELSE 0 END AS axis_신선도,

        -- 축4: axis_배송 (Python axis_keywords['배송'], 28개 키워드)
        -- 배송, 새벽, 포장, 박스, 아이스팩, 드라이아이스, 보냉, 깨, 파손,
        -- 부러, 흐르, 새, 샘, 터짐, 터져, 찢, 찌그, 도착, 받았, 온도,
        -- 미지근, 따뜻, 차갑, 택배, 기사님, 스티로폼, 냉동, 냉장
        CASE WHEN
            review_text LIKE '%배송%'       OR review_text LIKE '%새벽%'       OR review_text LIKE '%포장%'
            OR review_text LIKE '%박스%'       OR review_text LIKE '%아이스팩%'   OR review_text LIKE '%드라이아이스%'
            OR review_text LIKE '%보냉%'       OR review_text LIKE '%깨%'         OR review_text LIKE '%파손%'
            OR review_text LIKE '%부러%'       OR review_text LIKE '%흐르%'       OR review_text LIKE '%샘%'
            OR review_text LIKE '%터짐%'       OR review_text LIKE '%터져%'       OR review_text LIKE '%찢%'
            OR review_text LIKE '%찌그%'       OR review_text LIKE '%도착%'       OR review_text LIKE '%받았%'
            OR review_text LIKE '%온도%'       OR review_text LIKE '%미지근%'     OR review_text LIKE '%따뜻%'
            OR review_text LIKE '%차갑%'       OR review_text LIKE '%택배%'       OR review_text LIKE '%기사님%'
            OR review_text LIKE '%스티로폼%'   OR review_text LIKE '%냉동%'       OR review_text LIKE '%냉장%'
            OR review_text LIKE '%새%'
        THEN 1 ELSE 0 END AS axis_배송

    FROM negative_reviews
),

-- UNPIVOT: 4개 축 플래그를 행으로 변환 (Python의 중복 분류 구조 재현)
-- Python : axis_배송=1, axis_품질=1이 동시에 가능하고, 각각이 독립 집계 / SQL : UNION ALL로 4개 축을 각각 행으로 분리하여 동일한 효과를 구현
complaint_rows AS (
    SELECT review_id, category, review_length, '맛'    AS complaint_type FROM axis_flags WHERE axis_맛     = 1
    UNION ALL
    SELECT review_id, category, review_length, '품질'  AS complaint_type FROM axis_flags WHERE axis_품질   = 1
    UNION ALL
    SELECT review_id, category, review_length, '신선도' AS complaint_type FROM axis_flags WHERE axis_신선도 = 1
    UNION ALL
    SELECT review_id, category, review_length, '배송'  AS complaint_type FROM axis_flags WHERE axis_배송   = 1
)

-- 최종 집계
SELECT 
    category,
    complaint_type,
    COUNT(1) AS complaint_count,
    ROUND(AVG(review_length), 1) AS avg_complaint_length
FROM complaint_rows
GROUP BY 
    category,
    complaint_type
ORDER BY 
    category,
    complaint_count DESC;
