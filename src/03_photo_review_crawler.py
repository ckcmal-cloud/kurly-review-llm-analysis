import requests, pandas as pd, time, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / 'raw_reviews_kurly.csv'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
TARGET = 8000

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df['product_no'] = df['product_no'].astype(str)
seen_texts = set(df['review_text'].dropna().str.strip())
new_rows = []

products = df.groupby(['product_no', 'segment']).size().reset_index()
products.columns = ['product_no', 'segment', 'existing']
products = products.sort_values('existing', ascending=False)

print(f"기존 리뷰: {len(df)}건 | 목표: {TARGET}건 | 필요: {TARGET - len(df)}건")
print(f"대상 상품: {len(products)}개\n")

total_added = 0
for idx, row in products.iterrows():
    pno = row['product_no']
    seg = row['segment']
    existing = row['existing']
    
    url = f'https://api.kurly.com/product-review/v3/contents-products/{pno}/reviews'
    
    try:
        r = requests.get(url, params={'sortType': 'RECOMMEND', 'size': 100, 'onlyImage': 'true', 'filters': ''}, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            continue
        data = r.json()
        reviews = data.get('data', [])
        
        added = 0
        for rv in reviews:
            text = str(rv.get('contents', '')).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            new_rows.append({
                'product_no': pno,
                'segment': seg,
                'review_text': text,
                'registered_at': rv.get('registeredAt', ''),
                'source': 'photo'
            })
            added += 1
            total_added += 1
        
        current_total = len(df) + total_added
        print(f"[{seg:4s}] {pno} | 기존:{existing:3d} 포토신규:{added:3d} | 누적:{current_total}건")
        
        if current_total >= TARGET:
            print(f"\n목표 {TARGET}건 달성!")
            break
        
        time.sleep(0.3)
    except Exception as e:
        print(f"[오류] {pno}: {e}")

print(f"\n추가 수집: {total_added}건")

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {len(df_combined)}건 → {CSV_PATH}")
    
    print("\n[세그먼트별 현황]")
    for seg, cnt in df_combined.groupby('segment').size().items():
        print(f"  {seg}: {cnt}건")
else:
    print("추가된 리뷰 없음")
