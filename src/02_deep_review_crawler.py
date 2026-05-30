import requests, pandas as pd, time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = BASE_DIR / 'raw_reviews_kurly.csv'

df = pd.read_csv(RAW_CSV, encoding='utf-8-sig')
df['product_no'] = df['product_no'].astype(str)
seen = set(df['review_text'].dropna())
reviews = df.to_dict('records')

pc = df.groupby('product_no').size().sort_values(ascending=False)
targets = [(pno, df[df['product_no']==pno]['segment'].iloc[0]) for pno, count in pc.items() if count >= 50][:15]

print(f'\n깊은 크롤링: {len(targets)}개 상품\n')

HEADERS = {'User-Agent': 'Mozilla/5.0'}
URL = 'https://api.kurly.com/product-review/v3/contents-products/{}/reviews'
added = 0

for pno, seg in targets:
    for pg in range(1, 4):
        try:
            r = requests.get(URL.format(pno), params={'sortType':'RECOMMEND','size':100,'onlyImage':'false','filters':''}, headers=HEADERS, timeout=10)
            if r.status_code != 200: break
            
            data = r.json()
            revs = data.get('data', []) if isinstance(data.get('data'), list) else []
            if not revs: break
            
            for rv in revs:
                body = rv.get('contents', '').strip()
                if body and len(body) > 3 and body not in seen:
                    reviews.append({'product_no': pno, 'review_text': body, 'segment': seg})
                    seen.add(body)
                    added += 1
            
            time.sleep(0.5)
        except: break
    
    print(f'[{pno}] 추가: {added}건')

pd.DataFrame(reviews).to_csv(RAW_CSV, index=False, encoding='utf-8-sig')
print(f'\n완료! 총 {len(reviews)}건 (추가: {added}건)')
