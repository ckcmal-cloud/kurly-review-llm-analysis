import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_ORIGINAL = BASE_DIR / 'raw_reviews_kurly.csv'
CSV_TRIMMED = BASE_DIR / 'raw_reviews_kurly_trimmed.csv'

df = pd.read_csv(CSV_ORIGINAL, encoding='utf-8-sig')
df['review_text'] = df['review_text'].fillna('').astype(str)
df['text_len'] = df['review_text'].str.strip().str.len()

targets = {'과일': 1605, '육류': 1800, '채소': 2000, '수산': 636, '계란': 400}

print("=" * 70)
print(f"[원본] 총 {len(df)}건")
for seg, cnt in df.groupby('segment').size().items():
    print(f"  {seg}: {cnt}건")
print("=" * 70)

result_parts = []
for seg, target in targets.items():
    seg_df = df[df['segment'] == seg].copy()
    before = len(seg_df)
    
    if before <= target:
        result_parts.append(seg_df)
        print(f"[{seg}] {before}건 유지 (목표 {target} 이하)")
        continue
    
    need_to_remove = before - target
    short = seg_df[seg_df['text_len'] <= 10].sort_values('text_len')
    removed_short = min(len(short), need_to_remove)
    
    if removed_short >= need_to_remove:
        remove_idx = short.head(need_to_remove).index
        kept = seg_df.drop(remove_idx)
    else:
        kept = seg_df.drop(short.index)
        still_need = need_to_remove - removed_short
        extra_remove = kept.sort_values('text_len').head(still_need).index
        kept = kept.drop(extra_remove)
    
    result_parts.append(kept)
    print(f"[{seg}] {before} → {len(kept)}건 (제거:{before-len(kept)}, 짧은리뷰({len(short)}중:{removed_short}건))")

df_trimmed = pd.concat(result_parts, ignore_index=True)
df_trimmed = df_trimmed.drop(columns=['text_len'])
df_trimmed.to_csv(CSV_TRIMMED, index=False, encoding='utf-8-sig')

print("=" * 70)
print(f"[트림 후] 총 {len(df_trimmed)}건 → {CSV_TRIMMED}")
for seg, cnt in df_trimmed.groupby('segment').size().items():
    target = targets.get(seg, 0)
    print(f"  {seg}: {cnt}건 (목표 {target})")
print("\n✓ 원본 유지: raw_reviews_kurly.csv")
print("✓ 새파일:   raw_reviews_kurly_trimmed.csv")
