import requests, pandas as pd, time, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

SEGMENTS = {'과일':{'target':2400,'per_product':70},'육류':{'target':2000,'per_product':65},'채소':{'target':2000,'per_product':55},'수산':{'target':1200,'per_product':120},'계란':{'target':400,'per_product':10}}
MANUAL_PRODUCTS = {'과일':['5061259','1001303421','1000122762','5005943','1000908371','1001882653','5049096','1000065953','1001847779','1000343040','1000914411','5030220','5060606','1000873951','1001811209','5048935','1000350168','1000942891','5065323','1001848684','5062680','5106292','5065480','1001801374','1001801376','1001801377','1001801382','1001801369','5035027','1001801367','1001801372','1001801375','1001801379'],'육류':['5055607','1001335683','5010944','5054439','5067015','5103614','5051043','1000433103','1001201257','5054430','5037694','5052551','5009479','5065334','5051046','1000669892','1001351875','1000892007','5062534','1000630278','1001249381','5043074','1000662115','1001251426','5050080','5090106','1000892003','1000892005','5006172','5055609','5054441','5059981','5057636'],'채소':['5063690','5063578','5066038','5029438','5000100','1001234463','1000365252','1000147414','1000357035','1000376854','5136653','5006032','5029436','5067909','1000537255','5049265','5031060','5063866','1000956901','5049245','5157163','1001392267','1000127040','5000099','1001188148','1000220028','1000973395','5063864','1000479067','1001971868','1001881389','5002974','1001495528','5132941','1001137981','1002034192','1001341778','1001971858'],'수산':['1000750847','1000636870','1000636872','5138716','1000165819','5053884','5066649','1000636875','1000612746','1001451412'],'계란':['5056791','5029850','1000127449','5151849','5151851','1000179412','5031086','5119904','5151848','1001058180','1000206115','5119903','1000050986','1001293819','5029849','5115293','5004951','1000363410','1000137054','1001478560','1001478558','5093936','5038290','5034943','1001703856','5053113','5155499','5138999','1001998058','1001998060','1001478556','1000318940','1001971302','5031099','5152664','1000378376','5056790','5050055','5067039','1001365218','1001995045','1001365220','5041405','5038008','5054981','1001693706','5078347','5099478','5067038','1001384001','5000093','1001384003']}

os.makedirs(BASE_DIR / "reports", exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REVIEW_URL = "https://api.kurly.com/product-review/v3/contents-products/{product_no}/reviews"
RESULT_FILE = BASE_DIR / "raw_reviews_kurly.csv"

print("\n" + "="*60)
print("마켓컬리 크롤러 시작")
print("="*60)

all_reviews = []
if os.path.exists(RESULT_FILE):
    df = pd.read_csv(RESULT_FILE, encoding="utf-8-sig")
    all_reviews = df.to_dict("records")
    print(f"\n[재개] 기존 {len(all_reviews)}건 로드")

seen_texts = set(r.get("review_text", "") for r in all_reviews)

for seg in SEGMENTS:
    target = SEGMENTS[seg]["target"]
    current = len([r for r in all_reviews if r.get("segment") == seg])
    print(f"\n[{seg}] {current}/{target}건")
    
    for pno in MANUAL_PRODUCTS.get(seg, []):
        if len([r for r in all_reviews if r.get("segment") == seg]) >= target:
            break
        print(f"  [{pno}]...", end=" ", flush=True)
        
        for pg in range(1, 3):
            try:
                params = {"sortType": "RECOMMEND", "size": 100, "onlyImage": "false", "filters": ""}
                resp = requests.get(REVIEW_URL.format(product_no=pno), params=params, headers=HEADERS, timeout=10)
                if resp.status_code != 200:
                    break
                
                data = resp.json()
                reviews = data.get("data", []) if isinstance(data.get("data"), list) else []
                if not reviews:
                    break
                
                new_count = 0
                for r in reviews:
                    body = r.get("contents", "").strip()
                    if body and len(body) > 3 and body not in seen_texts:
                        all_reviews.append({"product_no": pno, "review_text": body, "segment": seg})
                        seen_texts.add(body)
                        new_count += 1
                
                print(f"수집:{len(reviews)} 신규:{new_count}", end=" ")
                time.sleep(0.6)
            except Exception as e:
                print(f"오류:{e}", end=" ")
                break
        
        print()
        if len(all_reviews) % 100 == 0:
            pd.DataFrame(all_reviews).to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

df = pd.DataFrame(all_reviews)
if not df.empty:
    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {RESULT_FILE} ({len(df)}건)")
    for seg in SEGMENTS:
        cnt = len(df[df["segment"] == seg])
        tgt = SEGMENTS[seg]["target"]
        print(f"  {seg}: {cnt}/{tgt}")
