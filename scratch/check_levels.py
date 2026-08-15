import json

def run():
    with open('data/parsed/ts_123501v171100p.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    texts = d.get('texts', [])
    targets = ['5 ', '5.3 ', '5.3.4 ', '8.2.1.2 ', 'D.2 ', 'A.1 ']
    
    print("Checking specific heading levels:\n")
    found = 0
    for t in texts:
        if t.get('label') == 'section_header':
            text = t.get('text', '')
            if any(text.startswith(tgt) for tgt in targets):
                print(f"Level: {t.get('level', 'None')} | Text: {text}")
                found += 1
                if found > 15:
                    break

if __name__ == '__main__':
    run()
