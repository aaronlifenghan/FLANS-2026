import wikipediaapi
import nltk
import re
nltk.download('punkt')
nltk.download('punkt_tab')

USER_AGENT = "SaraRAGProject/1.0 (https://github.com/SaraSun01/SaraSun01)"

wiki_zh = wikipediaapi.Wikipedia(
    language='zh',
    user_agent=USER_AGENT
)

wiki_en = wikipediaapi.Wikipedia(
    language='en',
    user_agent=USER_AGENT
)

wiki_es = wikipediaapi.Wikipedia(
    language='es',
    user_agent=USER_AGENT
)

zh_topics = [
    "中华人民共和国",
    "中国野生动物",
    "端午节",
    "中华人民共和国国庆节",
    "长江",
    "黄河",
    "中华人民共和国国旗",
    "中华人民共和国国歌",
    "春节",
    "中秋节",
    "北京",
    "熊猫",
    "珠穆朗玛峰",
    "中国足球",
    "中国菜",
    "长江三峡",
    "新加坡",
    "鱼尾狮",
    "新加坡樟宜机场",
    "新加坡国庆日",
    "濱海藝術中心",
    "新加坡元",
    "新加坡国旗",
    "新加坡国歌",
    "新加坡公共假期",
    "建屋發展局",        
    "组屋",              
    "濱海灣花園",
    "牛車水",
    "乌节路",
    ]

en_topics = [
    "Culture_of_the_United_States",
    "Culture_of_the_United_Kingdom",
    "Culture_of_Australia",
    "Public_holidays_in_the_United_States",
    "Public_holidays_in_the_United_Kingdom",
    "Public_holidays_in_Australia",
    "Sports_in_the_United_States",
    "Sport_in_the_United_Kingdom",
    "Sport_in_Australia",
    "United States",
    "United Kingdom",
    "Australia",
    "Blue Peter",
    "Battle of Hastings",
    "Cumbria",
    "Lake District",
    "London",
    "British cuisine",
    "History of the United Kingdom",
    "Bank holiday",
    "Children's television series",
    "Australian rules football",
    "Meat pie",
    "First Fleet",
    "Crocodile Dundee",
    "Australian English",
    "History of Australia",
    "Outback",
    "Culture of Australia",        
    "Sport in Australia",          
    "Brisbane Lions",
    "Fitzroy Football Club",
    ]

es_topics = [
    "España",
    "Cocina_de_España",
    "Deporte_en_España",
    "Educación_en_España",
    "Semana_Santa_en_España",
    "Conciliación_de_la_vida_familiar_y_laboral",
    "México",
    "Ecuador",
    "Bandera de España",
    "Marcha Real",        
    "Don Quijote de la Mancha",
    "Miguel de Cervantes",
    "Flamenco",
    "Acueducto de Segovia",
    "Juegos Olímpicos de Barcelona 1992",
    "Semana Santa en Sevilla",
    "Fútbol en España",
    "Fiesta Nacional de España",
    "Bandera de México",
    "Himno Nacional Mexicano",
    "Ciudad de México",
    "Día de Muertos",
    "Mariachi",
    "Gastronomía de México",
    "Fútbol en México",
    "Fiestas Patrias de México",
    "Bandera de Ecuador",
    "Quito",
    "Guayaquil",
    "Provincias de Ecuador",
    "Chimborazo",
    "Deporte en Ecuador",
    "Quito",
    ]

import re
import nltk

def extract_intro_sentences(
    wiki,
    topics,
    lang,
    max_chars=2000,      # 每个词条最多取多少字符（防止太长）
    max_sents=20         # 每个词条最多取多少句（防止句子太多）
):
    sentences = []

    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            print(f"[WARN] Page not found: {wiki.language}:{topic}")
            continue

        # 1) 用 summary（导语）优先；没有 summary 再退回全文
        text = (page.summary or page.text or "").strip()
        if not text:
            continue

        # 2) 截断：避免太长导致噪音+embedding慢
        text = text[:max_chars]

        # 3) 分句：中文 / 英文 / 西语分别处理
        if lang == "zh":
            # 中文简单分句：按 。！？；换行 切
            parts = re.split(r"[。！？；;\n]+", text)
            parts = [p.strip() for p in parts if len(p.strip()) > 5]
            sentences.extend(parts[:max_sents])

        elif lang == "en":
            sents = nltk.sent_tokenize(text, language="english")
            sents = [s.strip() for s in sents if len(s.strip()) > 5]
            sentences.extend(sents[:max_sents])

        elif lang == "es":
            sents = nltk.sent_tokenize(text, language="spanish")
            sents = [s.strip() for s in sents if len(s.strip()) > 5]
            sentences.extend(sents[:max_sents])

    return sentences

zh_sents = extract_intro_sentences(wiki_zh, zh_topics, "zh")
en_sents = extract_intro_sentences(wiki_en, en_topics, "en")
es_sents = extract_intro_sentences(wiki_es, es_topics, "es")

def clean_sentences(sents):
    cleaned = []
    for s in sents:
        s = s.strip()
        if len(s) > 5 and not s.startswith("This article"):
            cleaned.append(s)
    return list(set(cleaned))

zh_clean = clean_sentences(zh_sents)
en_clean = clean_sentences(en_sents)
es_clean = clean_sentences(es_sents)

def save_txt(filename, sents):
    with open(filename, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s + "\n")

save_txt("knowledge_base/culture_zh_wiki01.txt", zh_clean)
save_txt("knowledge_base/culture_en_wiki01.txt", en_clean)
save_txt("knowledge_base/culture_es_wiki01.txt", es_clean)

print("culture_zh_wiki:", len(zh_clean))
print("culture_en_wiki:", len(en_clean))
print("culture_es_wiki:", len(es_clean))
