import json
import re
import time
from pathlib import Path

import requests

API_URL = "https://gameofthrones.fandom.com/api.php"
OUTPUT_DIR = Path("knowledge_base_raw")
DELAY_SECONDS = 1.5

PAGES = [
    # Персонажи
    "Jon Snow",
    "Daenerys Targaryen",
    "Tyrion Lannister",
    "Cersei Lannister",
    "Jaime Lannister",
    "Arya Stark",
    "Sansa Stark",
    "Eddard Stark",
    "Bran Stark",
    "Robb Stark",
    "Theon Greyjoy",
    "Ramsay Bolton",
    "Melisandre",
    "Varys",
    "Night King",
    "Jorah Mormont",
    "Samwell Tarly",
    "Davos Seaworth",
    "Petyr Baelish",
    "Brienne of Tarth",
    "Tormund Giantsbane",
    # Локации
    "Winterfell",
    "King's Landing",
    "The Wall",
    "Dragonstone",
    "Casterly Rock",
    "Braavos",
    "Meereen",
    "The Eyrie",
    # Дома и фракции
    "House Stark",
    "House Lannister",
    "House Targaryen",
    "Night's Watch",
    "Dothraki",
    # Концепции и объекты
    "Iron Throne",
    "Dragon",
    "Valyrian steel",
    "Dragonglass",
    "Direwolf",
    "Wildfire",
    "White Walker",
    "Long Night",
]


def fetch_page(title: str) -> str | None:
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
        "redirects": "1",
    }
    headers = {"User-Agent": "RAG-educational-bot/1.0 (student project)"}

    try:
        resp = requests.get(API_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] Не удалось получить '{title}': {e}")
        return None

    if "error" in data:
        print(f"  [WARN] Страница не найдена: '{title}' — {data['error'].get('info', '')}")
        return None

    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    return wikitext_to_plain(wikitext) if wikitext else None


def wikitext_to_plain(text: str) -> str:
    # Удаляем <ref>...</ref> и самозакрывающиеся <ref/>
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    # Удаляем HTML-комментарии
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Удаляем внешние ссылки [http://... текст] → текст
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # [[File:...]] и [[Image:...]] → убираем
    text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # [[Ссылка|текст]] → текст
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    # [[Ссылка]] → Ссылка
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Шаблоны {{...}} — несколько проходов для вложенности
    for _ in range(6):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Заголовки == ... == → просто текст
    text = re.sub(r"={2,}([^=]+)={2,}", r"\n\1\n", text)
    # Жирный/курсив '''/''
    text = re.sub(r"'{2,}", "", text)
    # Табличная разметка
    text = re.sub(r"^\s*[|!{][|!}].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|[-}].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.MULTILINE)
    # Оставшиеся HTML-теги
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text: str) -> str:
    # Убираем очень длинные блоки пробелов/пустых строк
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Убираем строки, содержащие только символы пунктуации или цифры
    lines = [line for line in text.splitlines() if len(line.strip()) > 3 or line.strip() == ""]
    text = "\n".join(lines)
    return text.strip()


def title_to_filename(title: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title).strip()
    safe = re.sub(r"\s+", "_", safe)
    return f"{safe}.txt"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Скачиваем {len(PAGES)} страниц в '{OUTPUT_DIR}/'...\n")

    success, failed = 0, 0
    for title in PAGES:
        print(f"Скачиваем: {title}")
        text = fetch_page(title)

        if text and len(text) > 200:
            text = clean_text(text)
            filename = OUTPUT_DIR / title_to_filename(title)
            filename.write_text(text, encoding="utf-8")
            print(f"  OK → {filename.name} ({len(text)} символов)")
            success += 1
        else:
            print(f"  SKIP — слишком мало текста или ошибка")
            failed += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nГотово: {success} скачано, {failed} пропущено.")


if __name__ == "__main__":
    main()
