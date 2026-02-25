import json
import re
from pathlib import Path

TERMS_MAP_PATH = Path("terms_map.json")
INPUT_DIR = Path("knowledge_base_raw")
OUTPUT_DIR = Path("knowledge_base")


def load_terms(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    terms = [
        (src, dst)
        for src, dst in raw.items()
        if not src.startswith("_") and dst != "---"
    ]
    # Сортировка по убыванию длины ключа
    terms.sort(key=lambda x: len(x[0]), reverse=True)
    return terms


def apply_replacements(text: str, terms: list[tuple[str, str]]) -> str:
    for src, dst in terms:
        text = text.replace(src, dst)
    return text


def main():
    if not TERMS_MAP_PATH.exists():
        print(f"[ERROR] Файл {TERMS_MAP_PATH} не найден.")
        return
    if not INPUT_DIR.exists():
        print(f"[ERROR] Папка {INPUT_DIR} не найдена. Сначала запустите scrape.py.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    terms = load_terms(TERMS_MAP_PATH)
    print(f"Загружено {len(terms)} замен. Обрабатываем файлы из '{INPUT_DIR}/'...\n")

    files = list(INPUT_DIR.glob("*.txt")) + list(INPUT_DIR.glob("*.md"))
    if not files:
        print("[WARN] Нет файлов для обработки.")
        return

    for path in sorted(files):
        text = path.read_text(encoding="utf-8")
        replaced = apply_replacements(text, terms)

        out_path = OUTPUT_DIR / path.name
        out_path.write_text(replaced, encoding="utf-8")

        changed = sum(1 for s, d in terms if s in text)
        print(f"  {path.name} → {out_path.name}  (применено замен: {changed})")

    print(f"\nГотово: {len(files)} файлов сохранены в '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
