import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
FAISS_STORE_DIR = Path("faiss_store")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name, "title": path.stem.replace("_", " ")},
        ))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        parts = splitter.split_text(doc.page_content)
        for i, part in enumerate(parts):
            chunks.append(Document(
                page_content=part,
                metadata={**doc.metadata, "chunk_id": i},
            ))
    return chunks


def build_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    print(f"Генерируем эмбеддинги для {len(chunks)} чанков...")
    t0 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  Готово за {time.time() - t0:.1f} сек.")
    return vectorstore


def run_search_test(vectorstore: FAISS) -> None:
    test_queries = [
        # English
        "Who is Kael Dorn and what is his true origin?",
        "What is the Stone Throne and who rules from it?",
        # Russian — проверяем мультиязычность
        "Кто такой Каэль Дорн?",
        "Что такое Каменный Трон?",
        "Расскажи о Страже Вуали и их обязанностях",
    ]
    print("\n--- Тест поиска (top-2 чанка) ---")
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\nQ: {query}")
        for r in results:
            preview = r.page_content[:120].replace("\n", " ").strip()
            print(f"  [{r.metadata['source']} / chunk {r.metadata['chunk_id']}] {preview}...")


def main() -> None:
    print(f"1. Загружаем документы из '{KNOWLEDGE_BASE_DIR}/'...")
    raw_docs = load_documents()
    if not raw_docs:
        print("[ERROR] Нет файлов в knowledge_base/. Сначала запустите docker compose up scrape и docker compose up replace_terms.")
        return
    print(f"   Загружено: {len(raw_docs)} файлов")

    print(f"\n2. Разбиваем на чанки (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_documents(raw_docs)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"   Чанков: {len(chunks)}, средняя длина: {avg_len} символов")

    print(f"\n3. Загружаем модель '{EMBEDDING_MODEL}' (dim=384)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\n4. Строим FAISS-индекс...")
    vectorstore = build_index(chunks, embeddings)

    FAISS_STORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(FAISS_STORE_DIR))
    index_files = [f.name for f in FAISS_STORE_DIR.iterdir()]
    print(f"   Индекс сохранён в '{FAISS_STORE_DIR}/': {index_files}")

    run_search_test(vectorstore)

    print("\n✓ Индекс готов. Запустите docker compose up bot для работы с ботом.")


if __name__ == "__main__":
    main()
