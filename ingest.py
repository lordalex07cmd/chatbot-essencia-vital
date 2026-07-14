# =========================================================
# ingest.py - Indexar os documentos da empresa na base de dados vetorial
# =========================================================

import os
import glob
import requests
import chromadb

# --- CONSTANTES ---
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
DOCS_FOLDER = "./documentos"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "empresa_docs"

CHUNK_SIZE = 400          # máximo de palavras por chunk
CHUNK_OVERLAP = 40        # palavras que se repetem entre chunks

def get_embedding(text: str) -> list[float]:
    """
    Converte texto em embedding via Ollama.
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

def chunk_by_sections(content: str, filename: str) -> list[dict]:
    """
    Divide documento usando títulos (##) como separadores.
    """
    chunks = []
    current_section = ""
    current_title = filename

    for line in content.split("\n"):
        if line.startswith("#"):
            # Guardar secção anterior
            if current_section.strip():
                sub_chunks = split_long_section(current_section.strip(), current_title)
                chunks.extend(sub_chunks)
            # Nova secção
            current_title = line.lstrip("#").strip()
            current_section = line + "\n"
        else:
            current_section += line + "\n"

    # Última secção
    if current_section.strip():
        sub_chunks = split_long_section(current_section.strip(), current_title)
        chunks.extend(sub_chunks)

    return chunks

def split_long_section(text: str, section_title: str) -> list[dict]:
    """
    Divide secções longas em sub-chunks com overlap.
    """
    words = text.split()
    
    if len(words) <= CHUNK_SIZE:
        return [{"text": text, "section": section_title}]

    sub_chunks = []
    start = 0
    chunk_num = 0
    
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        sub_chunks.append({
            "text": chunk_text,
            "section": f"{section_title} (parte {chunk_num + 1})"
        })
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP
        chunk_num += 1
    
    return sub_chunks

def ingest_documents():
    """
    Função principal que faz a indexação.
    """
    print("\n=== INDEXADOR DE DOCUMENTOS ===")
    print(f"Pasta: {DOCS_FOLDER}")
    print(f"Modelo: {EMBED_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    # Conectar ao ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Encontrar documentos
    files = glob.glob(os.path.join(DOCS_FOLDER, "*.md")) + \
            glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))
    
    if not files:
        print(f"ERRO: Nenhum documento em '{DOCS_FOLDER}'")
        return
    
    print(f"\nEncontrados {len(files)} ficheiro(s):")
    for f in files:
        print(f" - {os.path.basename(f)}")
    
    # Processar cada ficheiro
    all_ids, all_texts, all_metadata = [], [], []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nA processar: {filename}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = chunk_by_sections(content, filename)
        print(f" - {len(chunks)} chunk(s) criado(s)")
        
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{filename}_chunk{i}")
            all_texts.append(chunk["text"])
            all_metadata.append({
                "source": filename,
                "section": chunk["section"],
                "chunk_index": i
            })
    
    # Verificar duplicados
    print(f"\nA verificar duplicados...")
    try:
        existing = collection.get(ids=all_ids)
        existing_ids = set(existing["ids"])
    except Exception:
        existing_ids = set()
    
    new_ids = [id for id in all_ids if id not in existing_ids]
    new_texts = [t for id, t in zip(all_ids, all_texts) if id not in existing_ids]
    new_metas = [m for id, m in zip(all_ids, all_metadata) if id not in existing_ids]
    
    if not new_ids:
        print(f"Todos os chunks já estão indexados. Total: {collection.count()}")
        return
    
    print(f"A indexar {len(new_ids)} chunk(s) novos...")
    
    # Gerar embeddings em lotes
    BATCH_SIZE = 5
    total_lotes = (len(new_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(new_ids), BATCH_SIZE):
        b_ids = new_ids[i:i + BATCH_SIZE]
        b_texts = new_texts[i:i + BATCH_SIZE]
        b_metas = new_metas[i:i + BATCH_SIZE]
        
        print(f" Lote {i // BATCH_SIZE + 1}/{total_lotes}: {len(b_ids)} embedding(s)...")
        
        b_embeddings = [get_embedding(text) for text in b_texts]
        
        collection.add(
            embeddings=b_embeddings,
            documents=b_texts,
            ids=b_ids,
            metadatas=b_metas
        )
    
    print(f"\n=== INDEXAÇÃO CONCLUÍDA! ===")
    print(f" Total de chunks: {collection.count()}")
    print(f" Agora podes correr: streamlit run app.py")

if __name__ == "__main__":
    ingest_documents()