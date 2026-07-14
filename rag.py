# =========================================================
# rag.py - Pesquisa RAG e geração de respostas com LLM
# =========================================================

import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- CONSTANTES ---
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "empresa_docs"

# Modelos para cada provider
GROQ_MODEL = "llama-3.3-70b-versatile"
# ALTERAÇÃO 1: Certifica-te que o modelo de chat está correto
# Se descarregaste qwen2.5:3b, mantém assim
# Se descarregaste outro, muda para o nome correto
OLLAMA_LOCAL_MODEL = "qwen2.5:3b"  # ← ALTERA AQUI SE NECESSÁRIO

TOP_K = 3
MIN_SIMILARITY = 0.65

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

def get_collection():
    """
    Obtém a coleção ChromaDB existente.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            f"Base de dados não encontrada em '{CHROMA_PATH}'.\n"
            "Solução: corre 'python ingest.py' primeiro."
        )

def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Encontra chunks relevantes para uma pergunta.
    """
    collection = get_collection()
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - dist
        if similarity >= MIN_SIMILARITY:
            chunks.append({
                "text": doc,
                "source": meta.get("source", "desconhecido"),
                "section": meta.get("section", ""),
                "similarity": round(similarity, 3)
            })
    return chunks

def build_context(chunks: list[dict]) -> str:
    """
    Formata chunks recuperados como contexto para o LLM.
    """
    if not chunks:
        return "Nenhuma informação relevante encontrada."
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Fonte {i}: {chunk['source']} - {chunk['section']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)

def _build_system_prompt(company_name: str, context: str) -> str:
    """
    System prompt partilhado por todos os providers.
    """
    return f"""
És o assistente virtual da {company_name}, uma clínica de saúde, beleza e bem-estar.

**REGRAS OBRIGATÓRIAS - LÊ COM ATENÇÃO:**

1. **USA APENAS a informação do CONTEXTO.** NUNCA inventes serviços, preços ou informações que não estejam no CONTEXTO.

2. **Se a informação estiver no CONTEXTO, USA-A.** Não digas que não tens informação se ela está disponível.

3. **Se a informação NÃO estiver no CONTEXTO**, diz:
   "Não tenho essa informação disponível. Para mais detalhes, contacte-nos através do email corp.fonseca@gmail.com ou do telefone +351 968 680 721."

4. **Responde SEMPRE em português europeu.**

5. **Sê caloroso, empático e profissional.** A tua voz deve transmitir cuidado e bem-estar.

6. **NUNCA dás conselhos médicos ou diagnósticos.** Não substituis um médico.

7. **Para situações de emergência**, instrui SEMPRE a ligar 112 (INEM).

8. **LISTA APENAS OS SERVIÇOS MENCIONADOS NO CONTEXTO.** Não adiciones serviços que não estejam lá.

9. **Quando relevante, menciona os contactos da clínica:**
   - Telefone: +351 968 680 721
   - Email: corp.fonseca@gmail.com
   - Website: alexandrefonseca.pt

10. **Sê conciso e direto.** Responde de forma clara e objetiva e de forma amigável, sem rodeios.

# LIMITES DO ASSISTENTE

Este assistente NÃO RESPONDE a perguntas sobre:

1. Conselhos médicos ou diagnósticos
2. Descontos não mencionados nos documentos
3. Cálculos matemáticos
4. Viagens, geografia ou turismo
5. Política, desporto ou entretenimento
6. Tecnologia ou produtos eletrónicos

Quando perguntarem sobre estes temas, o assistente DEVE recusar educadamente.

---

**CONTEXTO (informação da empresa - USA APENAS ISTO):**
{context}

---

**REGRA DE OURO:** Se o utilizador pergunta sobre serviços, preços, contactos ou horários, e o CONTEXTO tem essa informação, RESPONDE COM A INFORMAÇÃO DO CONTEXTO. Se o CONTEXTO não tem, RECUSA EDUCADAMENTE.
"""

def generate_response_groq(question, context, history, company_name, api_key):
    """Gera resposta via Groq."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    system_prompt = _build_system_prompt(company_name, context)
    
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content

def generate_response_ollama_local(question, context, history, company_name):
    """Gera resposta via Ollama local."""
    from openai import OpenAI
    
    # ALTERAÇÃO 2: Usar o cliente OpenAI com base_url apontando para o Ollama
    client = OpenAI(api_key="ollama", base_url=f"{OLLAMA_URL}/v1")
    system_prompt = _build_system_prompt(company_name, context)
    
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    
    # ALTERAÇÃO 3: Adicionar try/except para dar erro mais claro
    try:
        response = client.chat.completions.create(
            model=OLLAMA_LOCAL_MODEL,
            messages=messages,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Erro ao chamar o Ollama: {e}\nVerifica se o modelo '{OLLAMA_LOCAL_MODEL}' está instalado.")

def answer(question, history, company_name, provider, api_key) -> dict:
    """
    Pipeline RAG completo: Retrieve → Augment → Generate.
    """
    # 1. RETRIEVE: buscar chunks relevantes
    chunks = search(question)
    
    # 2. AUGMENT: construir contexto
    context = build_context(chunks)
    
    # 3. GENERATE: gerar resposta
    if provider == "groq":
        response = generate_response_groq(question, context, history, company_name, api_key)
    elif provider == "ollama_local":
        response = generate_response_ollama_local(question, context, history, company_name)
    else:
        raise ValueError(f"Provider desconhecido: {provider}")
    
    return {
        "response": response,
        "sources": chunks
    }