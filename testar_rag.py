# =========================================================
# testar_rag.py - Testar o pipeline RAG sem Streamlit
# =========================================================

import rag

def linha(char: str = "-", n: int = 60):
    print(char * n)

def testar(perguntas: list[str]):
    """Testa várias perguntas."""
    linha("=")
    print("🧪 TESTE DO PIPELINE RAG")
    linha("=")
    
    try:
        colecao = rag.get_collection()
        print(f"✅ Chunks indexados: {colecao.count()}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("Corre 'python ingest.py' primeiro!")
        return
    
    print(f"Modelo de Embedding: {rag.EMBED_MODEL}")
    print(f"Modelo de Chat: {rag.OLLAMA_LOCAL_MODEL}")
    print(f"TOP_K={rag.TOP_K} | MIN_SIMILARITY={rag.MIN_SIMILARITY}")
    linha()
    
    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n📝 Pergunta {i}: {pergunta}")
        linha("-")
        
        try:
            chunks = rag.search(pergunta)
            
            if not chunks:
                print("⚠️ Nenhum chunk encontrado acima do threshold.")
                print("   (A pergunta pode estar fora do âmbito ou o threshold está muito alto)")
            else:
                for j, c in enumerate(chunks, 1):
                    pct = int(c["similarity"] * 100)
                    if pct >= 70:
                        qualidade = "🌟 excelente"
                    elif pct >= 50:
                        qualidade = "👍 bom"
                    else:
                        qualidade = "⚠️ fraco"
                    
                    print(f"  Chunk {j} - {pct}% ({qualidade})")
                    print(f"  Secção: {c['section']}")
                    print(f"  Ficheiro: {c['source']}")
                    texto_limpo = c['text'][:120].replace('\n', ' ')
                    print(f"  Texto: {texto_limpo}...")
                    print()
        except Exception as e:
            print(f"❌ Erro ao pesquisar: {e}")

if __name__ == "__main__":
    # =============================================
    # PERGUNTAS PARA A ESSÊNCIA VITAL
    # =============================================
    
    PERGUNTAS_DE_TESTE = [
        # --- PERGUNTAS QUE DEVEM TER RESPOSTA ---
        "Qual é a missão da Essência Vital?",
        "Quem é o fundador da clínica?",
        "Que serviços de saúde oferecem?",
        "O que é a Osteopatia?",
        "Qual é a morada da clínica?",
        "Qual é o telefone de contacto?",
        "Quanto custa uma consulta de Osteopatia?",
        "Como posso marcar uma consulta?",
        
        # --- PERGUNTAS QUE NÃO DEVEM TER RESPOSTA ---
        "Qual é a senha do Wi-Fi?",
        "Quanto custa um iPhone 15?",
        "Quem ganhou o Euro 2024?",
        "Qual é a capital da França?",
        "Quanto custa uma consulta de Osteopatia?",
    ]
    
    testar(PERGUNTAS_DE_TESTE)