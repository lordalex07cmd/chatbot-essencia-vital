# =========================================================
# app.py - Chatbot empresarial com interface web Streamlit
# =========================================================

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

# =========================================================
# ⚠️ CONFIGURAÇÃO DA PÁGINA - DEVE VIR ANTES DE QUALQUER OUTRO st.xxx
# =========================================================
st.set_page_config(
    page_title="Essência Vital - Chatbot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar cabeçalho para permitir iframe (CORS)
st.markdown("""
    <meta http-equiv="Content-Security-Policy" content="frame-ancestors *;">
""", unsafe_allow_html=True)

# =========================================================
# IMPORTAR MÓDULOS DEPOIS DO st.set_page_config
# =========================================================
try:
    import database as db
    import rag
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.stop()

# Carregar variáveis de ambiente
load_dotenv()

# =========================================================
# FUNÇÃO DE INGESTÃO - SEM SUBPROCESS
# =========================================================
def verificar_e_ingerir():
    """
    Verifica se a base de dados existe e executa ingest se necessário.
    Versão SEM subprocess para compatibilidade com Streamlit Cloud.
    """
    try:
        # Verificar se o diretório chroma_db existe
        if os.path.exists("./chroma_db"):
            # Verificar se tem conteúdo
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            try:
                collections = client.list_collections()
                if collections:
                    st.success("✅ Base de dados carregada com sucesso!")
                    return True
                else:
                    st.warning("⚠️ Base de dados vazia. A ingerir documentos...")
            except Exception:
                st.warning("⚠️ Base de dados corrompida. A recriar...")
        
        # Executar ingest diretamente (sem subprocess)
        with st.spinner("🔍 A indexar documentos pela primeira vez..."):
            try:
                # Importar a função main do ingest
                from ingest import main as ingest_main
                ingest_main()
                st.success("✅ Documentos indexados com sucesso!")
                return True
            except ImportError:
                st.error("❌ Arquivo ingest.py não encontrado!")
                return False
            except Exception as e:
                st.error(f"❌ Erro durante a ingestão: {str(e)}")
                return False
                
    except Exception as e:
        st.error(f"❌ Erro ao verificar base de dados: {str(e)}")
        return False

# =========================================================
# VERIFICAR BASE DE DADOS
# =========================================================
if "ingest_done" not in st.session_state:
    st.session_state.ingest_done = verificar_e_ingerir()

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def get_api_key(provider: str) -> str | None:
    """Obtém API key para o provider escolhido."""
    if provider == "ollama_local":
        return "ollama"  # O Ollama ignora a key, mas o cliente OpenAI exige uma
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return None

def format_sources(sources: list[dict]) -> str:
    """Formata as fontes como texto Markdown para mostrar na UI."""
    if not sources:
        return ""
    
    lines = []
    for s in sources:
        pct = int(s.get("similarity", 0) * 100)
        source = s.get("source", "Fonte desconhecida")
        section = s.get("section", "Secção não especificada")
        lines.append(f"• **{section}** ({source}) – relevância: {pct}%")
    return "\n".join(lines)

def load_history_for_llm(conversa_id: int) -> list[dict]:
    """Carrega o histórico no formato que o LLM espera."""
    try:
        mensagens = db.obter_mensagens(conversa_id)
        return [
            {"role": m["role"], "content": m["content"]}
            for m in mensagens
            if m["role"] in ("user", "assistant")
        ]
    except Exception:
        return []

def init_session():
    """Inicializa o estado da sessão."""
    if "conversa_id" not in st.session_state:
        st.session_state.conversa_id = None
    if "mensagens_display" not in st.session_state:
        st.session_state.mensagens_display = []

# =========================================================
# RENDERIZAÇÃO DA BARRA LATERAL
# =========================================================
def render_sidebar():
    """Renderiza a barra lateral com configurações e histórico."""
    with st.sidebar:
        st.title("⚙️ Configuração")
        
        # Status da base de dados
        if st.session_state.get("ingest_done", False):
            st.success("✅ Base de dados: OK")
        else:
            st.warning("⚠️ Base de dados: Não carregada")
            if st.button("🔄 Tentar recarregar"):
                st.session_state.ingest_done = verificar_e_ingerir()
                st.rerun()
        
        st.divider()
        
        # Nome da empresa
        st.subheader("🏢 Empresa")
        company_name = st.text_input(
            "Nome da empresa",
            value="Essência Vital",
            key="company_name_input"
        )
        
        # Modelo LLM
        st.subheader("🧠 Modelo de Linguagem")
        PROVIDER_LABELS = {
            "groq": "Groq - llama-3.3-70b (GRÁTIS)",
            "ollama_local": "Ollama local - offline (GRÁTIS)",
            "anthropic": "Anthropic - Claude Haiku (pago)",
            "openai": "OpenAI - GPT-4o-mini (pago)",
        }
        
        provider = st.selectbox(
            "Fornecedor",
            options=list(PROVIDER_LABELS.keys()),
            format_func=lambda x: PROVIDER_LABELS[x],
            key="provider_select"
        )
        
        # API Key
        api_key = get_api_key(provider)
        if provider == "ollama_local":
            st.success("✅ Sem API key necessária")
            ollama_model = st.text_input(
                "Modelo Ollama",
                value=rag.OLLAMA_LOCAL_MODEL if hasattr(rag, "OLLAMA_LOCAL_MODEL") else "llama3.2",
                help="Corre 'ollama list' para ver os modelos instalados",
                key="ollama_model_input"
            )
            if hasattr(rag, "OLLAMA_LOCAL_MODEL"):
                rag.OLLAMA_LOCAL_MODEL = ollama_model
        elif api_key:
            st.success("✅ API key carregada do .env")
        else:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="gsk_... (Groq) ou sk-ant-... (Anthropic) ou sk-... (OpenAI)",
                key="api_key_input"
            )
        
        # Botão Nova Conversa
        st.divider()
        st.subheader("💬 Conversa")
        if st.button("➕ Nova conversa", use_container_width=True):
            titulo = f"Conversa {datetime.now().strftime('%d/%m %H:%M')}"
            try:
                cid = db.criar_conversa(titulo, company_name)
                st.session_state.conversa_id = cid
                st.session_state.mensagens_display = []
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao criar conversa: {e}")
        
        # Histórico de conversas
        st.divider()
        st.subheader("📜 Histórico")
        try:
            conversas = db.listar_conversas()
            
            if not conversas:
                st.info("Nenhuma conversa anterior")
            else:
                for conv in conversas:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            f"{conv.get('titulo', 'Conversa')} ({conv.get('num_mensagens', 0)} msgs)",
                            key=f"conv_{conv['id']}",
                            use_container_width=True
                        ):
                            st.session_state.conversa_id = conv['id']
                            st.session_state.mensagens_display = db.obter_mensagens(conv['id'])
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['id']}"):
                            db.apagar_conversa(conv['id'])
                            st.rerun()
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar histórico: {e}")
        
    return company_name, provider, api_key

# =========================================================
# RENDERIZAÇÃO DO CHAT
# =========================================================
def render_chat(company_name: str, provider: str, api_key: str):
    """Renderiza a área principal do chat."""
    st.title(f"🌿 {company_name}")
    st.caption("Assistente virtual com base de conhecimento RAG")
    
    # Verificações
    if not st.session_state.get("ingest_done", False):
        st.warning("⚠️ Base de dados não carregada. Tente recarregar na barra lateral.")
        return
    
    if not st.session_state.conversa_id:
        st.info("💡 Clica em **Nova conversa** na barra lateral para começar.")
        return
    
    if not api_key and provider != "ollama_local":
        st.warning("⚠️ Insere uma API key na barra lateral (ou cria um ficheiro .env).")
        return
    
    # Mostrar status da base de dados
    try:
        col = rag.get_collection()
        st.caption(f"📚 Base de conhecimento: {col.count()} secções indexadas")
    except RuntimeError as e:
        st.error(str(e))
        return
    except Exception:
        st.warning("⚠️ Não foi possível aceder à base de dados.")
        return
    
    # Mostrar mensagens
    for msg in st.session_state.mensagens_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📖 Fontes utilizadas", expanded=False):
                    st.markdown(format_sources(msg["sources"]))
    
    # Campo de input
    if prompt := st.chat_input("Escreve a tua pergunta..."):
        # Mostrar pergunta
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Guardar pergunta na BD
        try:
            db.adicionar_mensagem(st.session_state.conversa_id, "user", prompt)
        except Exception:
            pass  # Ignorar erro se não conseguir guardar
        
        st.session_state.mensagens_display.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        
        # Gerar resposta
        with st.chat_message("assistant"):
            with st.spinner("🔍 A pesquisar e a gerar resposta..."):
                try:
                    history = load_history_for_llm(st.session_state.conversa_id)
                    history = history[:-1] if history else []
                    
                    result = rag.answer(
                        question=prompt,
                        history=history,
                        company_name=company_name,
                        provider=provider,
                        api_key=api_key
                    )
                    
                    st.markdown(result["response"])
                    if result.get("sources"):
                        with st.expander("📖 Fontes utilizadas", expanded=False):
                            st.markdown(format_sources(result["sources"]))
                    
                    # Guardar resposta na BD
                    try:
                        db.adicionar_mensagem(
                            st.session_state.conversa_id,
                            "assistant",
                            result["response"],
                            fontes=result.get("sources", [])
                        )
                    except Exception:
                        pass
                    
                    st.session_state.mensagens_display.append({
                        "role": "assistant",
                        "content": result["response"],
                        "sources": result.get("sources", [])
                    })
                except Exception as e:
                    st.error(f"❌ Erro ao gerar resposta: {str(e)}")
                    st.info("💡 Tenta verificar a API key ou o modelo selecionado.")

# =========================================================
# MAIN
# =========================================================
def main():
    init_session()
    company_name, provider, api_key = render_sidebar()
    render_chat(company_name, provider, api_key)

if __name__ == "__main__":
    main()