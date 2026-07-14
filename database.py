# =========================================================
# database.py - Guardar e recuperar conversas em SQLite
# =========================================================

import sqlite3
import json
from datetime import datetime

DB_PATH = "./conversas.db"

def init_db():
    """Cria as tabelas se não existirem."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS conversas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            empresa TEXT NOT NULL,
            criada_em TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS mensagens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversa_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            conteudo TEXT NOT NULL,
            fontes TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversa_id) REFERENCES conversas (id)
        );
    """)
    
    conn.commit()
    conn.close()

def criar_conversa(titulo: str, empresa: str) -> int:
    """Cria nova conversa e devolve ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO conversas (titulo, empresa, criada_em) VALUES (?, ?, ?)",
        (titulo, empresa, now)
    )
    
    conversa_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversa_id

def adicionar_mensagem(conversa_id: int, role: str, conteudo: str, fontes=None):
    """Adiciona mensagem à conversa."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    fontes_json = json.dumps(fontes, ensure_ascii=False) if fontes else None
    
    cursor.execute(
        """INSERT INTO mensagens (conversa_id, role, conteudo, fontes, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (conversa_id, role, conteudo, fontes_json, now)
    )
    
    conn.commit()
    conn.close()

def obter_mensagens(conversa_id: int) -> list[dict]:
    """Obtém todas as mensagens de uma conversa, ordenadas por ordem cronológica."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT role, conteudo, fontes, timestamp
           FROM mensagens
           WHERE conversa_id = ?
           ORDER BY id ASC""",
        (conversa_id,)
    )
    
    mensagens = []
    for role, conteudo, fontes_json, timestamp in cursor.fetchall():
        mensagens.append({
            "role": role,
            "content": conteudo,
            "sources": json.loads(fontes_json) if fontes_json else [],
            "timestamp": timestamp
        })
    
    conn.close()
    return mensagens

def listar_conversas() -> list[dict]:
    """Lista todas as conversas, da mais recente para a mais antiga."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.id, c.titulo, c.empresa, c.criada_em, COUNT(m.id) as n
        FROM conversas c
        LEFT JOIN mensagens m ON c.id = m.conversa_id
        GROUP BY c.id
        ORDER BY c.id DESC
    """)
    
    conversas = []
    for id_, titulo, empresa, criada_em, n in cursor.fetchall():
        conversas.append({
            "id": id_,
            "titulo": titulo,
            "empresa": empresa,
            "criada_em": criada_em,
            "num_mensagens": n
        })
    
    conn.close()
    return conversas

def apagar_conversa(conversa_id: int):
    """Apaga uma conversa e todas as suas mensagens."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Apagar mensagens primeiro (por causa da FOREIGN KEY)
    cursor.execute("DELETE FROM mensagens WHERE conversa_id = ?", (conversa_id,))
    cursor.execute("DELETE FROM conversas WHERE id = ?", (conversa_id,))
    
    conn.commit()
    conn.close()

# Inicializar a base de dados quando o módulo é importado
init_db()