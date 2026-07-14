#!/bin/bash

# Instalar dependências (se necessário)
pip install -r requirements.txt

# Executar o ingest para criar a base de dados
python ingest.py