import fitz
import re 
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.documents import Document
import boto3
from botocore.exceptions import ClientError

def ler_e_unir_arquivos_s3(bucket_nome: str, prefixo_arquivos: str, separador: str = '\n\n') -> str:
    """
    Lê o conteúdo de todos os arquivos em um bucket S3 com um dado prefixo
    e os concatena em uma única string.
    Args:
        bucket_nome (str): O nome do bucket S3.
        prefixo_arquivos (str): O prefixo (caminho) para listar os arquivos (ex: 'data/logs/').
        separador (str, opcional): O separador a ser usado entre o conteúdo de cada arquivo.
                                    Padrão é duas quebras de linha ('\n\n').
    Returns:
        str: Uma string contendo o conteúdo unido de todos os arquivos.
    """
    
    # Lista para armazenar o conteúdo de cada arquivo
    conteudos_arquivos = []
    
    # 1. Configurar o Cliente S3
    try:
        s3_client = boto3.client('s3')
    except Exception as e:
        print(f"Erro ao inicializar o cliente S3: {e}")
        return ""

    try:
        # 2. Listar Objetos no Bucket
        # O paginate ajuda a lidar com um grande número de arquivos (mais de 1000)
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_nome, Prefix=prefixo_arquivos)

        # 3. Iterar sobre os arquivos e ler o conteúdo
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    chave_objeto = obj['Key']
                    
                    # Ignorar o próprio prefixo se for listado como um objeto (pasta 'vazia')
                    if chave_objeto.endswith('/') and chave_objeto == prefixo_arquivos:
                        continue

                    print(f"Lendo arquivo: s3://{bucket_nome}/{chave_objeto}")
                    
                    # Tenta ler o conteúdo do arquivo
                    try:
                        resposta = s3_client.get_object(Bucket=bucket_nome, Key=chave_objeto)
                        
                        # O corpo (Body) do objeto é um stream de bytes.
                        # Decodifica para string, assumindo UTF-8, o padrão mais comum.
                        conteudo = resposta['Body'].read().decode('utf-8')
                        conteudos_arquivos.append(conteudo)
                        
                    except ClientError as e:
                        print(f"  --> ERRO ao ler o objeto {chave_objeto}: {e}")
                        # Continua para o próximo arquivo mesmo se um falhar
                        continue
                        
    except ClientError as e:
        # Trata erros ao listar o bucket (e.g., bucket não existe, sem permissão)
        print(f"ERRO ao listar objetos no bucket '{bucket_nome}' com prefixo '{prefixo_arquivos}': {e}")
        return ""
    
    # 4. Unir os conteúdos em uma única string
    # O método str.join() é o mais eficiente para concatenar uma lista de strings.
    string_unida = separador.join(conteudos_arquivos)
    
    return string_unida


def extrair_texto_limpo_pdf(caminho_pdf: str) -> str:
    """
    Lê um PDF e retorna apenas o texto limpo (sem numeração de páginas, headers, footers etc.).
    """
    doc = fitz.open(caminho_pdf)
    texto_total = []

    for pagina in doc:
        texto = pagina.get_text("text")

        # Remove números isolados (como número de página ou índices)
        texto = re.sub(r'\b\d{1,3}\b(?=\s)', '', texto)

        # Remove múltiplos espaços e linhas vazias
        texto = re.sub(r'\n{2,}', '\n', texto)
        texto = re.sub(r'[ \t]+', ' ', texto).strip()

        texto_total.append(texto)

    doc.close()

    # Junta todas as páginas em um único texto contínuo
    texto_unificado = "\n".join(texto_total)
    return texto_unificado


def remove_between_text(documento:str):
    palavra_inicio = r"1\.1\. As Turbulências no Ambiente de Investimentos"
    palavra_fim = r"5\.5\. Conclusões"

    conteudo = documento

    # Regex para capturar tudo entre as duas palavras, incluindo quebras de linha
    padrao = re.compile(f"{palavra_inicio}.*?{palavra_fim}", re.DOTALL)
    resultado = padrao.search(conteudo)

    if resultado:
        trecho = resultado.group(0)
        with open("trecho_extraido.txt", "w", encoding="utf-8") as f_saida:
            f_saida.write(trecho)
        print("Trecho extraído com sucesso!")
    else:
        print("Não foi possível encontrar o trecho entre as palavras.")

def transformar_chunks(path):
    '''
    separar por seções numeradas (ex: 1, 1.1, 2.3.4 etc.)
    retorna uma lista.
    '''

    with open(path,'r') as f: texto = f.read()

    # Regex para separar por seções numeradas (ex: 1, 1.1, 2.3.4 etc.)
    pattern = r'(?=(^\d+(?:\.\d+)*\s+.+$))'

    # Separar com base no padrão e preservar os títulos
    chunks = re.split(pattern, texto, flags=re.MULTILINE)

    # Agrupar os títulos com seus respectivos textos
    secoes = []
    for i in range(1, len(chunks), 2):
        titulo = chunks[i].strip()
        corpo = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
        secoes.append((titulo, corpo))
    
    return secoes

def build_vecstore(arch_path:str ,save_path:str):
    with open(arch_path,'r') as f: texto = f.read()

    # Regex para separar por seções numeradas (ex: 1, 1.1, 2.3.4 etc.)
    pattern = r'(?=(^\d+(?:\.\d+)*\s+.+$))'

    # Separar com base no padrão e preservar os títulos
    chunks = re.split(pattern, texto, flags=re.MULTILINE)

    # Agrupar os títulos com seus respectivos textos
    secoes = []
    for i in range(1, len(chunks), 2):
        titulo = chunks[i].strip()
        corpo = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
        secoes.append((titulo, corpo))

    # 2. Transformar em documentos
    documents = [Document(page_content=secao[1]) for secao in secoes]

    # 3. Criação do embedder
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 4. Gerar embeddings para os documentos
    embeddings = embedding_function.embed_documents([doc.page_content for doc in documents])

    # 5. Criar índice FAISS e adicionar embeddings
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # 6. Criar o docstore e o index_to_docstore_id
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    
    vector_store.save_local(save_path)


