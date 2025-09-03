"""テキスト処理とベクトル化の統合モジュール"""

import re
import os
from typing import List, Dict
from dotenv import load_dotenv
from .config import OPENAI_CONFIG, get_embeddings_client

load_dotenv()

def chunk_document(text: str, prefix: str) -> List[Dict]:
    """
    空行でチャンク分割し、各チャンクにIDを付与。
    """
    chunks = re.split(r'\n\s*\n', text.strip())
    result = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"{prefix}_chunk_{i}"
        result.append({
            "id": chunk_id,
            "text": chunk.strip()
        })
    return result

def split_sentences(chunk_text: str) -> List[str]:
    """
    「。」で文分割（末尾に「。」を残す）。
    """
    sentences = re.findall(r'.+?。', chunk_text, flags=re.DOTALL)
    # 残りの文（「。」で終わらない）も追加
    rest = re.sub(r'.+?。', '', chunk_text, flags=re.DOTALL)
    if rest.strip():
        sentences.append(rest.strip())
    return [s.strip() for s in sentences if s.strip()]

def get_embeddings(sentences: List[str], model: str = None) -> List[list]:
    """
    文章リストのエンベディングを取得
    """
    if model is None:
        model = OPENAI_CONFIG["embedding_model"]
    
    # langchain_openai Embeddings クライアント経由で取得 (.env で openai/azure 切替)
    emb = get_embeddings_client(model=model)
    vectors = emb.embed_documents(sentences)
    return vectors

def process_chunks_to_embeddings(chunks: List[Dict]) -> tuple[List[List[str]], List[List]]:
    """
    チャンクリストから文分割とエンベディング生成を一括処理
    
    Returns:
        tuple: (文分割結果のリスト, エンベディングのリスト)
    """
    all_sentences = []
    sentence_lists = []
    chunk_sentence_counts = []
    
    # 全チャンクから文を抽出
    for chunk in chunks:
        sentences = split_sentences(chunk["text"])
        sentence_lists.append(sentences)
        chunk_sentence_counts.append(len(sentences))
        all_sentences.extend(sentences)
    
    # 一括でエンベディング取得
    if all_sentences:
        all_embeddings = get_embeddings(all_sentences)
        
        # チャンク別にエンベディングを分割
        embeddings_by_chunk = []
        start_idx = 0
        for count in chunk_sentence_counts:
            end_idx = start_idx + count
            chunk_embeddings = all_embeddings[start_idx:end_idx]
            embeddings_by_chunk.append(chunk_embeddings)
            start_idx = end_idx
    else:
        embeddings_by_chunk = [[] for _ in chunks]
    
    return sentence_lists, embeddings_by_chunk

def process_chunks_to_chunk_embeddings(chunks: List[Dict]) -> List[List[float]]:
    """
    チャンクリストから各チャンク全体のエンベディングを生成
    
    Args:
        chunks: チャンクリスト
        
    Returns:
        List[List[float]]: 各チャンクのエンベディングリスト
    """
    if not chunks:
        return []
    
    # 各チャンクのテキスト全体を取得
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    # 一括でエンベディング取得
    chunk_embeddings = get_embeddings(chunk_texts)
    
    return chunk_embeddings

def clean_text(text: str) -> str:
    """
    テキストの基本的なクリーニング
    """
    # 行番号プレフィックスを削除
    text = re.sub(r'^row_\d+:\s*', '', text, flags=re.MULTILINE)
    
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    return text

def extract_heading_from_text(text: str, max_length: int = 50) -> str:
    """
    テキストから見出しを抽出
    """
    # 改行で分割して最初の意味のある行を取得
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        clean_line = clean_text(line)
        if clean_line and len(clean_line) > 5:
            # 長すぎる場合は切り詰め
            if len(clean_line) > max_length:
                return clean_line[:max_length] + "..."
            return clean_line
    
    return "見出しなし" 