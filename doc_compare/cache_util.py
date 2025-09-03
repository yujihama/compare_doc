import numpy as np
import json
import hashlib
import os
import logging
from typing import List, Dict, Optional, Tuple
from .config import CACHE_CONFIG
from datetime import datetime, timedelta

def calculate_text_hash(chunks: List[Dict]) -> str:
    """チャンクテキストのハッシュ値を計算"""
    combined_text = "".join([chunk["text"] for chunk in chunks])
    return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

def calculate_pdf_hash(pdf_content: bytes) -> str:
    """PDFファイルの内容からハッシュ値を計算"""
    return hashlib.md5(pdf_content).hexdigest()

def save_embeddings_cache(chunks: List[Dict], sent_lists: List[List[str]], 
                         embeddings: List[List], prefix: str) -> bool:
    """エンベディングをファイルにキャッシュ保存"""
    # キャッシュが無効化されている場合はスキップ
    if not CACHE_CONFIG["enable_cache"]:
        logging.info("キャッシュが無効化されているため、保存をスキップします")
        return False
    
    try:
        # キャッシュディレクトリの作成
        cache_dir = CACHE_CONFIG["cache_directory"]
        if cache_dir != "." and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # ファイルパスの構築
        def get_cache_path(filename):
            return os.path.join(cache_dir, filename)
        
        # ハッシュ値計算
        text_hash = calculate_text_hash(chunks)
        
        # メタデータ作成
        metadata = {
            "prefix": prefix,
            "text_hash": text_hash,
            "total_chunks": len(chunks),
            "total_sentences": sum(len(sents) for sents in sent_lists),
            "created_at": datetime.now().isoformat(),
            "chunks_info": [
                {
                    "chunk_id": chunk["id"],
                    "sentence_count": len(sent_lists[i]),
                    "embedding_count": len(embeddings[i]) if embeddings[i] else 0
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        # エンベディングベクトルを保存（numpy形式）
        embeddings_array = []
        indices = []  # どのチャンクのどの文かのインデックス
        
        for chunk_idx, chunk_embeddings in enumerate(embeddings):
            for sent_idx, embedding in enumerate(chunk_embeddings):
                embeddings_array.append(embedding)
                indices.append((chunk_idx, sent_idx))
        
        if embeddings_array:
            embeddings_array = np.array(embeddings_array)
            np.savez_compressed(get_cache_path(f"{prefix}_embeddings.npz"), 
                              embeddings=embeddings_array, 
                              indices=np.array(indices))
        
        # メタデータをJSON保存
        with open(get_cache_path(f"{prefix}_cache_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 文分割結果も保存
        with open(get_cache_path(f"{prefix}_sentences.json"), "w", encoding="utf-8") as f:
            json.dump(sent_lists, f, ensure_ascii=False, indent=2)
        
        logging.info(f"{prefix}のエンベディングキャッシュを保存: {len(embeddings_array)}ベクトル")
        return True
        
    except Exception as e:
        logging.error(f"エンベディングキャッシュ保存エラー: {e}")
        return False

def load_embeddings_cache(chunks: List[Dict], prefix: str) -> Optional[Tuple[List[List[str]], List[List]]]:
    """エンベディングキャッシュを読み込み"""
    # キャッシュが無効化されている場合はスキップ
    if not CACHE_CONFIG["enable_cache"]:
        return None
    
    try:
        # キャッシュディレクトリの設定
        cache_dir = CACHE_CONFIG["cache_directory"]
        
        # ファイルパスの構築
        def get_cache_path(filename):
            return os.path.join(cache_dir, filename)
        
        # ファイル存在チェック
        metadata_file = get_cache_path(f"{prefix}_cache_metadata.json")
        embeddings_file = get_cache_path(f"{prefix}_embeddings.npz")
        sentences_file = get_cache_path(f"{prefix}_sentences.json")
        
        if not all(os.path.exists(f) for f in [metadata_file, embeddings_file, sentences_file]):
            return None
        
        # メタデータ読み込み
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # 有効期限チェック
        if "created_at" in metadata:
            created_at = datetime.fromisoformat(metadata["created_at"])
        else:
            # 既存キャッシュとの互換性のため、ファイルの更新日時を使用
            created_at = datetime.fromtimestamp(os.path.getmtime(metadata_file))
        
        max_age = timedelta(days=CACHE_CONFIG["max_cache_age_days"])
        
        if datetime.now() - created_at > max_age:
            return None
        
        # テキストハッシュ比較
        current_hash = calculate_text_hash(chunks)
        if current_hash != metadata["text_hash"]:
            return None
        
        # 文分割結果読み込み
        with open(sentences_file, "r", encoding="utf-8") as f:
            sent_lists = json.load(f)
        
        # エンベディング読み込み
        cache_data = np.load(embeddings_file)
        embeddings_array = cache_data["embeddings"]
        indices = cache_data["indices"]
        
        # エンベディングを元の構造に復元
        embeddings = [[] for _ in chunks]
        for i, (chunk_idx, sent_idx) in enumerate(indices):
            if chunk_idx < len(embeddings):
                embeddings[chunk_idx].append(embeddings_array[i].tolist())
        
        return sent_lists, embeddings
        
    except Exception as e:
        logging.warning(f"キャッシュ読み込みエラー: {e}")
        return None

def clear_cache(prefix: str) -> bool:
    """指定プレフィックスのキャッシュファイルを削除"""
    try:
        # キャッシュディレクトリの設定
        cache_dir = CACHE_CONFIG["cache_directory"]
        
        # ファイルパスの構築
        def get_cache_path(filename):
            return os.path.join(cache_dir, filename)
        
        files_to_remove = [
            get_cache_path(f"{prefix}_cache_metadata.json"),
            get_cache_path(f"{prefix}_embeddings.npz"), 
            get_cache_path(f"{prefix}_sentences.json"),
            get_cache_path(f"{prefix}_embeddings_info.json")  # 旧形式も削除
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        if removed_count > 0:
            logging.info(f"{prefix}のキャッシュファイル{removed_count}個を削除")
            return True
        else:
            logging.info(f"{prefix}のキャッシュファイルは見つかりませんでした")
            return False
            
    except Exception as e:
        logging.error(f"キャッシュクリアエラー: {e}")
        return False

def get_cache_info(prefix: str) -> Dict:
    """キャッシュ情報を取得"""
    try:
        # キャッシュディレクトリの設定
        cache_dir = CACHE_CONFIG["cache_directory"]
        
        # ファイルパスの構築
        def get_cache_path(filename):
            return os.path.join(cache_dir, filename)
        
        metadata_file = get_cache_path(f"{prefix}_cache_metadata.json")
        
        if not os.path.exists(metadata_file):
            return {"exists": False}
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # 有効期限チェック
        if "created_at" in metadata:
            created_at = datetime.fromisoformat(metadata["created_at"])
        else:
            # 既存キャッシュとの互換性のため、ファイルの更新日時を使用
            created_at = datetime.fromtimestamp(os.path.getmtime(metadata_file))
        
        max_age = timedelta(days=CACHE_CONFIG["max_cache_age_days"])
        
        if datetime.now() - created_at > max_age:
            return {"exists": False}
        
        # ファイルサイズ計算
        total_size = 0
        for filename in [f"{prefix}_cache_metadata.json", f"{prefix}_embeddings.npz", f"{prefix}_sentences.json"]:
            file_path = get_cache_path(filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        return {
            "exists": True,
            "total_chunks": metadata["total_chunks"],
            "total_sentences": metadata["total_sentences"],
            "file_size_mb": round(total_size / (1024 * 1024), 2),
            "text_hash": metadata["text_hash"][:8]  # 最初の8文字のみ
        }
        
    except Exception as e:
        logging.warning(f"キャッシュ情報取得エラー: {e}")
        return {"exists": False} 