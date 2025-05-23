import numpy as np
import json
import hashlib
import os
import logging
from typing import List, Dict, Optional, Tuple

def calculate_text_hash(chunks: List[Dict]) -> str:
    """チャンクテキストのハッシュ値を計算"""
    combined_text = "".join([chunk["text"] for chunk in chunks])
    return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

def save_embeddings_cache(chunks: List[Dict], sent_lists: List[List[str]], 
                         embeddings: List[List], prefix: str) -> bool:
    """エンベディングをファイルにキャッシュ保存"""
    try:
        # ハッシュ値計算
        text_hash = calculate_text_hash(chunks)
        
        # メタデータ作成
        metadata = {
            "prefix": prefix,
            "text_hash": text_hash,
            "total_chunks": len(chunks),
            "total_sentences": sum(len(sents) for sents in sent_lists),
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
            np.savez_compressed(f"{prefix}_embeddings.npz", 
                              embeddings=embeddings_array, 
                              indices=np.array(indices))
        
        # メタデータをJSON保存
        with open(f"{prefix}_cache_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 文分割結果も保存
        with open(f"{prefix}_sentences.json", "w", encoding="utf-8") as f:
            json.dump(sent_lists, f, ensure_ascii=False, indent=2)
        
        logging.info(f"{prefix}のエンベディングキャッシュを保存: {len(embeddings_array)}ベクトル")
        return True
        
    except Exception as e:
        logging.error(f"エンベディングキャッシュ保存エラー: {e}")
        return False

def load_embeddings_cache(chunks: List[Dict], prefix: str) -> Optional[Tuple[List[List[str]], List[List]]]:
    """エンベディングキャッシュを読み込み"""
    try:
        # ファイル存在チェック
        metadata_file = f"{prefix}_cache_metadata.json"
        embeddings_file = f"{prefix}_embeddings.npz"
        sentences_file = f"{prefix}_sentences.json"
        
        if not all(os.path.exists(f) for f in [metadata_file, embeddings_file, sentences_file]):
            logging.info(f"{prefix}のキャッシュファイルが見つかりません")
            return None
        
        # メタデータ読み込み
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # ハッシュ値チェック（テキストが変更されていないか確認）
        current_hash = calculate_text_hash(chunks)
        if metadata.get("text_hash") != current_hash:
            logging.info(f"{prefix}のテキストが変更されています。キャッシュを無効化")
            return None
        
        # チャンク数チェック
        if metadata.get("total_chunks") != len(chunks):
            logging.info(f"{prefix}のチャンク数が変更されています。キャッシュを無効化")
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
            embeddings[chunk_idx].append(embeddings_array[i].tolist())
        
        logging.info(f"{prefix}のエンベディングキャッシュを読み込み: {len(embeddings_array)}ベクトル")
        return sent_lists, embeddings
        
    except Exception as e:
        logging.error(f"エンベディングキャッシュ読み込みエラー: {e}")
        return None

def clear_cache(prefix: str) -> bool:
    """指定プレフィックスのキャッシュファイルを削除"""
    try:
        files_to_remove = [
            f"{prefix}_cache_metadata.json",
            f"{prefix}_embeddings.npz", 
            f"{prefix}_sentences.json",
            f"{prefix}_embeddings_info.json"  # 旧形式も削除
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
        metadata_file = f"{prefix}_cache_metadata.json"
        if not os.path.exists(metadata_file):
            return {"exists": False}
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # ファイルサイズ取得
        embeddings_file = f"{prefix}_embeddings.npz"
        file_size = os.path.getsize(embeddings_file) if os.path.exists(embeddings_file) else 0
        
        return {
            "exists": True,
            "total_chunks": metadata.get("total_chunks", 0),
            "total_sentences": metadata.get("total_sentences", 0),
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "text_hash": metadata.get("text_hash", "unknown")[:8]  # 最初の8文字のみ表示
        }
        
    except Exception as e:
        logging.error(f"キャッシュ情報取得エラー: {e}")
        return {"exists": False, "error": str(e)} 