"""TF-IDF特徴語抽出モジュール"""

import re
from typing import List, Dict, Set
from collections import defaultdict
import logging

try:
    from janome.tokenizer import Tokenizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"TF-IDF依存関係が不足: {e}")
    DEPENDENCIES_AVAILABLE = False


class KeywordExtractor:
    """TF-IDF特徴語抽出クラス"""
    
    def __init__(self, min_tfidf_score: float = 0.1):
        """
        Args:
            min_tfidf_score: 特徴語として採用するTF-IDFスコアの最低値
        """
        self.min_tfidf_score = min_tfidf_score
        self.tokenizer = None
        self.vectorizer = None
        self.keywords_cache = {}
        
        if DEPENDENCIES_AVAILABLE:
            self.tokenizer = Tokenizer()
        else:
            logging.error("janomeまたはscikit-learnがインストールされていません")
    
    def extract_nouns(self, text: str) -> List[str]:
        """
        テキストから名詞のみを抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            名詞リスト
        """
        if not self.tokenizer:
            return []
        
        # テキストのクリーニング
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        nouns = []
        try:
            for token in self.tokenizer.tokenize(cleaned_text):
                features = token.part_of_speech.split(',')
                # 名詞のみを抽出（一般名詞、固有名詞、代名詞、数詞）
                if features[0] == '名詞' and len(token.surface) > 1:
                    nouns.append(token.surface)
        except Exception as e:
            logging.debug(f"形態素解析エラー: {e}")
        
        return nouns
    
    def extract_keywords_from_chunks(self, chunks: List[Dict]) -> Dict[str, List[str]]:
        """
        全チャンクからTF-IDF特徴語を抽出
        
        Args:
            chunks: チャンクリスト
            
        Returns:
            チャンクID -> 特徴語リストの辞書
        """
        if not DEPENDENCIES_AVAILABLE:
            logging.error("TF-IDF抽出に必要な依存関係が不足しています")
            return {}
        
        logging.info(f"全チャンクの特徴語抽出を開始")
        
        # 各チャンクから名詞を抽出
        chunk_nouns = {}
        documents = []
        chunk_ids = []
        
        for chunk in chunks:
            chunk_id = chunk["id"]
            text = chunk["text"]
            
            # 名詞抽出
            nouns = self.extract_nouns(text)
            chunk_nouns[chunk_id] = nouns
            
            # TF-IDF用のドキュメント作成
            document = " ".join(nouns)
            documents.append(document)
            chunk_ids.append(chunk_id)
        
        if not documents:
            logging.warning("抽出可能な名詞が見つかりませんでした")
            return {}
        
        # TF-IDF計算
        try:
            self.vectorizer = TfidfVectorizer(
                min_df=1,  # 最低1回は出現
                max_df=0.8,  # 80%以上のドキュメントに出現する語は除外
                ngram_range=(1, 1),  # 単語のみ
                token_pattern=r'\S+'  # 空白以外の文字列
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            feature_names = self.vectorizer.get_feature_names_out()
            
        except Exception as e:
            logging.error(f"TF-IDF計算エラー: {e}")
            return {}
        
        # 各チャンクの特徴語を抽出
        keywords_dict = {}
        chunks_with_keywords = 0
        
        for i, chunk_id in enumerate(chunk_ids):
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            
            # スコアが閾値以上の特徴語を抽出
            keywords = []
            for j, score in enumerate(tfidf_scores):
                if score >= self.min_tfidf_score:
                    keyword = feature_names[j]
                    keywords.append(keyword)
            
            keywords_dict[chunk_id] = keywords
            if keywords:
                chunks_with_keywords += 1
            
            # キャッシュに保存
            self.keywords_cache[chunk_id] = keywords
        
        logging.info(f"特徴語抽出完了: {len(chunks)}チャンク中{chunks_with_keywords}チャンクで特徴語を検出")
        
        return keywords_dict
    
    def get_keywords_for_chunk(self, chunk_id: str) -> List[str]:
        """
        特定チャンクの特徴語を取得（キャッシュから）
        
        Args:
            chunk_id: チャンクID
            
        Returns:
            特徴語リスト
        """
        return self.keywords_cache.get(chunk_id, [])
    
    def has_keyword_match(self, chunk_keywords: List[str], target_text: str) -> bool:
        """
        特徴語がターゲットテキストに含まれているかチェック
        
        Args:
            chunk_keywords: チャンクの特徴語リスト
            target_text: 検索対象テキスト
            
        Returns:
            マッチするかどうか
        """
        if not chunk_keywords:
            return False
        
        # テキストを正規化
        normalized_text = target_text.lower()
        
        # 特徴語のいずれかが含まれているかチェック
        for keyword in chunk_keywords:
            if keyword.lower() in normalized_text:
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """
        特徴語抽出の統計情報を取得
        
        Returns:
            統計情報辞書
        """
        total_chunks = len(self.keywords_cache)
        chunks_with_keywords = sum(1 for keywords in self.keywords_cache.values() if keywords)
        total_keywords = sum(len(keywords) for keywords in self.keywords_cache.values())
        
        avg_keywords_per_chunk = total_keywords / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "chunks_with_keywords": chunks_with_keywords,
            "total_keywords": total_keywords,
            "avg_keywords_per_chunk": avg_keywords_per_chunk,
            "min_tfidf_score": self.min_tfidf_score
        }


def extract_keywords_from_chunks(chunks: List[Dict], min_tfidf_score: float = 0.1) -> Dict[str, List[str]]:
    """
    便利関数：チャンクリストから特徴語を抽出
    
    Args:
        chunks: チャンクリスト
        min_tfidf_score: TF-IDFスコア閾値
        
    Returns:
        チャンクID -> 特徴語リストの辞書
    """
    extractor = KeywordExtractor(min_tfidf_score=min_tfidf_score)
    return extractor.extract_keywords_from_chunks(chunks) 