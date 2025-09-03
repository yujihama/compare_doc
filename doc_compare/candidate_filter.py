"""候補絞り込みモジュール（ベクトル類似度 + キーワードマッチング）"""

import numpy as np
from typing import List, Dict, Tuple, Set
import logging

from .keyword_extractor import KeywordExtractor


class CandidateFilter:
    """候補絞り込みクラス"""
    
    def __init__(self, min_similarity: float = 0.5, max_candidates: int = 8, keyword_threshold: float = 0.1):
        """
        Args:
            min_similarity: ベクトル類似度最低ライン
            max_candidates: 最大候補数
            keyword_threshold: TF-IDFスコア閾値
        """
        self.min_similarity = min_similarity
        self.max_candidates = max_candidates
        self.keyword_threshold = keyword_threshold
        self.keyword_extractor = KeywordExtractor(min_tfidf_score=keyword_threshold)
        self.keywords_cache = {}
    
    def calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        2つのベクトル間の類似度を計算
        
        Args:
            vec1: ベクトル1（チャンク全体のベクトル）
            vec2: ベクトル2（チャンク全体のベクトル）
            
        Returns:
            類似度スコア
        """
        if not vec1 or not vec2:
            return 0.0
        
        try:
            # コサイン類似度を計算
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 負の値は0にクリップ
            
        except Exception as e:
            logging.debug(f"ベクトル類似度計算エラー: {e}")
            return 0.0
    
    def filter_by_vector_similarity(self, source_chunks: List[Dict], target_chunks: List[Dict],
                                  source_vectors: List[List[float]], target_vectors: List[List[float]]) -> Dict[str, List[str]]:
        """
        ベクトル類似度による候補絞り込み
        
        Args:
            source_chunks: ソースチャンクリスト
            target_chunks: ターゲットチャンクリスト
            source_vectors: ソースベクトルリスト
            target_vectors: ターゲットベクトルリスト
            
        Returns:
            ソースチャンクID -> ターゲット候補IDリストの辞書
        """
        candidates = {}
        
        for i, source_chunk in enumerate(source_chunks):
            source_id = source_chunk["id"]
            source_vector = source_vectors[i]
            
            # 各ターゲットチャンクとの類似度を計算
            similarities = []
            for j, target_chunk in enumerate(target_chunks):
                target_id = target_chunk["id"]
                target_vector = target_vectors[j]
                
                similarity = self.calculate_vector_similarity(source_vector, target_vector)
                
                if similarity >= self.min_similarity:
                    similarities.append((target_id, similarity))
            
            # 類似度でソートして上位候補を選択
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [target_id for target_id, _ in similarities[:self.max_candidates]]
            
            candidates[source_id] = top_candidates
        
        return candidates
    
    def filter_by_keyword_matching(self, source_chunks: List[Dict], target_chunks: List[Dict],
                                 vector_candidates: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        キーワードマッチングによる追加候補抽出
        
        Args:
            source_chunks: ソースチャンクリスト
            target_chunks: ターゲットチャンクリスト
            vector_candidates: ベクトル類似度による候補辞書
            
        Returns:
            拡張された候補辞書
        """
        # ターゲットチャンクのIDからオブジェクトへのマッピング
        target_chunks_dict = {chunk["id"]: chunk for chunk in target_chunks}
        
        enhanced_candidates = {}
        
        for source_chunk in source_chunks:
            source_id = source_chunk["id"]
            source_keywords = self.keywords_cache.get(source_id, [])
            
            # ベクトル類似度による候補を取得
            vector_candidate_ids = vector_candidates.get(source_id, [])
            
            # キーワードマッチングによる追加候補
            keyword_candidate_ids = []
            
            if source_keywords:
                for target_id, target_chunk in target_chunks_dict.items():
                    # 既にベクトル候補に含まれている場合はスキップ
                    if target_id in vector_candidate_ids:
                        continue
                    
                    # キーワードマッチングチェック
                    if self.keyword_extractor.has_keyword_match(source_keywords, target_chunk["text"]):
                        keyword_candidate_ids.append(target_id)
            
            # ベクトル候補 + キーワード候補を統合
            all_candidates = vector_candidate_ids + keyword_candidate_ids
            
            # 最大候補数で制限
            enhanced_candidates[source_id] = all_candidates[:self.max_candidates]
        
        return enhanced_candidates
    
    def filter_candidates_for_direction(self, source_chunks: List[Dict], target_chunks: List[Dict],
                                      source_vectors: List[List[float]], target_vectors: List[List[float]],
                                      direction: str) -> Dict[str, List[str]]:
        """
        一方向の候補絞り込み
        
        Args:
            source_chunks: ソースチャンクリスト
            target_chunks: ターゲットチャンクリスト
            source_vectors: ソースベクトルリスト
            target_vectors: ターゲットベクトルリスト
            direction: 方向（"old_to_new" または "new_to_old"）
            
        Returns:
            候補辞書
        """
        logging.info(f"Phase {'1' if direction == 'old_to_new' else '2'}: {direction.replace('_', ' → ').title()} 候補絞り込み開始")
        
        # Step 1: ベクトル類似度による絞り込み
        vector_candidates = self.filter_by_vector_similarity(
            source_chunks, target_chunks, source_vectors, target_vectors
        )
        
        # Step 2: キーワードマッチングによる拡張
        enhanced_candidates = self.filter_by_keyword_matching(
            source_chunks, target_chunks, vector_candidates
        )
        
        # 統計情報
        total_candidates = sum(len(candidates) for candidates in enhanced_candidates.values())
        chunks_with_candidates = sum(1 for candidates in enhanced_candidates.values() if candidates)
        
        logging.debug(f"{direction}: {total_candidates}候補, {chunks_with_candidates}/{len(source_chunks)}チャンクに候補あり")
        
        return enhanced_candidates
    
    def filter_all_candidates(self, old_chunks: List[Dict], new_chunks: List[Dict],
                            old_vectors: List[List[float]], new_vectors: List[List[float]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        双方向候補絞り込み
        
        Args:
            old_chunks: 旧チャンクリスト
            new_chunks: 新チャンクリスト
            old_vectors: 旧ベクトルリスト
            new_vectors: 新ベクトルリスト
            
        Returns:
            (old_to_new_candidates, new_to_old_candidates)
        """
        # 全チャンクの特徴語を事前抽出
        all_chunks = old_chunks + new_chunks
        self.keywords_cache = self.keyword_extractor.extract_keywords_from_chunks(all_chunks)
        
        # Phase 1: Old → New 候補絞り込み
        old_to_new_candidates = self.filter_candidates_for_direction(
            old_chunks, new_chunks, old_vectors, new_vectors, "old_to_new"
        )
        
        # Phase 2: New → Old 候補絞り込み
        new_to_old_candidates = self.filter_candidates_for_direction(
            new_chunks, old_chunks, new_vectors, old_vectors, "new_to_old"
        )
        
        # 統計情報
        total_old_candidates = sum(len(candidates) for candidates in old_to_new_candidates.values())
        total_new_candidates = sum(len(candidates) for candidates in new_to_old_candidates.values())
        
        old_with_candidates = sum(1 for candidates in old_to_new_candidates.values() if candidates)
        new_with_candidates = sum(1 for candidates in new_to_old_candidates.values() if candidates)
        
        logging.info(f"候補絞り込み完了: Old {old_with_candidates}/{len(old_chunks)}, New {new_with_candidates}/{len(new_chunks)}")
        
        return old_to_new_candidates, new_to_old_candidates
    
    def get_statistics(self) -> Dict:
        """
        候補絞り込みの統計情報を取得
        
        Returns:
            統計情報辞書
        """
        keyword_stats = self.keyword_extractor.get_statistics()
        
        return {
            "min_similarity": self.min_similarity,
            "max_candidates": self.max_candidates,
            "keyword_threshold": self.keyword_threshold,
            "keyword_extraction": keyword_stats
        }


def filter_candidates(old_chunks: List[Dict], new_chunks: List[Dict],
                     old_vectors: List[List[List]], new_vectors: List[List[List]],
                     min_similarity: float = 0.5, max_candidates: int = 8,
                     keyword_threshold: float = 0.1) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    便利関数：候補絞り込み
    
    Args:
        old_chunks: 旧チャンクリスト
        new_chunks: 新チャンクリスト
        old_vectors: 旧ベクトルリスト
        new_vectors: 新ベクトルリスト
        min_similarity: ベクトル類似度最低ライン
        max_candidates: 最大候補数
        keyword_threshold: TF-IDFスコア閾値
        
    Returns:
        (old_to_new_candidates, new_to_old_candidates)
    """
    filter_obj = CandidateFilter(
        min_similarity=min_similarity,
        max_candidates=max_candidates,
        keyword_threshold=keyword_threshold
    )
    
    return filter_obj.filter_all_candidates(old_chunks, new_chunks, old_vectors, new_vectors) 