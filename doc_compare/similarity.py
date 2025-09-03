import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from .config import SIMILARITY_THRESHOLDS, CLUSTERING_CONFIG
from .text_processing import get_embeddings
import re

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def extract_hierarchy_from_chunk(chunk):
    """チャンクから階層情報を抽出"""
    hierarchy_levels = {}
    text = chunk.get('text', '')
    
    # 上位階層1, 2, 3を抽出
    for level in range(1, 4):
        pattern = rf'\[上位階層{level}\]\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text)
        if match:
            hierarchy_levels[f'level_{level}'] = match.group(1).strip()
    
    return hierarchy_levels

def normalize_hierarchy_title(chunk):
    """上位階層のタイトルを結合・正規化"""
    hierarchy = extract_hierarchy_from_chunk(chunk)
    hierarchy_parts = []
    
    # レベル1, 2, 3を順次結合
    for level in ['level_1', 'level_2', 'level_3']:
        if level in hierarchy and hierarchy[level]:
            hierarchy_parts.append(hierarchy[level])
    
    # 結合してスペース除去
    combined_title = ''.join(hierarchy_parts)
    
    # 基本的な正規化
    normalized = combined_title.replace(' ', '').replace('　', '')  # 全角・半角スペース除去
    normalized = normalized.replace('\n', '').replace('\t', '')     # 改行・タブ除去
    
    return normalized

def extract_hierarchy_info(normalized_title):
    """正規化されたタイトルから構造情報を抽出"""
    return {
        'full_title': normalized_title,
        'content_without_numbers': remove_all_numbering(normalized_title)
    }

def remove_all_numbering(text):
    """すべての項番パターンを除去"""
    # 各種項番パターンを除去
    patterns = [
        r'第\d+編',      # 第2編
        r'第\d+章',      # 第3章  
        r'第\d+節',      # 第1節
        r'第\d+条',      # 第15条
        r'\d+\.',        # 1. 2. 3.
        r'\(\d+\)',      # (1) (2) (3)
        r'[A-Z]\.',      # A. B. C.
        r'[ア-ン]\.',    # ア. イ. ウ.
        r'[①-⑳]',       # 丸数字
    ]
    
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    
    return result.strip()

def calculate_hierarchy_vector_similarity(text1, text2):
    """階層テキストのベクトル類似度を計算（既存のget_embeddings関数を使用）"""
    
    if not text1 or not text2:
        return 0.0
    
    try:
        # 既存のget_embeddings関数を使用してベクトル取得
        embeddings = get_embeddings([text1, text2])
        
        if len(embeddings) == 2 and embeddings[0] and embeddings[1]:
            return cosine_similarity(embeddings[0], embeddings[1])
        else:
            return 0.0
            
    except Exception as e:
        logging.warning(f"階層ベクトル類似度計算エラー: {e}")
        return 0.0

def calculate_hierarchy_similarity_staged(old_chunk, new_chunk):
    """段階的階層類似度計算"""
    
    old_hierarchy = normalize_hierarchy_title(old_chunk)
    new_hierarchy = normalize_hierarchy_title(new_chunk)
    
    old_info = extract_hierarchy_info(old_hierarchy)
    new_info = extract_hierarchy_info(new_hierarchy)
    
    # Stage 1: 完全一致
    if old_info['full_title'] == new_info['full_title']:
        return 1.0, 'exact_match'
    
    # Stage 2: 項番変更のみ（内容は同じ）
    if (old_info['content_without_numbers'] and 
        new_info['content_without_numbers'] and
        old_info['content_without_numbers'] == new_info['content_without_numbers']):
        return 0.9, 'numbering_change'
    
    # Stage 3: 名称変更（ベクトル類似度で判定）
    if old_info['content_without_numbers'] and new_info['content_without_numbers']:
        vector_similarity = calculate_hierarchy_vector_similarity(
            old_info['content_without_numbers'], 
            new_info['content_without_numbers']
        )
        
        if vector_similarity >= 0.8:
            return vector_similarity * 0.8, 'semantic_change'  # 最大0.64に制限
        elif vector_similarity >= 0.6:
            return vector_similarity * 0.6, 'weak_semantic_change'  # 最大0.36に制限
    
    # Stage 4: 類似度なし
    return 0.0, 'no_match'

def calculate_integrated_hierarchy_score(old_chunk, new_chunk):
    """統合された階層マッチングスコア"""
    
    # 階層類似度を段階的に計算
    hierarchy_score, match_type = calculate_hierarchy_similarity_staged(old_chunk, new_chunk)
    
    # マッチタイプに応じた重み調整
    type_weights = {
        'exact_match': 1.0,           # 完全一致
        'numbering_change': 0.9,      # 項番変更のみ
        'semantic_change': 0.8,       # 意味的変更
        'weak_semantic_change': 0.6,  # 弱い意味的変更
        'no_match': 0.0              # マッチなし
    }
    
    final_score = hierarchy_score * type_weights.get(match_type, 0.0)
    
    return {
        'score': final_score,
        'match_type': match_type,
        'raw_hierarchy_score': hierarchy_score
    }

def apply_hierarchy_constraints_to_matching(old_chunks, new_chunks):
    """階層制約をマッチングに適用"""
    
    hierarchy_matrix = {}
    
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            hierarchy_result = calculate_integrated_hierarchy_score(old_chunk, new_chunk)
            hierarchy_matrix[(i, j)] = hierarchy_result
    
    return hierarchy_matrix

def adjust_similarity_with_hierarchy(content_similarity, hierarchy_result, weights=None):
    """内容類似度を階層制約で調整"""
    
    if weights is None:
        weights = {
            'content': 0.6,      # 内容類似度の重み
            'hierarchy': 0.4     # 階層類似度の重み
        }
    
    hierarchy_score = hierarchy_result['score']
    match_type = hierarchy_result['match_type']
    
    # 階層マッチタイプに応じた調整
    if match_type == 'exact_match':
        # 完全一致の場合は階層重みを上げる
        adjusted_weights = {'content': 0.5, 'hierarchy': 0.5}
    elif match_type == 'numbering_change':
        # 項番変更の場合は階層を重視
        adjusted_weights = {'content': 0.6, 'hierarchy': 0.4}
    elif match_type in ['semantic_change', 'weak_semantic_change']:
        # 意味的変更の場合は内容を重視
        adjusted_weights = {'content': 0.7, 'hierarchy': 0.3}
    else:
        # マッチなしの場合は内容のみ
        adjusted_weights = {'content': 1.0, 'hierarchy': 0.0}
    
    final_similarity = (
        adjusted_weights['content'] * content_similarity +
        adjusted_weights['hierarchy'] * hierarchy_score
    )
    
    return final_similarity, match_type

def enhanced_similarity_calculation(old_chunks, new_chunks, old_vectors, new_vectors):
    """階層制約を組み込んだ類似度計算"""
    
    logging.debug("階層制約付き類似度計算開始")
    
    # 1. 全チャンクの階層タイトルを抽出
    old_hierarchy_titles = []
    new_hierarchy_titles = []
    old_hierarchy_map = {}  # chunk_id -> hierarchy_title
    new_hierarchy_map = {}  # chunk_id -> hierarchy_title
    
    for i, chunk in enumerate(old_chunks):
        hierarchy_title = normalize_hierarchy_title(chunk)
        old_hierarchy_titles.append(hierarchy_title)
        old_hierarchy_map[chunk['id']] = hierarchy_title
    
    for i, chunk in enumerate(new_chunks):
        hierarchy_title = normalize_hierarchy_title(chunk)
        new_hierarchy_titles.append(hierarchy_title)
        new_hierarchy_map[chunk['id']] = hierarchy_title
    
    # 2. 項番除去後の内容のみを抽出（エンベディング対象）
    all_content_without_numbers = set()
    
    for title in old_hierarchy_titles + new_hierarchy_titles:
        if title.strip():
            info = extract_hierarchy_info(title)
            content = info['content_without_numbers'].strip()
            if content:  # 空でない内容のみ
                all_content_without_numbers.add(content)
    
    unique_contents = sorted(list(all_content_without_numbers))  # ソートして順序を決定的にする
    
    logging.debug(f"階層タイトル数: 旧{len(old_hierarchy_titles)}個, 新{len(new_hierarchy_titles)}個")
    logging.debug(f"項番除去後のユニーク内容: {len(unique_contents)}個")
    
    # 3. ユニークな内容のエンベディングを一括取得
    hierarchy_embeddings_cache = {}
    if unique_contents:
        try:
            logging.debug(f"階層内容のエンベディングを一括取得中: {len(unique_contents)}個")
            embeddings = get_embeddings(unique_contents)
            
            for content, embedding in zip(unique_contents, embeddings):
                hierarchy_embeddings_cache[content] = embedding
                
            logging.debug(f"階層エンベディング一括取得完了: {len(hierarchy_embeddings_cache)}個")
        except Exception as e:
            logging.warning(f"階層エンベディング一括取得エラー: {e}")
            hierarchy_embeddings_cache = {}
    
    # 4. 階層制約マトリックスを計算（キャッシュ使用）
    hierarchy_matrix = {}
    
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            old_hierarchy = old_hierarchy_map[old_chunk['id']]
            new_hierarchy = new_hierarchy_map[new_chunk['id']]
            
            # キャッシュされたエンベディングを使用して階層類似度を計算
            hierarchy_result = calculate_hierarchy_similarity_with_cache(
                old_chunk, new_chunk, old_hierarchy, new_hierarchy, hierarchy_embeddings_cache
            )
            hierarchy_matrix[(i, j)] = hierarchy_result
    
    # 5. 統合類似度を計算
    enhanced_similarities = {}
    
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            # 従来の内容類似度
            content_similarity = calculate_chunk_similarity(old_vectors[i], new_vectors[j])
            
            # 階層制約結果
            hierarchy_result = hierarchy_matrix[(i, j)]
            
            # 統合類似度
            final_similarity, match_type = adjust_similarity_with_hierarchy(
                content_similarity, hierarchy_result
            )
            
            enhanced_similarities[(old_chunk['id'], new_chunk['id'])] = {
                'similarity': final_similarity,
                'content_similarity': content_similarity,  # バイパス判定用
                'hierarchy_score': hierarchy_result['score'],
                'match_type': match_type
            }
                
    logging.debug(f"階層制約付き類似度計算完了: {len(enhanced_similarities)}ペア")
    return enhanced_similarities

def calculate_hierarchy_similarity_with_cache(old_chunk, new_chunk, old_hierarchy, new_hierarchy, embeddings_cache):
    """キャッシュを使用した階層類似度計算"""
    
    # 階層タイトルが空の場合は類似度なし
    if not old_hierarchy.strip() or not new_hierarchy.strip():
        return {
            'score': 0.0,
            'match_type': 'no_hierarchy',
            'raw_hierarchy_score': 0.0
        }
    
    old_info = extract_hierarchy_info(old_hierarchy)
    new_info = extract_hierarchy_info(new_hierarchy)
    
    # Stage 1: 完全一致
    if old_info['full_title'] == new_info['full_title']:
        return {
            'score': 1.0,
            'match_type': 'exact_match',
            'raw_hierarchy_score': 1.0
        }
    
    # Stage 2: 項番変更のみ（内容は同じ）
    old_content = old_info['content_without_numbers'].strip()
    new_content = new_info['content_without_numbers'].strip()
    
    if (old_content and new_content and old_content == new_content):
        return {
            'score': 0.9,
            'match_type': 'numbering_change',
            'raw_hierarchy_score': 0.9
        }
    
    # Stage 3: 名称変更（キャッシュされたベクトル類似度で判定）
    if (old_content and new_content and
        old_content in embeddings_cache and
        new_content in embeddings_cache):
        
        old_embedding = embeddings_cache[old_content]
        new_embedding = embeddings_cache[new_content]
        
        vector_similarity = cosine_similarity(old_embedding, new_embedding)
        
        if vector_similarity >= 0.8:
            final_score = vector_similarity * 0.8  # 最大0.64に制限
            return {
                'score': final_score,
                'match_type': 'semantic_change',
                'raw_hierarchy_score': vector_similarity
            }
        elif vector_similarity >= 0.6:
            final_score = vector_similarity * 0.6  # 最大0.36に制限
            return {
                'score': final_score,
                'match_type': 'weak_semantic_change',
                'raw_hierarchy_score': vector_similarity
            }
    
    # Stage 4: 類似度なし
    return {
        'score': 0.0,
        'match_type': 'no_match',
        'raw_hierarchy_score': 0.0
    }

def calculate_chunk_similarity(old_vector, new_vector):
    """チャンク間の類似度を計算（チャンク全体のベクトル同士を比較）"""
    if old_vector is None or new_vector is None:
        return 0.0
    
    try:
        # チャンク全体のベクトル同士でコサイン類似度を計算
        return cosine_similarity(old_vector, new_vector)
        
    except Exception as e:
        logging.warning(f"チャンク類似度計算エラー: {e}")
        return 0.0

def find_strict_clusters(old_to_new, new_to_old, similarities, threshold):
    """
    厳格なクラスター形成：より高い閾値と密度要件でクラスター形成
    """
    logging.info(f"厳格クラスター形成: 高閾値={threshold + 0.1:.2f}")
    
    # より高い閾値で強い関係のみを抽出
    strict_threshold = min(threshold + 0.1, SIMILARITY_THRESHOLDS["strict"])
    strong_old_to_new = {}
    strong_new_to_old = {}
    strong_similarities = {}
    
    for old_id, new_ids in old_to_new.items():
        strong_old_to_new[old_id] = set()
        for new_id in new_ids:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim >= strict_threshold:
                strong_old_to_new[old_id].add(new_id)
                if new_id not in strong_new_to_old:
                    strong_new_to_old[new_id] = set()
                strong_new_to_old[new_id].add(old_id)
                strong_similarities[(old_id, new_id)] = sim
    
    # 高閾値での連結成分検出
    strict_clusters = find_connected_clusters(strong_old_to_new, strong_new_to_old, strong_similarities)
    
    logging.info(f"厳格クラスター形成完了: {len(strict_clusters)}個")
    return strict_clusters

def find_adaptive_clusters(old_to_new, new_to_old, similarities, threshold):
    """
    適応的クラスター形成：類似度に応じて段階的にクラスター形成
    """
    logging.info("適応的クラスター形成開始")
    
    # 類似度の分布を分析
    all_similarities = list(similarities.values())
    if not all_similarities:
        return []
    
    percentiles = np.percentile(all_similarities, [50, 75, 90, 95])
    thresholds = [percentiles[3], percentiles[2], percentiles[1], threshold]
    
    logging.info(f"適応的閾値: {[f'{t:.3f}' for t in thresholds]}")
    
    clusters = []
    used_old = set()
    used_new = set()
    
    # 段階的にクラスター形成
    for i, adaptive_threshold in enumerate(thresholds):
        stage_old_to_new = {}
        stage_new_to_old = {}
        stage_similarities = {}
        
        # 未使用チャンクでの関係抽出
        for old_id, new_ids in old_to_new.items():
            if old_id in used_old:
                continue
            stage_old_to_new[old_id] = set()
            
            for new_id in new_ids:
                if new_id in used_new:
                    continue
                sim = similarities.get((old_id, new_id), 0.0)
                if sim >= adaptive_threshold:
                    stage_old_to_new[old_id].add(new_id)
                    if new_id not in stage_new_to_old:
                        stage_new_to_old[new_id] = set()
                    stage_new_to_old[new_id].add(old_id)
                    stage_similarities[(old_id, new_id)] = sim
        
        # この段階でのクラスター形成（サイズ制限付き）
        if i < 2:  # 高い段階では小さなクラスターのみ
            stage_clusters = form_small_clusters_only(
                stage_old_to_new, stage_new_to_old, stage_similarities,
                max_size=8
            )
        else:  # 低い段階では中程度まで許可
            stage_clusters = form_small_clusters_only(
                stage_old_to_new, stage_new_to_old, stage_similarities,
                max_size=CLUSTERING_CONFIG["max_cluster_size"]
            )
        
        clusters.extend(stage_clusters)
        
        # 使用済みチャンクを更新
        for cluster in stage_clusters:
            used_old.update(cluster['old'])
            used_new.update(cluster['new'])
        
        logging.info(f"段階{i+1} (閾値={adaptive_threshold:.3f}): {len(stage_clusters)}個のクラスター")
    
    logging.info(f"適応的クラスター結果: 合計{len(clusters)}個")
    return clusters

def form_small_clusters_only(old_to_new, new_to_old, similarities, max_size=8):
    """
    サイズ制限付きクラスター形成
    """
    visited_old = set()
    visited_new = set()
    clusters = []
    
    def explore_small_cluster(start_old_id):
        cluster_old = set()
        cluster_new = set()
        cluster_similarities = {}
        queue = [('old', start_old_id)]
        
        while queue and len(cluster_old) + len(cluster_new) < max_size:
            node_type, node_id = queue.pop(0)
            
            if node_type == 'old':
                if node_id in visited_old:
                    continue
                visited_old.add(node_id)
                cluster_old.add(node_id)
                
                # 関連する新チャンクを追加
                for new_id in old_to_new.get(node_id, []):
                    sim = similarities.get((node_id, new_id), 0.0)
                    cluster_similarities[(node_id, new_id)] = sim
                    if new_id not in visited_new and len(cluster_old) + len(cluster_new) < max_size:
                        queue.append(('new', new_id))
            
            else:  # node_type == 'new'
                if node_id in visited_new:
                    continue
                visited_new.add(node_id)
                cluster_new.add(node_id)
                
                # 関連する旧チャンクを追加
                for old_id in new_to_old.get(node_id, []):
                    sim = similarities.get((old_id, node_id), 0.0)
                    cluster_similarities[(old_id, node_id)] = sim
                    if old_id not in visited_old and len(cluster_old) + len(cluster_new) < max_size:
                        queue.append(('old', old_id))
        
        if cluster_old and cluster_new:
            return {
                'old': sorted(list(cluster_old)),
                'new': sorted(list(cluster_new)),
                'similarities': cluster_similarities,
                'strength': calculate_cluster_strength(cluster_old, cluster_new, cluster_similarities)
            }
        return None
    
    # 全ての未訪問旧チャンクから探索開始
    for old_id in old_to_new.keys():
        if old_id not in visited_old:
            cluster = explore_small_cluster(old_id)
            if cluster:
                clusters.append(cluster)
    
    return clusters

def find_connected_clusters(old_to_new, new_to_old, similarities):
    """
    深さ優先探索（DFS）を使用して強く結合されたチャンククラスターを特定
    """
    visited_old = set()
    visited_new = set()
    clusters = []
    
    def dfs_explore_cluster(start_old_id):
        """一つのクラスターを完全に探索"""
        cluster_old = set()
        cluster_new = set()
        cluster_similarities = {}
        
        def visit_old(old_id):
            if old_id in visited_old:
                return
            visited_old.add(old_id)
            cluster_old.add(old_id)
            
            # この旧チャンクに関連する全ての新チャンクを探索
            for new_id in old_to_new.get(old_id, []):
                similarity = similarities.get((old_id, new_id), 0.0)
                cluster_similarities[(old_id, new_id)] = similarity
                visit_new(new_id)
        
        def visit_new(new_id):
            if new_id in visited_new:
                return
            visited_new.add(new_id)
            cluster_new.add(new_id)
            
            # この新チャンクに関連する全ての旧チャンクを探索
            for old_id in new_to_old.get(new_id, []):
                similarity = similarities.get((old_id, new_id), 0.0)
                cluster_similarities[(old_id, new_id)] = similarity
                visit_old(old_id)
        
        # 探索開始
        visit_old(start_old_id)
        
        return {
            'old': sorted(list(cluster_old)),
            'new': sorted(list(cluster_new)),
            'similarities': cluster_similarities,
            'strength': calculate_cluster_strength(cluster_old, cluster_new, cluster_similarities)
        }
    
    # 全ての旧チャンクから未訪問のものを起点として探索
    for old_id in old_to_new.keys():
        if old_id not in visited_old:
            cluster = dfs_explore_cluster(old_id)
            if cluster['old'] and cluster['new']:  # 空でないクラスターのみ追加
                clusters.append(cluster)
    
    return clusters

def calculate_cluster_strength(old_ids, new_ids, similarities):
    """
    クラスターの結合強度を計算
    複数の指標を組み合わせて総合的な強度を算出
    """
    old_count = len(old_ids)
    new_count = len(new_ids)
    
    if not similarities:
        return 0.0
    
    # 指標1: 平均類似度
    avg_similarity = sum(similarities.values()) / len(similarities)
    
    # 指標2: 最大類似度  
    max_similarity = max(similarities.values())
    
    # 指標3: 結合密度（実際の関係数 / 可能な関係数）
    possible_connections = old_count * new_count
    actual_connections = len(similarities)
    density = actual_connections / possible_connections if possible_connections > 0 else 0.0
    
    # 指標4: サイズペナルティ（大きすぎるクラスターにペナルティ）
    size_penalty = 1.0 / (1.0 + 0.1 * (old_count + new_count - 2))
    
    # 総合強度（重み付き平均）
    strength = (
        0.4 * avg_similarity +      # 平均類似度重視
        0.3 * max_similarity +      # 最強結合重視  
        0.2 * density +             # 密度重視
        0.1 * size_penalty          # サイズ調整
    )
    
    return strength

def determine_group_type(old_count, new_count):
    """クラスターサイズに基づいてグループタイプを決定"""
    if old_count == 1 and new_count == 1:
        return "1:1"
    elif old_count == 1 and new_count > 1:
        return "1:N"
    elif old_count > 1 and new_count == 1:
        return "N:1"
    elif old_count > 1 and new_count > 1:
        return "N:N"
    else:
        return "unknown"


def is_large_cluster(cluster, similarities):
    """
    大きなクラスターかどうかを判定
    
    判定基準:
    1. サイズ: 総チャンク数が6個以上
    2. 大きなサイズ（設定値以上）の場合は強制的に分割対象
    3. そうでない場合は品質基準で判定:
       - 密度: 実際の関係数/可能な関係数 < 0.6
       - 類似度分散: 類似度の標準偏差 > 0.1
       - 類似度範囲: 最大類似度 - 最小類似度 > 0.3
    """
    old_count = len(cluster['old'])
    new_count = len(cluster['new'])
    total_chunks = old_count + new_count
    
    logging.debug(f"クラスター判定開始: 旧{old_count}個, 新{new_count}個, 合計{total_chunks}個")
    
    # 基準1: 最小サイズチェック
    if total_chunks < 6:
        logging.debug(f"クラスター判定: サイズ{total_chunks} < 6, 小さなクラスター")
        return False
    
    # 基準2: 大きなサイズの場合は強制分割
    large_size_threshold = CLUSTERING_CONFIG["large_size_threshold"]
    if total_chunks >= large_size_threshold:
        logging.debug(f"クラスター分析: サイズ={total_chunks} >= {large_size_threshold}, 強制分割対象=True")
        return True
    
    # クラスター内の類似度を取得
    cluster_similarities = []
    for old_id in cluster['old']:
        for new_id in cluster['new']:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim > 0:
                cluster_similarities.append(sim)
    
    if len(cluster_similarities) < 3:
        logging.debug(f"クラスター判定: 類似度数{len(cluster_similarities)} < 3, 判定不可")
        return False
    
    # 基準3: 密度チェック
    possible_connections = old_count * new_count
    actual_connections = len(cluster_similarities)
    density = actual_connections / possible_connections
    
    # 基準4: 類似度分散チェック
    sim_std = np.std(cluster_similarities)
    
    # 基準5: 類似度範囲チェック
    sim_range = max(cluster_similarities) - min(cluster_similarities)
    
    # 品質基準による複合判定（従来の条件）
    is_large_by_quality = (
        density < 0.6 and          # 密度が低い
        sim_std > 0.1 and          # 分散が大きい
        sim_range > 0.3            # 範囲が大きい
    )
    
    logging.debug(f"クラスター分析: サイズ={total_chunks}, 密度={density:.3f}, 分散={sim_std:.3f}, 範囲={sim_range:.3f}, 品質基準大きい={is_large_by_quality}")
    
    return is_large_by_quality

def hierarchical_cluster_refinement(large_cluster, similarities, threshold_high=None, expand_threshold=None):
    """
    大きなクラスターを階層的に細分化（段階的分割版）
    """
    if threshold_high is None:
        threshold_high = SIMILARITY_THRESHOLDS["strict"]
    if expand_threshold is None:
        expand_threshold = CLUSTERING_CONFIG["refinement_threshold"]
    
    original_size = len(large_cluster['old']) + len(large_cluster['new'])
    
    # 1. 高い類似度のペアを抽出
    strong_pairs = []
    for old_id in large_cluster['old']:
        for new_id in large_cluster['new']:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim >= threshold_high:
                strong_pairs.append(((old_id, new_id), sim))
    
    # 2. 強い類似度順にソート
    strong_pairs.sort(key=lambda x: x[1], reverse=True)
    
    logging.debug(f"階層的細分化: {len(strong_pairs)}個の強いペアを検出")
    
    # 3. クラスターサイズに応じた適応的パラメータ
    if original_size >= 100:  # 超巨大クラスター
        target_clusters = min(max(6, len(strong_pairs) // 3), 12)
        max_cluster_size = max(20, original_size // target_clusters)
        min_remaining_for_split = 30  # 残りが30個以上なら再分割
        logging.info(f"パラメータ設定（超巨大）: 目標={target_clusters}, 最大サイズ={max_cluster_size}, 分割閾値={min_remaining_for_split}")
    elif original_size >= 50:  # 大きなクラスター
        target_clusters = min(max(4, len(strong_pairs) // 4), 8)
        max_cluster_size = max(CLUSTERING_CONFIG["max_cluster_size"], original_size // target_clusters)
        min_remaining_for_split = 20
        logging.info(f"パラメータ設定（大）: 目標={target_clusters}, 最大サイズ={max_cluster_size}, 分割閾値={min_remaining_for_split}")
    else:  # 中程度のクラスター
        target_clusters = min(max(2, len(strong_pairs) // 5), 6)
        max_cluster_size = max(CLUSTERING_CONFIG["max_cluster_size"], original_size // target_clusters)
        min_remaining_for_split = CLUSTERING_CONFIG["min_remaining_for_split"]
        logging.info(f"パラメータ設定（中）: 目標={target_clusters}, 最大サイズ={max_cluster_size}, 分割閾値={min_remaining_for_split}")
    
    # 4. 適応的な閾値調整
    all_similarities = list(similarities.values())
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        adaptive_expand_threshold = max(0.7, min(expand_threshold, avg_similarity + 0.05))
        logging.info(f"適応的拡張閾値: {adaptive_expand_threshold:.3f} (平均類似度: {avg_similarity:.3f})")
    else:
        adaptive_expand_threshold = expand_threshold
    
    logging.info(f"段階的分割開始: 元サイズ={original_size}, 強いペア={len(strong_pairs)}個")
    
    # 5. 段階的分割を実行
    result_clusters = perform_staged_clustering(
        large_cluster, similarities, strong_pairs, 
        target_clusters, max_cluster_size, adaptive_expand_threshold,
        min_remaining_for_split
    )
    
    logging.info(f"段階的分割結果: {len(result_clusters)}個のクラスター")
    return result_clusters

def perform_staged_clustering(large_cluster, similarities, strong_pairs, target_clusters, max_cluster_size, expand_threshold, min_remaining_for_split):
    """
    段階的クラスタリング：大→中→小の順で分割
    """
    remaining_cluster = large_cluster
    final_clusters = []
    stage = 1
    
    while True:
        current_size = len(remaining_cluster['old']) + len(remaining_cluster['new'])
        logging.info(f"段階{stage}分割: 残りサイズ={current_size}")
        
        if current_size < min_remaining_for_split:
            # 残りが小さい場合は終了
            if current_size >= 4:  # 最小サイズチェック
                final_clusters.append(remaining_cluster)
                logging.info(f"段階{stage}: 残りクラスター追加 (旧{len(remaining_cluster['old'])}個, 新{len(remaining_cluster['new'])}個)")
            break
        
        # この段階での分割を実行
        stage_clusters = execute_single_stage_clustering(
            remaining_cluster, similarities, strong_pairs, 
            target_clusters, max_cluster_size, expand_threshold
        )
        
        if len(stage_clusters) <= 1:
            # 分割できない場合は終了
            final_clusters.append(remaining_cluster)
            logging.info(f"段階{stage}: 分割不可のため残りクラスターを追加")
            break
        
        # 最大のクラスターを次段階の対象とし、他は確定
        stage_clusters.sort(key=lambda x: len(x['old']) + len(x['new']), reverse=True)
        largest_cluster = stage_clusters[0]
        largest_size = len(largest_cluster['old']) + len(largest_cluster['new'])
        
        # 最大クラスターも十分小さくなった場合は全て確定
        if largest_size < min_remaining_for_split:
            final_clusters.extend(stage_clusters)
            logging.info(f"段階{stage}: 全クラスター確定 ({len(stage_clusters)}個)")
            break
        
        # 小さなクラスターを確定し、最大クラスターは次段階へ
        final_clusters.extend(stage_clusters[1:])
        remaining_cluster = largest_cluster
        logging.info(f"段階{stage}: {len(stage_clusters[1:])}個確定, 最大クラスター(サイズ={largest_size})を次段階へ")
        
        stage += 1
        if stage > 5:  # 無限ループ防止
            final_clusters.append(remaining_cluster)
            logging.info("段階分割: 最大段階数到達のため終了")
            break
    
    logging.info(f"段階的分割完了: {stage}段階, 最終{len(final_clusters)}個のクラスター")
    return final_clusters

def execute_single_stage_clustering(cluster, similarities, strong_pairs, target_clusters, max_cluster_size, expand_threshold):
    """
    単一段階でのクラスタリング実行
    """
    candidate_clusters = []
    threshold_steps = [expand_threshold, expand_threshold - 0.05, expand_threshold - 0.1]
    
    # クラスター内の強いペアのみを使用
    cluster_strong_pairs = []
    cluster_old_set = set(cluster['old'])
    cluster_new_set = set(cluster['new'])
    
    for (old_id, new_id), sim in strong_pairs:
        if old_id in cluster_old_set and new_id in cluster_new_set:
            cluster_strong_pairs.append(((old_id, new_id), sim))
    
    # 各強いペアからクラスターを形成
    for i, ((core_old, core_new), core_sim) in enumerate(cluster_strong_pairs[:target_clusters * 2]):
        for step_threshold in threshold_steps:
            local_cluster = expand_from_core_competitive(
                core_old, core_new, cluster, similarities, 
                expand_threshold=step_threshold,
                max_size=max_cluster_size
            )
            
            if local_cluster and len(local_cluster['old']) + len(local_cluster['new']) >= 4:
                candidate_clusters.append(local_cluster)
                logging.debug(f"候補クラスター{len(candidate_clusters)} (閾値={step_threshold:.3f}): "
                           f"旧{len(local_cluster['old'])}個, 新{len(local_cluster['new'])}個")
                break
    
    # 重複解決とクラスター選択
    if len(candidate_clusters) >= 2:
        selected_clusters = resolve_cluster_overlaps(candidate_clusters, cluster, similarities, target_clusters)
        logging.info(f"重複解決後: {len(selected_clusters)}個のクラスターを選択")
        return selected_clusters
    else:
        logging.info("単一段階クラスタリング: 有効なサブクラスターを生成できませんでした")
        return [cluster]

def expand_from_core_competitive(core_old, core_new, large_cluster, similarities, expand_threshold=0.85, max_size=50):
    """
    コアペアから局所クラスターを拡張（競合版・サイズ制限付き）
    """
    local_old = {core_old}
    local_new = {core_new}
    
    # 段階的に関連チャンクを追加（サイズ制限付き）
    changed = True
    iteration = 0
    max_iterations = 5
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        current_size = len(local_old) + len(local_new)
        if current_size >= max_size:
            logging.debug(f"クラスター拡張: サイズ制限({max_size})に到達")
            break
        
        # 拡張候補を類似度順でソート
        expansion_candidates = []
        
        # 旧チャンク側の候補
        for old_id in large_cluster['old']:
            if old_id in local_old:
                continue
            max_sim_to_local = max([similarities.get((old_id, new_id), 0.0) 
                                   for new_id in local_new], default=0.0)
            if max_sim_to_local >= expand_threshold:
                expansion_candidates.append(('old', old_id, max_sim_to_local))
        
        # 新チャンク側の候補
        for new_id in large_cluster['new']:
            if new_id in local_new:
                continue
            max_sim_to_local = max([similarities.get((old_id, new_id), 0.0) 
                                   for old_id in local_old], default=0.0)
            if max_sim_to_local >= expand_threshold:
                expansion_candidates.append(('new', new_id, max_sim_to_local))
        
        # 類似度の高い順に追加（サイズ制限まで）
        expansion_candidates.sort(key=lambda x: x[2], reverse=True)
        added_this_iteration = 0
        
        for chunk_type, chunk_id, sim_score in expansion_candidates:
            if current_size + added_this_iteration >= max_size:
                break
                
            if chunk_type == 'old':
                local_old.add(chunk_id)
            else:
                local_new.add(chunk_id)
            
            added_this_iteration += 1
            changed = True
    
    if len(local_old) == 0 or len(local_new) == 0:
        return None
    
    # 局所クラスターの類似度情報を収集
    local_similarities = {}
    for old_id in local_old:
        for new_id in local_new:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim > 0:
                local_similarities[(old_id, new_id)] = sim
    
    return {
        'old': sorted(list(local_old)),
        'new': sorted(list(local_new)),
        'similarities': local_similarities,
        'type': determine_group_type(len(local_old), len(local_new)),
        'core_similarity': similarities.get((core_old, core_new), 0.0),
        'refinement_method': 'hierarchical_core'
    }

def resolve_cluster_overlaps(candidate_clusters, original_cluster, similarities, target_count):
    """
    重複するクラスターを解決して最適な組み合わせを選択
    """
    if len(candidate_clusters) <= target_count:
        return candidate_clusters
    
    # 各候補クラスターの品質を評価
    cluster_scores = []
    for i, cluster in enumerate(candidate_clusters):
        # 内部結合度
        cluster_sims = list(cluster['similarities'].values())
        internal_cohesion = sum(cluster_sims) / len(cluster_sims) if cluster_sims else 0.0
        
        # サイズバランス
        size = len(cluster['old']) + len(cluster['new'])
        size_score = min(1.0, size / 20)  # 適度なサイズを評価
        
        # 総合スコア
        total_score = 0.7 * internal_cohesion + 0.3 * size_score
        cluster_scores.append((i, total_score, cluster))
    
    # スコア順にソート
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 貪欲法で重複を最小化しながら選択
    selected_clusters = []
    used_old = set()
    used_new = set()
    
    for i, score, cluster in cluster_scores:
        # 重複チェック
        cluster_old = set(cluster['old'])
        cluster_new = set(cluster['new'])
        
        overlap_old = len(cluster_old & used_old) / len(cluster_old) if cluster_old else 0
        overlap_new = len(cluster_new & used_new) / len(cluster_new) if cluster_new else 0
        max_overlap = max(overlap_old, overlap_new)
        
        # 重複が50%未満なら採用
        if max_overlap < 0.5 and len(selected_clusters) < target_count:
            selected_clusters.append(cluster)
            used_old.update(cluster_old)
            used_new.update(cluster_new)
            logging.info(f"クラスター{i+1}を選択 (スコア={score:.3f}, 重複={max_overlap:.3f})")
    
    # 残ったチャンクの処理
    if len(selected_clusters) > 0:
        remaining_old = [oid for oid in original_cluster['old'] if oid not in used_old]
        remaining_new = [nid for nid in original_cluster['new'] if nid not in used_new]
        remaining_size = len(remaining_old) + len(remaining_new)
        
        if remaining_old and remaining_new and remaining_size >= 4:
            # 残りクラスターが大きい場合は段階的分割を適用しない（単純に追加）
            # 段階的分割は上位レベルで処理される
            remaining_similarities = {}
            for old_id in remaining_old:
                for new_id in remaining_new:
                    sim = similarities.get((old_id, new_id), 0.0)
                    if sim > 0:
                        remaining_similarities[(old_id, new_id)] = sim
            
            selected_clusters.append({
                'old': remaining_old,
                'new': remaining_new,
                'similarities': remaining_similarities,
                'type': determine_group_type(len(remaining_old), len(remaining_new)),
                'refinement_method': 'hierarchical_remaining'
            })
            logging.info(f"残りクラスター追加: 旧{len(remaining_old)}個, 新{len(remaining_new)}個")
    
    return selected_clusters

def semantic_refinement(large_cluster, old_chunks, new_chunks, old_vectors, new_vectors):
    """
    エンベディングの意味的類似性に基づく細分化
    """
    try:
        # 1. クラスター内のチャンクのエンベディングを取得
        cluster_embeddings = []
        chunk_info = []
        
        for old_id in large_cluster['old']:
            idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
            if idx is not None and old_vectors[idx]:
                # チャンク全体のベクトルを直接使用
                cluster_embeddings.append(old_vectors[idx])
                chunk_info.append({'id': old_id, 'type': 'old', 'index': idx})
        
        for new_id in large_cluster['new']:
            idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            if idx is not None and new_vectors[idx]:
                # チャンク全体のベクトルを直接使用
                cluster_embeddings.append(new_vectors[idx])
                chunk_info.append({'id': new_id, 'type': 'new', 'index': idx})
        
        # 2. K-means クラスタリング
        if len(cluster_embeddings) >= 4:  # 最小限の数がある場合
            # 各クラスターの平均サイズを10個程度に設定
            target_avg_size = 10
            ideal_k = len(cluster_embeddings) // target_avg_size
            
            # クラスターサイズに応じた適応的K値設定
            if len(cluster_embeddings) >= 100:  # 超巨大クラスター
                n_clusters = min(20, max(5, ideal_k))
                logging.info(f"K-means設定（超巨大）: K={n_clusters} (理想K={ideal_k}, 総数={len(cluster_embeddings)})")
            elif len(cluster_embeddings) >= 50:  # 大きなクラスター
                n_clusters = min(15, max(4, ideal_k))
                logging.info(f"K-means設定（大）: K={n_clusters} (理想K={ideal_k}, 総数={len(cluster_embeddings)})")
            elif len(cluster_embeddings) >= 20:  # 中程度のクラスター
                n_clusters = min(8, max(3, ideal_k))
                logging.info(f"K-means設定（中）: K={n_clusters} (理想K={ideal_k}, 総数={len(cluster_embeddings)})")
            else:  # 小さなクラスター
                n_clusters = min(4, max(2, len(cluster_embeddings) // 5))
                logging.info(f"K-means設定（小）: K={n_clusters} (総数={len(cluster_embeddings)})")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(cluster_embeddings)
            
            # 全体の類似度情報を取得
            all_similarities = large_cluster.get('similarities', {})
            
            # 3. クラスターごとに分割
            sub_clusters = []
            for cluster_id in range(n_clusters):
                cluster_chunks = [chunk_info[i] for i, label in enumerate(labels) if label == cluster_id]
                
                old_in_cluster = [c['id'] for c in cluster_chunks if c['type'] == 'old']
                new_in_cluster = [c['id'] for c in cluster_chunks if c['type'] == 'new']
                
                if old_in_cluster and new_in_cluster:
                    # このサブクラスターの類似度情報を収集
                    sub_similarities = {}
                    for old_id in old_in_cluster:
                        for new_id in new_in_cluster:
                            if (old_id, new_id) in all_similarities:
                                sub_similarities[(old_id, new_id)] = all_similarities[(old_id, new_id)]
                    
                    sub_clusters.append({
                        'old': old_in_cluster,
                        'new': new_in_cluster,
                        'similarities': sub_similarities,
                        'type': determine_group_type(len(old_in_cluster), len(new_in_cluster)),
                        'semantic_cluster_id': cluster_id,
                        'refinement_method': 'semantic'
                    })
            
            logging.info(f"意味的細分化: {len(sub_clusters)}個のサブクラスターを生成")
            return sub_clusters
        
    except Exception as e:
        logging.warning(f"意味的細分化でエラー: {e}")
    
    return [large_cluster]  # 細分化できない場合は元のまま

def calculate_internal_cohesion(sub_clusters, similarities):
    """
    サブクラスター内部の結合度を計算
    """
    total_cohesion = 0.0
    total_pairs = 0
    
    for cluster in sub_clusters:
        cluster_sims = []
        for old_id in cluster['old']:
            for new_id in cluster['new']:
                sim = similarities.get((old_id, new_id), 0.0)
                if sim > 0:
                    cluster_sims.append(sim)
        
        if cluster_sims:
            total_cohesion += np.mean(cluster_sims) * len(cluster_sims)
            total_pairs += len(cluster_sims)
    
    return total_cohesion / total_pairs if total_pairs > 0 else 0.0

def calculate_external_separation(sub_clusters, similarities):
    """
    サブクラスター間の分離度を計算
    """
    if len(sub_clusters) < 2:
        return 1.0
    
    inter_cluster_sims = []
    
    for i, cluster1 in enumerate(sub_clusters):
        for j, cluster2 in enumerate(sub_clusters[i+1:], i+1):
            # クラスター間の類似度
            for old_id1 in cluster1['old']:
                for new_id2 in cluster2['new']:
                    sim = similarities.get((old_id1, new_id2), 0.0)
                    if sim > 0:
                        inter_cluster_sims.append(sim)
            
            for old_id2 in cluster2['old']:
                for new_id1 in cluster1['new']:
                    sim = similarities.get((old_id2, new_id1), 0.0)
                    if sim > 0:
                        inter_cluster_sims.append(sim)
    
    # 分離度は inter-cluster similarity の逆数
    return 1.0 - (np.mean(inter_cluster_sims) if inter_cluster_sims else 0.0)

def select_best_refinement(refinement_results, original_cluster, similarities):
    """
    最適な細分化結果を選択（改善版）
    """
    scores = {}
    
    # 元のクラスターの特性を分析
    original_size = len(original_cluster['old']) + len(original_cluster['new'])
    original_similarities = list(similarities.values())
    original_avg_sim = sum(original_similarities) / len(original_similarities) if original_similarities else 0.0
    
    logging.info(f"元クラスター分析: サイズ={original_size}, 平均類似度={original_avg_sim:.3f}")
    
    # 巨大クラスター用の緩和された評価基準を事前に設定
    if original_size >= 50:  # 巨大クラスターの場合
        min_score_threshold = 0.2  # 閾値を緩和
        cohesion_weight = 0.4
        separation_weight = 0.2
        balance_weight = 0.3
        penalty_weight = 0.1
    else:
        min_score_threshold = 0.3
        cohesion_weight = 0.5
        separation_weight = 0.3
        balance_weight = 0.1
        penalty_weight = 0.1
    
    for method, sub_clusters in refinement_results.items():
        if not sub_clusters or len(sub_clusters) <= 1:
            scores[method] = 0.0
            logging.info(f"細分化評価 {method}: サブクラスター数不足 (数={len(sub_clusters) if sub_clusters else 0})")
            continue
        
        # 評価指標：内部結合度と外部分離度
        internal_cohesion = calculate_internal_cohesion(sub_clusters, similarities)
        external_separation = calculate_external_separation(sub_clusters, similarities)
        
        # 細分化効果の評価
        sub_cluster_sizes = [len(cluster['old']) + len(cluster['new']) for cluster in sub_clusters]
        size_balance = 1.0 - (max(sub_cluster_sizes) / sum(sub_cluster_sizes)) if sum(sub_cluster_sizes) > 0 else 0.0
        
        # 細分化数のペナルティ（過度な細分化を防ぐ）
        num_clusters = len(sub_clusters)
        if num_clusters > original_size / 4:  # 過度な細分化
            size_penalty = 0.5
        else:
            size_penalty = 1.0 / (1.0 + 0.1 * (num_clusters - 2))
        
        scores[method] = (
            cohesion_weight * internal_cohesion + 
            separation_weight * external_separation + 
            balance_weight * size_balance +
            penalty_weight * size_penalty
        )
        
        logging.info(f"細分化評価 {method}: 内部結合={internal_cohesion:.3f}, 外部分離={external_separation:.3f}, "
                    f"サイズバランス={size_balance:.3f}, スコア={scores[method]:.3f}")
    
    if not scores or max(scores.values()) < min_score_threshold:
        logging.info(f"細分化スコアが低いため、元のクラスターを維持 (最高スコア={max(scores.values()) if scores else 0:.3f}, 閾値={min_score_threshold})")
        return [original_cluster]
    
    best_method = max(scores, key=scores.get)
    best_score = scores[best_method]
    logging.info(f"最適な細分化方法: {best_method} (スコア={best_score:.3f})")
    return refinement_results[best_method]

def refine_large_clusters(clusters, similarities, old_chunks, new_chunks, old_vectors, new_vectors, refinement_mode="auto"):
    """
    大きなクラスターを細分化（改善版）
    """
    refined_clusters = []
    
    for i, cluster in enumerate(clusters, 1):
        if is_large_cluster(cluster, similarities):
            cluster_size = len(cluster['old']) + len(cluster['new'])
            logging.info(f"大きなクラスター{i}を検出: 旧{len(cluster['old'])}個, 新{len(cluster['new'])}個 (合計{cluster_size}個)")
            
            if refinement_mode == "none":
                logging.info(f"クラスター{i}: 細分化モード'none'のため処理をスキップ")
                refined_clusters.append(cluster)
                continue
                
            refinement_results = {}
            
            # 階層的細分化
            if refinement_mode in ["auto", "hierarchical"]:
                logging.info(f"クラスター{i}: 階層的細分化を開始")
                hierarchical_result = hierarchical_cluster_refinement(cluster, similarities)
                refinement_results['hierarchical'] = hierarchical_result
                logging.debug(f"クラスター{i}: 階層的細分化完了 -> {len(hierarchical_result)}個のサブクラスター")
            
            # 意味的細分化
            if refinement_mode in ["auto", "semantic"]:
                logging.debug(f"クラスター{i}: 意味的細分化を開始")
                semantic_result = semantic_refinement(cluster, old_chunks, new_chunks, old_vectors, new_vectors)
                refinement_results['semantic'] = semantic_result
                logging.debug(f"クラスター{i}: 意味的細分化完了 -> {len(semantic_result)}個のサブクラスター")
            
            # 最適な細分化を選択
            if refinement_mode == "auto":
                logging.debug(f"クラスター{i}: 最適な細分化方法を選択中...")
                best_refinement = select_best_refinement(refinement_results, cluster, similarities)
                logging.debug(f"クラスター{i}: 選択完了 -> {len(best_refinement)}個のクラスター")
            else:
                best_refinement = refinement_results.get(refinement_mode, [cluster])
                logging.debug(f"クラスター{i}: {refinement_mode}細分化結果を採用 -> {len(best_refinement)}個のクラスター")
            
            refined_clusters.extend(best_refinement)
        else:
            logging.debug(f"クラスター{i}: サイズが小さいため細分化をスキップ (旧{len(cluster['old'])}個, 新{len(cluster['new'])}個)")
            refined_clusters.append(cluster)
    
    logging.debug(f"細分化処理完了: {len(clusters)}個 -> {len(refined_clusters)}個のクラスター")
    return refined_clusters

def find_similar_groups(
    old_chunks: List[Dict], new_chunks: List[Dict],
    old_vectors: List[List[float]], new_vectors: List[List[float]],
    threshold: float = None,
    refinement_mode: str = "auto",
    force_clustering: bool = True,
    initial_clustering_mode: str = "strict",
    structural_integration: bool = True,
    perfect_match_threshold: float = None,
    use_hierarchy_constraints: bool = True
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    強く結合されたチャンククラスターを検出し、適切なグループ化を行う
    refinement_mode: "auto", "hierarchical", "semantic", "none"
    initial_clustering_mode: "strict", "relaxed", "adaptive"
    perfect_match_threshold: 完全一致とみなす閾値（この値以上は完全一致として除外）
    use_hierarchy_constraints: 階層制約を使用するかどうか
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLDS["default"]
    
    # 完全一致閾値のデフォルト設定（通常の閾値より高く設定）
    if perfect_match_threshold is None:
        perfect_match_threshold = SIMILARITY_THRESHOLDS["bypass"]  # bypass閾値を直接使用
    
    logging.info(f"類似度処理開始: 通常閾値={threshold:.3f}, 完全一致閾値={perfect_match_threshold:.3f}, 階層制約={'有効' if use_hierarchy_constraints else '無効'}")
    
    old_chunk_ids = [c["id"] for c in old_chunks]
    new_chunk_ids = [c["id"] for c in new_chunks]
    
    # 階層制約を使用する場合は拡張類似度計算を実行
    if use_hierarchy_constraints:
        logging.info("階層制約付き類似度計算を実行中...")
        enhanced_similarities = enhanced_similarity_calculation(old_chunks, new_chunks, old_vectors, new_vectors)
        
        # 拡張類似度から従来形式のデータ構造を構築
        old_to_new = {cid: set() for cid in old_chunk_ids}
        new_to_old = {cid: set() for cid in new_chunk_ids}
        chunk_similarities = {}
        content_similarities = {}  # バイパス判定用の内容類似度を保存
        
        for (old_id, new_id), sim_info in enhanced_similarities.items():
            final_similarity = sim_info['similarity']
            content_similarity = sim_info['content_similarity']
            
            if final_similarity >= threshold:
                old_to_new[old_id].add(new_id)
                new_to_old[new_id].add(old_id)
                chunk_similarities[(old_id, new_id)] = final_similarity
                content_similarities[(old_id, new_id)] = content_similarity  # 内容類似度を保存
        
        logging.debug(f"階層制約付き類似度計算完了: 閾値超え={len(chunk_similarities)}個")
    else:
        # 従来の類似度計算
        logging.info("従来の類似度計算を実行中...")
        old_to_new = {cid: set() for cid in old_chunk_ids}
        new_to_old = {cid: set() for cid in new_chunk_ids}
        chunk_similarities = {}
        content_similarities = {}  # 従来方式では最終類似度と同じ
        
        for i, old_chunk in enumerate(old_chunks):
            for j, new_chunk in enumerate(new_chunks):
                max_sim = calculate_chunk_similarity(old_vectors[i], new_vectors[j])
                if max_sim >= threshold:
                    old_to_new[old_chunk["id"]].add(new_chunk["id"])
                    new_to_old[new_chunk["id"]].add(old_chunk["id"])
                    chunk_similarities[(old_chunk["id"], new_chunk["id"])] = max_sim
                    content_similarities[(old_chunk["id"], new_chunk["id"])] = max_sim  # 同じ値
        
        logging.info(f"従来類似度計算完了: 閾値超え={len(chunk_similarities)}個")
    
    # 完全一致ペアの検出と処理
    perfect_matches = []
    perfect_match_used_old = set()
    perfect_match_used_new = set()
    
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            # 完全一致判定は常に内容類似度で行う
            content_sim = calculate_chunk_similarity(old_vectors[i], new_vectors[j])
            
            if content_sim >= perfect_match_threshold:
                perfect_match_pair = {
                    'old': [old_chunk["id"]],
                    'new': [new_chunk["id"]],
                    'type': '1:1',
                    'strength': content_sim,
                    'refinement_method': 'perfect_match',
                    'forced_clustering': False,
                    'similarities': {(old_chunk["id"], new_chunk["id"]): content_sim},
                    'perfect_match': True
                }
                perfect_matches.append(perfect_match_pair)
                perfect_match_used_old.add(old_chunk["id"])
                perfect_match_used_new.add(new_chunk["id"])
    
    logging.info(f"完全一致検出完了: {len(perfect_matches)}個")
    
    # 完全一致で使用されたチャンクを除外してクラスタリング用のデータを準備
    clustering_old_to_new = {}
    clustering_new_to_old = {}
    clustering_similarities = {}
    
    for old_id, new_ids in old_to_new.items():
        if old_id not in perfect_match_used_old:
            clustering_old_to_new[old_id] = set()
            for new_id in new_ids:
                if new_id not in perfect_match_used_new:
                    clustering_old_to_new[old_id].add(new_id)
                    if new_id not in clustering_new_to_old:
                        clustering_new_to_old[new_id] = set()
                    clustering_new_to_old[new_id].add(old_id)
                    clustering_similarities[(old_id, new_id)] = chunk_similarities[(old_id, new_id)]
    
    logging.info(f"クラスタリング用データ準備完了: 対象類似度={len(clustering_similarities)}個")

    # 初期クラスター形成方法を選択（完全一致除外後のデータで実行）
    if initial_clustering_mode == "strict":
        clusters = find_strict_clusters(clustering_old_to_new, clustering_new_to_old, clustering_similarities, threshold)
    elif initial_clustering_mode == "adaptive":
        clusters = find_adaptive_clusters(clustering_old_to_new, clustering_new_to_old, clustering_similarities, threshold)
    else:  # "relaxed" (従来方式)
        clusters = find_connected_clusters(clustering_old_to_new, clustering_new_to_old, clustering_similarities)
    
    logging.info(f"初期クラスター形成完了: {len(clusters)}個 (モード: {initial_clustering_mode})")
    
    # クラスター強度順にソート
    clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    # 大きなクラスターを細分化
    if refinement_mode != "none":
        clusters = refine_large_clusters(clusters, clustering_similarities, old_chunks, new_chunks, old_vectors, new_vectors, refinement_mode)
        # 細分化後に再度強度順にソート
        for cluster in clusters:
            if 'strength' not in cluster:
                cluster['strength'] = calculate_cluster_strength(cluster['old'], cluster['new'], 
                                                               {k: v for k, v in clustering_similarities.items() 
                                                                if k[0] in cluster['old'] and k[1] in cluster['new']})
        clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    # クラスターをグループに変換
    groups = []
    used_old = set(perfect_match_used_old)  # 完全一致で使用済みのチャンクを初期値に設定
    used_new = set(perfect_match_used_new)
    
    # 完全一致グループを最初に追加
    groups.extend(perfect_matches)
    
    for cluster in clusters:
        old_ids = cluster['old']
        new_ids = cluster['new']
        
        # 既に使用されたチャンクを除外
        available_old = [oid for oid in old_ids if oid not in used_old]
        available_new = [nid for nid in new_ids if nid not in used_new]
        
        if available_old and available_new:
            group_type = determine_group_type(len(available_old), len(available_new))
            
            group_data = {
                "old": available_old,
                "new": available_new,
                "type": group_type,
                "strength": cluster['strength'],
                "refinement_method": cluster.get('refinement_method', 'original'),
                "similarities": {k: v for k, v in cluster.get('similarities', {}).items() 
                               if k[0] in available_old and k[1] in available_new},
                "content_similarities": {k: content_similarities.get(k, v) for k, v in cluster.get('similarities', {}).items() 
                                       if k[0] in available_old and k[1] in available_new}  # バイパス判定用
            }
            
            # 階層制約情報を追加
            if use_hierarchy_constraints:
                group_data['hierarchy_enhanced'] = True
            
            groups.append(group_data)
            
            used_old.update(available_old)
            used_new.update(available_new)
            
            # 高強度グループのログ記録
            if cluster['strength'] >= 0.99:
                logging.debug(f"高強度グループ生成: {group_type}マッチ: {available_old} -> {available_new} (強度: {cluster['strength']:.6f})")
                logging.debug(f"  細分化方法: {cluster.get('refinement_method', 'original')}")
            else:
                logging.debug(f"{group_type}マッチ: {available_old} -> {available_new} (強度: {cluster['strength']:.4f})")
    
    # デバッグ用：生成されたグループタイプの統計を出力
    group_type_debug_stats = {}
    for group in groups:
        group_type = group['type']
        if group_type not in group_type_debug_stats:
            group_type_debug_stats[group_type] = 0
        group_type_debug_stats[group_type] += 1
    
    logging.info(f"生成されたグループタイプ統計: {group_type_debug_stats}")
    
    # 孤立チャンクの強制的クラスター化（完全一致で使用されたチャンクは除外）
    orphan_old = [cid for cid in old_chunk_ids if cid not in used_old]
    orphan_new = [cid for cid in new_chunk_ids if cid not in used_new]
    
    if force_clustering and (orphan_old or orphan_new):
        logging.info(f"孤立チャンク検出: 旧{len(orphan_old)}個, 新{len(orphan_new)}個 (完全一致除外済み)")
        forced_groups = force_orphan_clustering(
            orphan_old, orphan_new, old_chunks, new_chunks, 
            old_vectors, new_vectors, groups, used_old, used_new,
            clustering_similarities, threshold, structural_integration,
            perfect_match_threshold  # 完全一致閾値を渡す
        )
        groups.extend(forced_groups)
        
        # 強制的にクラスター化されたチャンクを使用済みに追加
        for group in forced_groups:
            used_old.update(group['old'])
            used_new.update(group['new'])
    elif not force_clustering and (orphan_old or orphan_new):
        logging.info(f"強制クラスター化無効: 孤立チャンク 旧{len(orphan_old)}個, 新{len(orphan_new)}個をスキップ")
    
    # 最終的な孤立チャンク（強制クラスター化できなかった）
    deleted = [cid for cid in old_chunk_ids if cid not in used_old]
    added = [cid for cid in new_chunk_ids if cid not in used_new]
    
    logging.info(f"処理結果: グループ{len(groups)}個 (完全一致{len(perfect_matches)}個含む), 削除{len(deleted)}個, 追加{len(added)}個")
    
    return groups, added, deleted

def force_orphan_clustering(
    orphan_old, orphan_new, old_chunks, new_chunks, 
    old_vectors, new_vectors, existing_groups, used_old, used_new,
    chunk_similarities, original_threshold, structural_integration,
    perfect_match_threshold
):
    """
    孤立チャンクを強制的にクラスター化する複数段階のアプローチ
    
    1. 段階的閾値降下法 - より低い閾値で再マッチング
    2. 最近傍マッチング - 最も類似度の高いペアを形成
    3. クラスター拡張法 - 既存クラスターへの追加可能性を検討
    4. 残存チャンク間クラスター探索 - 残った新旧チャンク間で追加クラスター形成 (NEW)
    5. 構造的組み入れ法 - 前後チャンクのクラスターに組み入れ
    6. 残り物同士マッチング - 残った孤立チャンク同士を組み合わせ
    
    注意: 完全一致閾値以上の類似度は除外される
    """
    forced_groups = []
    remaining_old = set(orphan_old)
    remaining_new = set(orphan_new)
    
    logging.info(f"孤立チャンク強制クラスター化開始: 旧{len(remaining_old)}個, 新{len(remaining_new)}個 (完全一致閾値={perfect_match_threshold:.3f})")
    
    # アプローチ1: 段階的閾値降下法
    thresholds = [0.7, 0.5, 0.3, 0.1]  # 段階的に閾値を下げる
    for low_threshold in thresholds:
        if not remaining_old or not remaining_new:
            break
            
        logging.info(f"段階的閾値降下: 閾値={low_threshold}")
        threshold_groups = apply_lower_threshold_matching(
            remaining_old, remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, low_threshold, perfect_match_threshold
        )
        
        for group in threshold_groups:
            forced_groups.append(group)
            remaining_old -= set(group['old'])
            remaining_new -= set(group['new'])
            logging.debug(f"低閾値マッチング成功 (閾値={low_threshold}): {group['old']} -> {group['new']}")
    
    # アプローチ2: 既存クラスターへの拡張可能性チェック
    if remaining_old or remaining_new:
        logging.debug(f"クラスター拡張検討: 残り旧{len(remaining_old)}個, 新{len(remaining_new)}個")
        expansion_groups = try_cluster_expansion(
            remaining_old, remaining_new, existing_groups, old_chunks, new_chunks,
            old_vectors, new_vectors, min_similarity=0.2, perfect_match_threshold=perfect_match_threshold
        )
        
        for group in expansion_groups:
            forced_groups.append(group)
            remaining_old -= set(group['old'])
            remaining_new -= set(group['new'])
            logging.debug(f"クラスター拡張成功: {group['old']} -> {group['new']}")
    
    # アプローチ3: 最近傍マッチング
    if remaining_old and remaining_new:
        logging.info(f"最近傍マッチング: 残り旧{len(remaining_old)}個, 新{len(remaining_new)}個")
        nearest_groups = apply_nearest_neighbor_matching(
            remaining_old, remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, perfect_match_threshold
        )
        
        for group in nearest_groups:
            forced_groups.append(group)
            remaining_old -= set(group['old'])
            remaining_new -= set(group['new'])
            logging.info(f"最近傍マッチング成功: {group['old']} -> {group['new']}")
    
    # アプローチ4: 残存チャンク間クラスター探索 (NEW)
    if remaining_old and remaining_new:
        logging.info(f"残存チャンク間クラスター探索: 残り旧{len(remaining_old)}個, 新{len(remaining_new)}個")
        
        # 非常に低い閾値でクラスター形成を試行
        residual_clusters = explore_residual_clusters(
            remaining_old, remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, min_threshold=0.05, perfect_match_threshold=perfect_match_threshold
        )
        
        for group in residual_clusters:
            forced_groups.append(group)
            remaining_old -= set(group['old'])
            remaining_new -= set(group['new'])
            logging.info(f"残存クラスター形成成功: {group['old']} -> {group['new']} (強度={group['strength']:.3f})")
    
    # アプローチ5: 構造的組み入れ法
    if structural_integration and (remaining_old or remaining_new):
        logging.info(f"構造的組み入れ検討: 残り旧{len(remaining_old)}個, 新{len(remaining_new)}個")
        
        # 組み入れ前の状態を記録
        pre_integration_old = len(remaining_old)
        pre_integration_new = len(remaining_new)
        
        # 構造的組み入れを実行（既存グループに直接統合される）
        structural_groups = apply_structural_integration(
            remaining_old, remaining_new, old_chunks, new_chunks,
            existing_groups, forced_groups
        )
        
        # 統合後の状況をログ出力
        post_integration_old = len(remaining_old)
        post_integration_new = len(remaining_new)
        
        integrated_old_count = pre_integration_old - post_integration_old
        integrated_new_count = pre_integration_new - post_integration_new
        
        if integrated_old_count > 0 or integrated_new_count > 0:
            logging.info(f"構造的組み入れ完了: 旧{integrated_old_count}個, 新{integrated_new_count}個を既存グループに統合")
            
            # 構造的組み入れで統合されたチャンクをused_oldとused_newに追加
            # remaining_oldとremaining_newから削除されたチャンクを特定し、used_*に追加する
            for group in existing_groups + forced_groups:
                for old_id in group.get('old', []):
                    if old_id not in used_old:
                        used_old.add(old_id)
                for new_id in group.get('new', []):
                    if new_id not in used_new:
                        used_new.add(new_id)
        else:
            logging.info("構造的組み入れ: 統合可能なチャンクなし")
        
        # 返された独立グループがあれば追加（通常は空）
        for group in structural_groups:
            forced_groups.append(group)
            remaining_old -= set(group['old'])
            remaining_new -= set(group['new'])
            logging.info(f"構造的組み入れ独立グループ: {group['old']} -> {group['new']}")
    elif not structural_integration:
        logging.info("構造的組み入れ無効化")
    
    # アプローチ6: 残り物の統合（サイズ制限付き）
    if remaining_old or remaining_new:
        logging.info(f"残り物統合: 旧{len(remaining_old)}個, 新{len(remaining_new)}個")
        
        # 完全孤立チャンクの個別処理（新チャンクのみの場合）
        if not remaining_old and remaining_new:
            logging.info(f"完全孤立新チャンクの個別組み入れ: {len(remaining_new)}個")
            for new_id in list(remaining_new):
                # 各孤立チャンクを個別の「追加」グループとして作成
                individual_group = {
                    'old': [],
                    'new': [new_id],
                    'type': '追加',
                    'strength': 0.05,  # 最低強度
                    'refinement_method': 'isolated_individual_addition',
                    'forced_clustering': True,
                    'integration_info': {
                        'neighbor_direction': 'isolated',
                        'neighbor_offset': 999,
                        'neighbor_group_id': -1,
                        'integration_reason': '完全孤立チャンクの個別追加'
                    },
                    'similarities': {}
                }
                forced_groups.append(individual_group)
                logging.info(f"個別孤立組み入れ: -> {new_id}")
            
            remaining_new.clear()
        
        # 完全孤立チャンクの個別処理（旧チャンクのみの場合）
        elif remaining_old and not remaining_new:
            logging.info(f"完全孤立旧チャンクの個別組み入れ: {len(remaining_old)}個")
            for old_id in list(remaining_old):
                # 各孤立チャンクを個別の「削除」グループとして作成
                individual_group = {
                    'old': [old_id],
                    'new': [],
                    'type': '削除',
                    'strength': 0.05,  # 最低強度
                    'refinement_method': 'isolated_individual_deletion',
                    'forced_clustering': True,
                    'integration_info': {
                        'neighbor_direction': 'isolated',
                        'neighbor_offset': 999,
                        'neighbor_group_id': -1,
                        'integration_reason': '完全孤立チャンクの個別削除'
                    },
                    'similarities': {}
                }
                forced_groups.append(individual_group)
                logging.info(f"個別孤立組み入れ: {old_id} -> ")
            
            remaining_old.clear()
        
        elif len(remaining_old) <= 10 and len(remaining_new) <= 10:  # 従来のサイズ制限
            if remaining_old and remaining_new:
                # 残り物を一つの大きなグループにまとめる
                remaining_group = {
                    'old': list(remaining_old),
                    'new': list(remaining_new),
                    'type': determine_group_type(len(remaining_old), len(remaining_new)),
                    'strength': 0.1,  # 低い強度
                    'refinement_method': 'orphan_consolidation',
                    'forced_clustering': True,
                    'similarities': {}
                }
                forced_groups.append(remaining_group)
                logging.info(f"残り物統合: {list(remaining_old)} -> {list(remaining_new)}")
                remaining_old.clear()
                remaining_new.clear()
        else:
            logging.info(f"残り物が多すぎるため統合をスキップ: 旧{len(remaining_old)}個, 新{len(remaining_new)}個")
    
    logging.info(f"強制クラスター化完了: {len(forced_groups)}個のグループ生成, 最終残り旧{len(remaining_old)}個, 新{len(remaining_new)}個")
    return forced_groups

def apply_lower_threshold_matching(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, threshold, perfect_match_threshold):
    """低い閾値での再マッチング（完全一致除外）"""
    matches = []
    old_to_new = {}
    new_to_old = {}
    similarities = {}
    
    # 残りのチャンク間で類似度を再計算
    for old_id in remaining_old:
        old_idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
        if old_idx is None:
            continue
            
        for new_id in remaining_new:
            new_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            if new_idx is None:
                continue
                
            max_sim = calculate_chunk_similarity(old_vectors[old_idx], new_vectors[new_idx])
            # 完全一致閾値以上は除外し、通常閾値以上のみを対象とする
            if threshold <= max_sim < perfect_match_threshold:
                if old_id not in old_to_new:
                    old_to_new[old_id] = set()
                if new_id not in new_to_old:
                    new_to_old[new_id] = set()
                    
                old_to_new[old_id].add(new_id)
                new_to_old[new_id].add(old_id)
                similarities[(old_id, new_id)] = max_sim
    
    # 小さなクラスターを形成
    if old_to_new and new_to_old:
        mini_clusters = find_connected_clusters(old_to_new, new_to_old, similarities)
        for cluster in mini_clusters:
            if cluster['old'] and cluster['new']:
                cluster['refinement_method'] = 'low_threshold_forced'
                cluster['forced_clustering'] = True
                matches.append(cluster)
    
    return matches

def try_cluster_expansion(remaining_old, remaining_new, existing_groups, old_chunks, new_chunks, old_vectors, new_vectors, min_similarity=0.2, perfect_match_threshold=None):
    """既存クラスターに孤立チャンクを追加可能かチェック（完全一致除外）"""
    if perfect_match_threshold is None:
        perfect_match_threshold = SIMILARITY_THRESHOLDS["bypass"]
    expansion_groups = []
    
    # まだ十分に小さい既存グループを対象とする
    expandable_groups = [g for g in existing_groups if len(g['old']) + len(g['new']) <= 20]
    
    for old_id in list(remaining_old):
        old_idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
        if old_idx is None:
            continue
            
        best_group = None
        best_similarity = 0.0
        
        # 各既存グループとの類似度を計算
        for group in expandable_groups:
            group_similarity = 0.0
            connection_count = 0
            
            # グループ内の新チャンクとの類似度を計算
            for new_id in group['new']:
                new_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
                if new_idx is not None:
                    sim = calculate_chunk_similarity(old_vectors[old_idx], new_vectors[new_idx])
                    # 完全一致閾値以上は除外
                    if min_similarity <= sim < perfect_match_threshold:
                        group_similarity += sim
                        connection_count += 1
            
            avg_similarity = group_similarity / connection_count if connection_count > 0 else 0.0
            
            if avg_similarity > best_similarity and connection_count >= 1:
                best_similarity = avg_similarity
                best_group = group
        
        # 十分な類似度があれば拡張グループを作成
        if best_group and best_similarity >= min_similarity:
            expansion_group = {
                'old': [old_id],
                'new': [],  # 新チャンクは対応する最も近いものを選択
                'type': '1:1',  # 暫定
                'strength': best_similarity,
                'refinement_method': 'cluster_expansion_forced',
                'forced_clustering': True,
                'parent_group': best_group,
                'similarities': {}
            }
            
            # 最も類似度の高い新チャンクを1つ選択（残っているもの）
            best_new_id = None
            best_new_sim = 0.0
            
            for new_id in remaining_new:
                new_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
                if new_idx is not None:
                    sim = calculate_chunk_similarity(old_vectors[old_idx], new_vectors[new_idx])
                    # 完全一致閾値以上は除外
                    if sim < perfect_match_threshold and sim > best_new_sim:
                        best_new_sim = sim
                        best_new_id = new_id
            
            if best_new_id and best_new_sim >= min_similarity:
                expansion_group['new'] = [best_new_id]
                expansion_group['similarities'][(old_id, best_new_id)] = best_new_sim
                expansion_groups.append(expansion_group)
                remaining_old.discard(old_id)
                remaining_new.discard(best_new_id)
    
    return expansion_groups

def apply_nearest_neighbor_matching(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, perfect_match_threshold):
    """最近傍マッチング：残ったチャンク同士で最も近いペアを形成（完全一致除外）"""
    matches = []
    old_list = list(remaining_old)
    new_list = list(remaining_new)
    
    # 全ペア間の類似度を計算
    similarity_matrix = []
    for old_id in old_list:
        old_idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
        row_similarities = []
        
        for new_id in new_list:
            new_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            
            if old_idx is not None and new_idx is not None:
                sim = calculate_chunk_similarity(old_vectors[old_idx], new_vectors[new_idx])
                # 完全一致閾値以上は除外
                if sim >= perfect_match_threshold:
                    sim = 0.0  # 完全一致は除外
                row_similarities.append(sim)
            else:
                row_similarities.append(0.0)
        
        similarity_matrix.append(row_similarities)
    
    # 貪欲法で最適なペアを選択
    used_old_indices = set()
    used_new_indices = set()
    
    # 類似度順にソートして処理
    all_pairs = []
    for i, old_id in enumerate(old_list):
        for j, new_id in enumerate(new_list):
            all_pairs.append((similarity_matrix[i][j], i, j, old_id, new_id))
    
    all_pairs.sort(reverse=True)  # 類似度の高い順
    
    for sim, old_idx, new_idx, old_id, new_id in all_pairs:
        if old_idx not in used_old_indices and new_idx not in used_new_indices and sim > 0.05:  # 最低閾値
            match_group = {
                'old': [old_id],
                'new': [new_id],
                'type': '1:1',
                'strength': sim,
                'refinement_method': 'nearest_neighbor_forced',
                'forced_clustering': True,
                'similarities': {(old_id, new_id): sim}
            }
            matches.append(match_group)
            used_old_indices.add(old_idx)
            used_new_indices.add(new_idx)
    
    return matches

def apply_structural_integration(remaining_old, remaining_new, old_chunks, new_chunks, existing_groups, forced_groups):
    """
    構造的組み入れ法：前後のチャンクが所属するクラスターに孤立チャンクを組み入れ
    """
    # 全てのグループ（既存 + 強制生成済み）をマージ
    all_groups = existing_groups + forced_groups
    
    # チャンクID → グループのマッピングを作成
    chunk_to_groups = {}
    for i, group in enumerate(all_groups):
        for old_id in group['old']:
            if old_id not in chunk_to_groups:
                chunk_to_groups[old_id] = []
            chunk_to_groups[old_id].append(('old', i, group))
        
        for new_id in group['new']:
            if new_id not in chunk_to_groups:
                chunk_to_groups[new_id] = []
            chunk_to_groups[new_id].append(('new', i, group))
    
    # チャンクIDから数値インデックスを抽出する関数
    def extract_chunk_index(chunk_id):
        """chunk_id（例：'old_chunk_15'）から数値部分（15）を抽出"""
        import re
        match = re.search(r'_(\d+)$', chunk_id)
        return int(match.group(1)) if match else 0
    
    # 統合されたチャンクを追跡
    integrated_chunks = []
    groups_requiring_strength_recalc = set()  # 強度再計算が必要なグループを追跡
    
    # 旧チャンクの構造的組み入れ
    for old_id in list(remaining_old):
        current_index = extract_chunk_index(old_id)
        
        # 前後のチャンクが所属するグループを特定
        neighbor_groups = []
        
        # 前のチャンクをチェック（複数候補）
        for prev_offset in range(1, 4):  # 最大3つ前まで
            prev_id = f"old_chunk_{current_index - prev_offset}"
            if old_id == "old_chunk_56":
                logging.info(f"  前方向チェック: {prev_id} (offset={prev_offset})")
            
            if prev_id in chunk_to_groups:
                for group_info in chunk_to_groups[prev_id]:
                    neighbor_groups.append(('prev', prev_offset, group_info))
                    if old_id == "old_chunk_56":
                        logging.info(f"    → {prev_id} はグループ{group_info[1]}に所属")
                break  # 最初に見つかったものを採用
            elif old_id == "old_chunk_56":
                logging.info(f"    → {prev_id} はグループに所属していない")
        
        # 後のチャンクをチェック（複数候補）
        for next_offset in range(1, 4):  # 最大3つ後まで
            next_id = f"old_chunk_{current_index + next_offset}"
            if old_id == "old_chunk_56":
                logging.info(f"  後方向チェック: {next_id} (offset={next_offset})")
            
            if next_id in chunk_to_groups:
                for group_info in chunk_to_groups[next_id]:
                    neighbor_groups.append(('next', next_offset, group_info))
                    if old_id == "old_chunk_56":
                        logging.info(f"    → {next_id} はグループ{group_info[1]}に所属")
                break  # 最初に見つかったものを採用
            elif old_id == "old_chunk_56":
                logging.info(f"    → {next_id} はグループに所属していない")
        
        if neighbor_groups:
            if old_id == "old_chunk_56":
                logging.info(f"  ★ 近隣グループ発見: {len(neighbor_groups)}個")
                for i, (direction, offset, (chunk_type, group_idx, target_group)) in enumerate(neighbor_groups):
                    logging.info(f"    グループ{i}: {direction}方向, offset={offset}, グループ{group_idx}")
                    logging.info(f"      旧チャンク: {target_group['old']}")
                    logging.info(f"      新チャンク: {target_group['new']}")
            
            logging.info(f"  近隣グループ発見: {len(neighbor_groups)}個")
            
            # 最も近い（offsetが小さい）グループを優先
            neighbor_groups.sort(key=lambda x: x[1])
            
            # 各グループに対応する新チャンクを探す
            for direction, offset, (chunk_type, group_idx, target_group) in neighbor_groups:
                logging.info(f"  {direction}方向のグループ{group_idx}を検討 (offset={offset})")
                
                # このグループに対応する新チャンクを探す
                candidate_new_chunks = []
                
                # 同じ相対位置の新チャンクを優先的に探す
                if direction == 'prev':
                    # 前方向のグループなら、やや後ろの新チャンクを探す
                    for new_offset in range(0, 3):
                        candidate_new_id = f"new_chunk_{current_index + new_offset}"
                        
                        if candidate_new_id in remaining_new:
                            candidate_new_chunks.append(candidate_new_id)
                else:  # direction == 'next'
                    # 後方向のグループなら、やや前の新チャンクを探す
                    for new_offset in range(0, 3):
                        candidate_new_id = f"new_chunk_{current_index - new_offset}"
                        
                        if candidate_new_id in remaining_new:
                            candidate_new_chunks.append(candidate_new_id)
                
                # 新チャンクが見つかった場合はペアで組み入れ
                if candidate_new_chunks:
                    selected_new_id = candidate_new_chunks[0]  # 最も近いものを選択
                    
                    # 既存グループに直接追加
                    target_group['old'].append(old_id)
                    target_group['new'].append(selected_new_id)
                    
                    # グループタイプを更新
                    old_count = len(target_group['old'])
                    new_count = len(target_group['new'])
                    target_group['type'] = determine_group_type(old_count, new_count)
                    
                    # 組み入れ情報を追加
                    if 'integration_info' not in target_group:
                        target_group['integration_info'] = []
                    
                    target_group['integration_info'].append({
                        'integrated_old': old_id,
                        'integrated_new': selected_new_id,
                        'neighbor_direction': direction,
                        'neighbor_offset': offset,
                        'integration_reason': f'{direction}方向への構造的組み入れ (距離={offset})'
                    })
                    
                    # 強制クラスター化フラグを設定
                    target_group['forced_clustering'] = True
                    if 'refinement_method' not in target_group:
                        target_group['refinement_method'] = 'original'
                    target_group['refinement_method'] += '+structural_integration'
                    
                    # 強度再計算の対象として記録
                    groups_requiring_strength_recalc.add(id(target_group))
                    
                    # 使用済みセットから削除
                    remaining_old.discard(old_id)
                    remaining_new.discard(selected_new_id)
                    
                    integrated_chunks.append((old_id, selected_new_id, group_idx, direction, offset))
                    
                    logging.info(f"  構造的組み入れ成功: {old_id} -> {selected_new_id} ({direction}グループ{group_idx}, 距離={offset})")
                    break  # 一つのグループに組み入れたら終了
                
                # 新チャンクが見つからない場合は旧チャンクのみを組み入れ（削除として扱う）
                else:
                    logging.info(f"  対応新チャンクなし、旧チャンクのみ組み入れ: {old_id}")
                    
                    # 既存グループに旧チャンクのみ追加
                    target_group['old'].append(old_id)
                    
                    # グループタイプを更新
                    old_count = len(target_group['old'])
                    new_count = len(target_group['new'])
                    target_group['type'] = determine_group_type(old_count, new_count)
                    
                    # 組み入れ情報を追加
                    if 'integration_info' not in target_group:
                        target_group['integration_info'] = []
                    
                    target_group['integration_info'].append({
                        'integrated_old': old_id,
                        'integrated_new': None,
                        'neighbor_direction': direction,
                        'neighbor_offset': offset,
                        'integration_reason': f'{direction}方向への削除チャンク組み入れ (距離={offset})',
                        'integration_type': 'deletion_only'
                    })
                    
                    # 強制クラスター化フラグを設定
                    target_group['forced_clustering'] = True
                    if 'refinement_method' not in target_group:
                        target_group['refinement_method'] = 'original'
                    target_group['refinement_method'] += '+structural_integration_deletion'
                    
                    # 強度再計算の対象として記録
                    groups_requiring_strength_recalc.add(id(target_group))
                    
                    # 使用済みセットから削除
                    remaining_old.discard(old_id)
                    
                    integrated_chunks.append((old_id, None, group_idx, direction, offset))
                    
                    logging.info(f"  削除チャンク組み入れ成功: {old_id} ({direction}グループ{group_idx}, 距離={offset})")
                    break  # 一つのグループに組み入れたら終了
        else:
            logging.info(f"  近隣グループが見つかりませんでした")
    
    # 新チャンクの構造的組み入れ（旧チャンクでペアになれなかった場合）
    for new_id in list(remaining_new):
        current_index = extract_chunk_index(new_id)
        
        # 前後のチャンクが所属するグループを特定（段階的に範囲を拡大）
        neighbor_groups = []
        
        # 段階1: 近い範囲（1-3）
        for prev_offset in range(1, 4):
            prev_id = f"new_chunk_{current_index - prev_offset}"
            if prev_id in chunk_to_groups:
                for group_info in chunk_to_groups[prev_id]:
                    neighbor_groups.append(('prev', prev_offset, group_info))
                break
        
        for next_offset in range(1, 4):
            next_id = f"new_chunk_{current_index + next_offset}"
            if next_id in chunk_to_groups:
                for group_info in chunk_to_groups[next_id]:
                    neighbor_groups.append(('next', next_offset, group_info))
                break
        
        # 段階2: 近隣が見つからない場合、より広い範囲で検索（4-10）
        if not neighbor_groups:
            logging.info(f"  近隣グループなし、広域検索開始: {new_id}")
            for prev_offset in range(4, 11):
                prev_id = f"new_chunk_{current_index - prev_offset}"
                if prev_id in chunk_to_groups:
                    for group_info in chunk_to_groups[prev_id]:
                        neighbor_groups.append(('prev_far', prev_offset, group_info))
                    break
            
            for next_offset in range(4, 11):
                next_id = f"new_chunk_{current_index + next_offset}"
                if next_id in chunk_to_groups:
                    for group_info in chunk_to_groups[next_id]:
                        neighbor_groups.append(('next_far', next_offset, group_info))
                    break
        
        # 段階3: それでも見つからない場合、最大のグループに組み入れ
        if not neighbor_groups:
            logging.info(f"  広域でも近隣グループなし、最大グループに組み入れ: {new_id}")
            # 既存グループの中で最も大きなグループを特定
            largest_group = None
            largest_size = 0
            largest_group_idx = -1
            
            for i, group in enumerate(all_groups):
                group_size = len(group['old']) + len(group['new'])
                if group_size > largest_size:
                    largest_size = group_size
                    largest_group = group
                    largest_group_idx = i
            
            if largest_group:
                neighbor_groups.append(('largest', 999, ('mixed', largest_group_idx, largest_group)))
        
        if neighbor_groups:
            logging.info(f"  近隣グループ発見: {len(neighbor_groups)}個")
            
            neighbor_groups.sort(key=lambda x: x[1])  # 距離順
            
            # 最初のグループに新チャンクを追加
            for direction, offset, (chunk_type, group_idx, target_group) in neighbor_groups:
                # 既存グループに直接追加
                target_group['new'].append(new_id)
                
                # グループタイプを更新
                old_count = len(target_group['old'])
                new_count = len(target_group['new'])
                target_group['type'] = determine_group_type(old_count, new_count)
                
                # 組み入れ情報を追加
                if 'integration_info' not in target_group:
                    target_group['integration_info'] = []
                
                target_group['integration_info'].append({
                    'integrated_old': None,
                    'integrated_new': new_id,
                    'neighbor_direction': direction,
                    'neighbor_offset': offset,
                    'integration_reason': f'{direction}方向への新規追加 (距離={offset})',
                    'integration_type': 'new_only'
                })
                
                # 強制クラスター化フラグを設定
                target_group['forced_clustering'] = True
                if 'refinement_method' not in target_group:
                    target_group['refinement_method'] = 'original'
                target_group['refinement_method'] += '+structural_integration_addition'
                
                # 強度再計算の対象として記録
                groups_requiring_strength_recalc.add(id(target_group))
                
                # 使用済みセットから削除
                remaining_new.discard(new_id)
                
                integrated_chunks.append((None, new_id, group_idx, direction, offset))
                
                logging.info(f"  新規追加組み入れ: -> {new_id} ({direction}グループ{group_idx}, 距離={offset})")
                break
        else:
            logging.info(f"  完全孤立チャンク: {new_id} - 全範囲で近隣グループが見つかりません")
    
    # 強度再計算：構造的組み入れによって変更されたグループのみ
    for group_idx, group in enumerate(all_groups):
        if id(group) in groups_requiring_strength_recalc:
            original_strength = group.get('strength', 0.0)
            
            # 類似度辞書から該当する組み合わせのみを抽出
            group_similarities = {}
            for old_id in group['old']:
                for new_id in group['new']:
                    # 元々の類似度がある場合のみ使用（構造的組み入れは実際の類似度なし）
                    if (old_id, new_id) in group.get('similarities', {}):
                        group_similarities[(old_id, new_id)] = group['similarities'][(old_id, new_id)]
            
            if group_similarities:
                # 実際の類似度に基づく強度のみ計算
                old_count = len(group['old'])
                new_count = len(group['new'])
                
                # 指標1: 平均類似度（実際の類似度のみ）
                avg_similarity = sum(group_similarities.values()) / len(group_similarities)
                
                # 指標2: 最大類似度
                max_similarity = max(group_similarities.values())
                
                # 指標3: 結合密度（実際の関係数 / 全チャンク数）
                # 構造的組み入れを考慮して調整
                actual_connections = len(group_similarities)
                possible_connections = old_count * new_count
                adjusted_density = actual_connections / possible_connections if possible_connections > 0 else 0.0
                
                # 指標4: サイズペナルティ（構造的組み入れは強いペナルティ）
                integration_penalty = 1.0 / (1.0 + 0.5 * (old_count + new_count - 2))  # より強いペナルティ
                
                # 構造的組み入れ特別ペナルティ（減算方式）
                integration_count = len(group.get('integration_info', []))
                integration_deduction = min(0.4, 0.05 * integration_count)  # 最大0.4の減算
                
                # 強度計算（理論的最大値1.0を自然に保証）
                base_strength = (
                    0.4 * avg_similarity +
                    0.3 * max_similarity +
                    0.2 * adjusted_density +
                    0.1 * integration_penalty
                )
                
                # ペナルティを減算で適用（自然に1.0以下を保証）
                recalculated_strength = max(0.0, base_strength - integration_deduction)
                
                group['strength'] = recalculated_strength
                
                logging.info(f"強度再計算: グループ{group_idx} {original_strength:.4f} -> {recalculated_strength:.4f} "
                           f"(組み入れ数={integration_count}, 実関係数={actual_connections}, ペナルティ={integration_deduction:.3f})")
            else:
                # 実際の類似度が全くない場合は非常に低い強度
                group['strength'] = 0.05
                logging.info(f"強度再計算: グループ{group_idx} 実類似度なし -> 0.05")
    
    # 統合結果のサマリーをログ出力
    if integrated_chunks:
        logging.info(f"構造的組み入れ完了: {len(integrated_chunks)}個のチャンクを既存グループに統合")
        for old_id, new_id, group_idx, direction, offset in integrated_chunks:
            if old_id:
                logging.info(f"  グループ{group_idx}: {old_id} + {new_id} ({direction}, 距離={offset})")
            else:
                logging.info(f"  グループ{group_idx}: 新規追加 {new_id} ({direction}, 距離={offset})")
    
    # 統合後はグループの作成ではなく空のリストを返す（既存グループが直接変更されるため）
    return []

def explore_residual_clusters(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, min_threshold=0.05, perfect_match_threshold=None):
    """
    残存チャンク間で追加のクラスターを探索（完全一致除外）
    
    複数のアプローチを組み合わせて、残ったチャンク間で自然なクラスターを形成:
    1. 意味的類似性に基づくクラスタリング
    2. 位置的近接性を考慮したクラスタリング
    3. 弱い類似度でのクラスター形成
    """
    if perfect_match_threshold is None:
        perfect_match_threshold = SIMILARITY_THRESHOLDS["bypass"]
    residual_clusters = []
    local_remaining_old = set(remaining_old)
    local_remaining_new = set(remaining_new)
    
    logging.info(f"残存クラスター探索開始: 旧{len(local_remaining_old)}個, 新{len(local_remaining_new)}個")
    
    # アプローチ1: 意味的類似性に基づくクラスタリング
    if len(local_remaining_old) >= 2 and len(local_remaining_new) >= 2:
        semantic_clusters = apply_semantic_residual_clustering(
            local_remaining_old, local_remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, min_threshold, perfect_match_threshold
        )
        
        for cluster in semantic_clusters:
            residual_clusters.append(cluster)
            local_remaining_old -= set(cluster['old'])
            local_remaining_new -= set(cluster['new'])
            logging.info(f"意味的残存クラスター: {cluster['old']} -> {cluster['new']} (強度={cluster['strength']:.3f})")
    
    # アプローチ2: 位置的近接性を考慮したクラスタリング
    if local_remaining_old and local_remaining_new:
        positional_clusters = apply_positional_residual_clustering(
            local_remaining_old, local_remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, min_threshold, perfect_match_threshold
        )
        
        for cluster in positional_clusters:
            residual_clusters.append(cluster)
            local_remaining_old -= set(cluster['old'])
            local_remaining_new -= set(cluster['new'])
            logging.info(f"位置的残存クラスター: {cluster['old']} -> {cluster['new']} (強度={cluster['strength']:.3f})")
    
    # アプローチ3: 弱い類似度でのペア形成
    if local_remaining_old and local_remaining_new:
        weak_similarity_clusters = apply_weak_similarity_clustering(
            local_remaining_old, local_remaining_new, old_chunks, new_chunks,
            old_vectors, new_vectors, min_threshold, perfect_match_threshold
        )
        
        for cluster in weak_similarity_clusters:
            residual_clusters.append(cluster)
            local_remaining_old -= set(cluster['old'])
            local_remaining_new -= set(cluster['new'])
            logging.info(f"弱類似度残存クラスター: {cluster['old']} -> {cluster['new']} (強度={cluster['strength']:.3f})")
    
    # 残存チャンクを元のセットから削除
    for cluster in residual_clusters:
        remaining_old -= set(cluster['old'])
        remaining_new -= set(cluster['new'])
    
    logging.info(f"残存クラスター探索完了: {len(residual_clusters)}個のクラスター形成")
    return residual_clusters

def apply_semantic_residual_clustering(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, min_threshold, perfect_match_threshold):
    """
    意味的類似性に基づく残存クラスタリング（完全一致除外）
    """
    clusters = []
    
    try:
        # チャンクの平均エンベディングを計算
        old_embeddings = []
        old_chunk_info = []
        
        for old_id in remaining_old:
            idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
            if idx is not None and old_vectors[idx]:
                # チャンク全体のベクトルを直接使用
                old_embeddings.append(old_vectors[idx])
                old_chunk_info.append({'id': old_id, 'index': idx})
        
        new_embeddings = []
        new_chunk_info = []
        
        for new_id in remaining_new:
            idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            if idx is not None and new_vectors[idx]:
                # チャンク全体のベクトルを直接使用
                new_embeddings.append(new_vectors[idx])
                new_chunk_info.append({'id': new_id, 'index': idx})
        
        # 意味的類似度マトリックスを計算
        if old_embeddings and new_embeddings:
            similarity_matrix = []
            for old_emb in old_embeddings:
                row = []
                for new_emb in new_embeddings:
                    sim = cosine_similarity(old_emb, new_emb)
                    row.append(sim)
                similarity_matrix.append(row)
            
            # 閾値を超える類似度ペアを抽出（完全一致除外）
            valid_pairs = []
            for i, old_info in enumerate(old_chunk_info):
                for j, new_info in enumerate(new_chunk_info):
                    sim = similarity_matrix[i][j]
                    # 完全一致閾値以上は除外し、最小閾値以上のみを対象とする
                    if min_threshold <= sim < perfect_match_threshold:
                        valid_pairs.append((old_info['id'], new_info['id'], sim))
            
            # 類似度の高い順にソートしてクラスター形成
            valid_pairs.sort(key=lambda x: x[2], reverse=True)
            used_old = set()
            used_new = set()
            
            for old_id, new_id, sim in valid_pairs:
                if old_id not in used_old and new_id not in used_new:
                    cluster = {
                        'old': [old_id],
                        'new': [new_id],
                        'type': '1:1',
                        'strength': sim,
                        'refinement_method': 'semantic_residual',
                        'forced_clustering': True,
                        'similarities': {(old_id, new_id): sim}
                    }
                    clusters.append(cluster)
                    used_old.add(old_id)
                    used_new.add(new_id)
    
    except Exception as e:
        logging.warning(f"意味的残存クラスタリングでエラー: {e}")
    
    return clusters

def apply_positional_residual_clustering(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, min_threshold, perfect_match_threshold):
    """
    位置的近接性を考慮した残存クラスタリング（完全一致除外）
    """
    clusters = []
    
    def extract_chunk_index(chunk_id):
        """chunk_id（例：'old_chunk_15'）から数値部分（15）を抽出"""
        import re
        match = re.search(r'_(\d+)$', chunk_id)
        return int(match.group(1)) if match else 0
    
    # チャンクを位置順にソート
    old_sorted = sorted(remaining_old, key=extract_chunk_index)
    new_sorted = sorted(remaining_new, key=extract_chunk_index)
    
    used_old = set()
    used_new = set()
    
    # 位置的に近いチャンク同士をペアリング
    for old_id in old_sorted:
        if old_id in used_old:
            continue
            
        old_idx_in_doc = extract_chunk_index(old_id)
        old_chunk_idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
        
        if old_chunk_idx is None:
            continue
        
        best_new_id = None
        best_similarity = 0.0
        best_position_score = 0.0
        
        for new_id in new_sorted:
            if new_id in used_new:
                continue
                
            new_idx_in_doc = extract_chunk_index(new_id)
            new_chunk_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            
            if new_chunk_idx is None:
                continue
            
            # 類似度を計算
            similarity = calculate_chunk_similarity(old_vectors[old_chunk_idx], new_vectors[new_chunk_idx])
            
            # 完全一致閾値以上は除外
            if similarity >= perfect_match_threshold:
                continue
            
            # 位置的近接性スコア（距離が近いほど高スコア）
            position_distance = abs(old_idx_in_doc - new_idx_in_doc)
            position_score = 1.0 / (1.0 + 0.1 * position_distance)  # 距離が近いほど高スコア
            
            # 総合スコア（類似度 + 位置的近接性）
            combined_score = 0.7 * similarity + 0.3 * position_score
            
            if similarity >= min_threshold and combined_score > best_similarity + best_position_score:
                best_new_id = new_id
                best_similarity = similarity
                best_position_score = position_score
        
        # 最適なペアが見つかった場合
        if best_new_id and best_similarity >= min_threshold:
            cluster = {
                'old': [old_id],
                'new': [best_new_id],
                'type': '1:1',
                'strength': best_similarity,
                'refinement_method': 'positional_residual',
                'forced_clustering': True,
                'similarities': {(old_id, best_new_id): best_similarity},
                'position_score': best_position_score
            }
            clusters.append(cluster)
            used_old.add(old_id)
            used_new.add(best_new_id)
    
    return clusters

def apply_weak_similarity_clustering(remaining_old, remaining_new, old_chunks, new_chunks, old_vectors, new_vectors, min_threshold, perfect_match_threshold):
    """
    弱い類似度でのクラスタリング（最後の手段、完全一致除外）
    """
    clusters = []
    
    # 全ペア間の類似度を計算
    similarity_pairs = []
    
    for old_id in remaining_old:
        old_idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
        if old_idx is None:
            continue
            
        for new_id in remaining_new:
            new_idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            if new_idx is None:
                continue
                
            similarity = calculate_chunk_similarity(old_vectors[old_idx], new_vectors[new_idx])
            # 完全一致閾値以上は除外し、最小閾値以上のみを対象とする
            if min_threshold <= similarity < perfect_match_threshold:
                similarity_pairs.append((old_id, new_id, similarity))
    
    # 類似度順にソート
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    used_old = set()
    used_new = set()
    
    # 貪欲法でペアを選択
    for old_id, new_id, similarity in similarity_pairs:
        if old_id not in used_old and new_id not in used_new:
            cluster = {
                'old': [old_id],
                'new': [new_id],
                'type': '1:1',
                'strength': similarity,
                'refinement_method': 'weak_similarity_residual',
                'forced_clustering': True,
                'similarities': {(old_id, new_id): similarity}
            }
            clusters.append(cluster)
            used_old.add(old_id)
            used_new.add(new_id)
    
    return clusters

def get_chunk_hierarchy_key(chunk):
    """チャンクから上位階層キーを取得（階層結合用）"""
    hierarchy_levels = extract_hierarchy_from_chunk(chunk)
    
    # レベル1, 2, 3を順次結合してキーを作成
    hierarchy_parts = []
    for level in ['level_1', 'level_2', 'level_3']:
        if level in hierarchy_levels and hierarchy_levels[level]:
            hierarchy_parts.append(hierarchy_levels[level])
    
    # 結合してスペース・改行除去
    if hierarchy_parts:
        combined = ''.join(hierarchy_parts)
        normalized = combined.replace(' ', '').replace('　', '').replace('\n', '').replace('\t', '')
        return normalized
    else:
        return None

def can_merge_by_hierarchy(group1, group2, old_chunks, new_chunks):
    """2つの1:1グループが階層的に結合可能かチェック"""
    # 両方とも1:1グループの場合のみ結合対象
    if (group1.get('type') != '1:1' or group2.get('type') != '1:1' or 
        len(group1['old']) != 1 or len(group1['new']) != 1 or
        len(group2['old']) != 1 or len(group2['new']) != 1):
        return False
    
    # 各グループの旧チャンクの階層キーを取得
    old_chunk1 = next((c for c in old_chunks if c['id'] == group1['old'][0]), None)
    old_chunk2 = next((c for c in old_chunks if c['id'] == group2['old'][0]), None)
    
    if not old_chunk1 or not old_chunk2:
        return False
    
    hierarchy_key1 = get_chunk_hierarchy_key(old_chunk1)
    hierarchy_key2 = get_chunk_hierarchy_key(old_chunk2)
    
    # 階層キーが同じ場合は結合可能
    return (hierarchy_key1 is not None and 
            hierarchy_key2 is not None and 
            hierarchy_key1 == hierarchy_key2)

def extract_chunk_index(chunk_id):
    """chunk_id（例：'old_chunk_15'）から数値部分（15）を抽出"""
    match = re.search(r'_(\d+)$', chunk_id)
    return int(match.group(1)) if match else 0