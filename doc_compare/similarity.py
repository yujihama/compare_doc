import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def calculate_chunk_similarity(old_vectors, new_vectors):
    """チャンク間の最大類似度を計算"""
    max_sim = 0.0
    for v1 in old_vectors:
        for v2 in new_vectors:
            sim = cosine_similarity(v1, v2)
            max_sim = max(max_sim, sim)
    return max_sim

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

def find_similar_pairs(
    old_chunks: List[Dict], new_chunks: List[Dict],
    old_vectors: List[List], new_vectors: List[List],
    threshold: float = 0.8
) -> Tuple[List[Tuple], List[str], List[str]]:
    """
    類似文ペア・チャンクペア・追加/削除チャンクを抽出
    """
    chunk_pairs = []
    old_chunk_ids = [c["id"] for c in old_chunks]
    new_chunk_ids = [c["id"] for c in new_chunks]
    old_to_new = {cid: set() for cid in old_chunk_ids}
    new_to_old = {cid: set() for cid in new_chunk_ids}

    # 各チャンク内の文ベクトル同士で類似度計算
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            found = False
            for vi, v1 in enumerate(old_vectors[i]):
                for vj, v2 in enumerate(new_vectors[j]):
                    sim = cosine_similarity(v1, v2)
                    if sim >= threshold:
                        old_to_new[old_chunk["id"]].add(new_chunk["id"])
                        new_to_old[new_chunk["id"]].add(old_chunk["id"])
                        found = True
            if found:
                chunk_pairs.append((old_chunk["id"], new_chunk["id"]))

    # 追加・削除チャンク
    deleted = [cid for cid, s in old_to_new.items() if not s]
    added = [cid for cid, s in new_to_old.items() if not s]
    return chunk_pairs, added, deleted

def is_large_cluster(cluster, similarities):
    """
    大きなクラスターかどうかを判定
    
    判定基準:
    1. サイズ: 総チャンク数が6個以上
    2. 密度: 実際の関係数/可能な関係数 < 0.6
    3. 類似度分散: 類似度の標準偏差 > 0.1
    4. 類似度範囲: 最大類似度 - 最小類似度 > 0.3
    """
    old_count = len(cluster['old'])
    new_count = len(cluster['new'])
    total_chunks = old_count + new_count
    
    # 基準1: サイズチェック
    if total_chunks < 6:
        return False
    
    # クラスター内の類似度を取得
    cluster_similarities = []
    for old_id in cluster['old']:
        for new_id in cluster['new']:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim > 0:
                cluster_similarities.append(sim)
    
    if len(cluster_similarities) < 3:
        return False
    
    # 基準2: 密度チェック
    possible_connections = old_count * new_count
    actual_connections = len(cluster_similarities)
    density = actual_connections / possible_connections
    
    # 基準3: 類似度分散チェック
    sim_std = np.std(cluster_similarities)
    
    # 基準4: 類似度範囲チェック
    sim_range = max(cluster_similarities) - min(cluster_similarities)
    
    # 複合判定
    is_large = (
        density < 0.6 and          # 密度が低い
        sim_std > 0.1 and          # 分散が大きい
        sim_range > 0.3            # 範囲が大きい
    )
    
    logging.info(f"クラスター分析: サイズ={total_chunks}, 密度={density:.3f}, 分散={sim_std:.3f}, 範囲={sim_range:.3f}, 大きい={is_large}")
    
    return is_large

def hierarchical_cluster_refinement(large_cluster, similarities, threshold_high=0.95, expand_threshold=0.85):
    """
    大きなクラスターを階層的に細分化
    """
    sub_clusters = []
    used_old = set()
    used_new = set()
    
    # 1. 高い類似度のペアを抽出
    strong_pairs = []
    for old_id in large_cluster['old']:
        for new_id in large_cluster['new']:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim >= threshold_high:
                strong_pairs.append(((old_id, new_id), sim))
    
    # 2. 強い類似度順にソート
    strong_pairs.sort(key=lambda x: x[1], reverse=True)
    
    logging.info(f"階層的細分化: {len(strong_pairs)}個の強いペアを検出")
    
    # 3. 各強いペアを核として局所クラスターを拡張
    for (core_old, core_new), core_sim in strong_pairs:
        if core_old in used_old or core_new in used_new:
            continue
            
        # 局所クラスターを形成
        local_cluster = expand_from_core(core_old, core_new, large_cluster, similarities, expand_threshold)
        
        if local_cluster:
            sub_clusters.append(local_cluster)
            
            # 使用済みとしてマーク
            used_old.update(local_cluster['old'])
            used_new.update(local_cluster['new'])
    
    # 4. 未使用のチャンクがあれば残りクラスターとして追加
    remaining_old = [oid for oid in large_cluster['old'] if oid not in used_old]
    remaining_new = [nid for nid in large_cluster['new'] if nid not in used_new]
    
    if remaining_old and remaining_new:
        sub_clusters.append({
            'old': remaining_old,
            'new': remaining_new,
            'type': determine_group_type(len(remaining_old), len(remaining_new)),
            'refinement_method': 'hierarchical_remaining'
        })
    
    return sub_clusters

def expand_from_core(core_old, core_new, large_cluster, similarities, expand_threshold=0.85):
    """
    コアペアから局所クラスターを拡張
    """
    local_old = {core_old}
    local_new = {core_new}
    
    # 段階的に関連チャンクを追加
    changed = True
    iteration = 0
    max_iterations = 5  # 無限ループ防止
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # 旧チャンク側の拡張
        for old_id in large_cluster['old']:
            if old_id in local_old:
                continue
            
            # 既存の新チャンクとの類似度を確認
            max_sim_to_local = max([similarities.get((old_id, new_id), 0.0) 
                                   for new_id in local_new], default=0.0)
            if max_sim_to_local >= expand_threshold:
                local_old.add(old_id)
                changed = True
        
        # 新チャンク側の拡張
        for new_id in large_cluster['new']:
            if new_id in local_new:
                continue
                
            # 既存の旧チャンクとの類似度を確認
            max_sim_to_local = max([similarities.get((old_id, new_id), 0.0) 
                                   for old_id in local_old], default=0.0)
            if max_sim_to_local >= expand_threshold:
                local_new.add(new_id)
                changed = True
    
    if len(local_old) == 0 or len(local_new) == 0:
        return None
    
    return {
        'old': sorted(list(local_old)),
        'new': sorted(list(local_new)),
        'type': determine_group_type(len(local_old), len(local_new)),
        'core_similarity': similarities.get((core_old, core_new), 0.0),
        'refinement_method': 'hierarchical_core'
    }

def semantic_refinement(large_cluster, old_chunks, new_chunks, old_vecs, new_vecs):
    """
    エンベディングの意味的類似性に基づく細分化
    """
    try:
        # 1. クラスター内のチャンクのエンベディングを取得
        cluster_embeddings = []
        chunk_info = []
        
        for old_id in large_cluster['old']:
            idx = next((i for i, c in enumerate(old_chunks) if c['id'] == old_id), None)
            if idx is not None and old_vecs[idx]:
                avg_embedding = np.mean(old_vecs[idx], axis=0)
                cluster_embeddings.append(avg_embedding)
                chunk_info.append({'id': old_id, 'type': 'old', 'index': idx})
        
        for new_id in large_cluster['new']:
            idx = next((i for i, c in enumerate(new_chunks) if c['id'] == new_id), None)
            if idx is not None and new_vecs[idx]:
                avg_embedding = np.mean(new_vecs[idx], axis=0)
                cluster_embeddings.append(avg_embedding)
                chunk_info.append({'id': new_id, 'type': 'new', 'index': idx})
        
        # 2. K-means クラスタリング
        if len(cluster_embeddings) >= 4:  # 最小限の数がある場合
            n_clusters = min(3, max(2, len(cluster_embeddings) // 3))  # 適応的にクラスター数決定
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(cluster_embeddings)
            
            # 3. クラスターごとに分割
            sub_clusters = []
            for cluster_id in range(n_clusters):
                cluster_chunks = [chunk_info[i] for i, label in enumerate(labels) if label == cluster_id]
                
                old_in_cluster = [c['id'] for c in cluster_chunks if c['type'] == 'old']
                new_in_cluster = [c['id'] for c in cluster_chunks if c['type'] == 'new']
                
                if old_in_cluster and new_in_cluster:
                    sub_clusters.append({
                        'old': old_in_cluster,
                        'new': new_in_cluster,
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
    最適な細分化結果を選択
    """
    scores = {}
    
    for method, sub_clusters in refinement_results.items():
        if not sub_clusters or len(sub_clusters) <= 1:
            scores[method] = 0.0
            continue
            
        # 評価指標：内部結合度と外部分離度
        internal_cohesion = calculate_internal_cohesion(sub_clusters, similarities)
        external_separation = calculate_external_separation(sub_clusters, similarities)
        
        # 細分化数のペナルティ（過度な細分化を防ぐ）
        size_penalty = 1.0 / (1.0 + 0.1 * (len(sub_clusters) - 2))
        
        scores[method] = 0.5 * internal_cohesion + 0.3 * external_separation + 0.2 * size_penalty
        
        logging.info(f"細分化評価 {method}: 内部結合={internal_cohesion:.3f}, 外部分離={external_separation:.3f}, スコア={scores[method]:.3f}")
    
    if not scores or max(scores.values()) < 0.3:
        logging.info("細分化スコアが低いため、元のクラスターを維持")
        return [original_cluster]
    
    best_method = max(scores, key=scores.get)
    logging.info(f"最適な細分化方法: {best_method}")
    return refinement_results[best_method]

def refine_large_clusters(clusters, similarities, old_chunks, new_chunks, old_vecs, new_vecs, refinement_mode="auto"):
    """
    大きなクラスターを細分化
    """
    refined_clusters = []
    
    for cluster in clusters:
        if is_large_cluster(cluster, similarities):
            logging.info(f"大きなクラスターを検出: 旧{len(cluster['old'])}個, 新{len(cluster['new'])}個")
            
            if refinement_mode == "none":
                refined_clusters.append(cluster)
                continue
                
            refinement_results = {}
            
            # 階層的細分化
            if refinement_mode in ["auto", "hierarchical"]:
                refinement_results['hierarchical'] = hierarchical_cluster_refinement(cluster, similarities)
            
            # 意味的細分化
            if refinement_mode in ["auto", "semantic"]:
                refinement_results['semantic'] = semantic_refinement(cluster, old_chunks, new_chunks, old_vecs, new_vecs)
            
            # 最適な細分化を選択
            if refinement_mode == "auto":
                best_refinement = select_best_refinement(refinement_results, cluster, similarities)
            else:
                best_refinement = refinement_results.get(refinement_mode, [cluster])
            
            refined_clusters.extend(best_refinement)
        else:
            refined_clusters.append(cluster)
    
    return refined_clusters

def find_similar_groups(
    old_chunks: List[Dict], new_chunks: List[Dict],
    old_vectors: List[List], new_vectors: List[List],
    threshold: float = 0.8,
    refinement_mode: str = "auto"
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    強く結合されたチャンククラスターを検出し、適切なグループ化を行う
    refinement_mode: "auto", "hierarchical", "semantic", "none"
    """
    old_chunk_ids = [c["id"] for c in old_chunks]
    new_chunk_ids = [c["id"] for c in new_chunks]
    old_to_new = {cid: set() for cid in old_chunk_ids}
    new_to_old = {cid: set() for cid in new_chunk_ids}
    
    # チャンク間類似度を計算・記録
    chunk_similarities = {}
    
    for i, old_chunk in enumerate(old_chunks):
        for j, new_chunk in enumerate(new_chunks):
            max_sim = calculate_chunk_similarity(old_vectors[i], new_vectors[j])
            if max_sim >= threshold:
                old_to_new[old_chunk["id"]].add(new_chunk["id"])
                new_to_old[new_chunk["id"]].add(old_chunk["id"])
                chunk_similarities[(old_chunk["id"], new_chunk["id"])] = max_sim

    # 強く結合されたクラスターを検出
    clusters = find_connected_clusters(old_to_new, new_to_old, chunk_similarities)
    
    # クラスター強度順にソート
    clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    # 大きなクラスターを細分化
    if refinement_mode != "none":
        clusters = refine_large_clusters(clusters, chunk_similarities, old_chunks, new_chunks, old_vectors, new_vectors, refinement_mode)
        # 細分化後に再度強度順にソート
        for cluster in clusters:
            if 'strength' not in cluster:
                cluster['strength'] = calculate_cluster_strength(cluster['old'], cluster['new'], 
                                                               {k: v for k, v in chunk_similarities.items() 
                                                                if k[0] in cluster['old'] and k[1] in cluster['new']})
        clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    # デバッグログ出力
    with open("similarity_debug.log", "w", encoding="utf-8") as f:
        f.write(f"閾値: {threshold}\n")
        f.write(f"総類似度計算数: {len(chunk_similarities)}\n")
        f.write(f"閾値超え: {len(chunk_similarities)}\n")
        f.write(f"細分化モード: {refinement_mode}\n")
        f.write("計算方式: クラスターベース最大類似度\n\n")
        
        f.write("=== 検出されたクラスター ===\n")
        for i, cluster in enumerate(clusters, 1):
            group_type = cluster.get('type', determine_group_type(len(cluster['old']), len(cluster['new'])))
            refinement_method = cluster.get('refinement_method', 'original')
            f.write(f"クラスター{i} ({group_type}): 強度={cluster['strength']:.4f}, 細分化={refinement_method}\n")
            f.write(f"  旧: {cluster['old']}\n")
            f.write(f"  新: {cluster['new']}\n")
            if 'similarities' in cluster:
                f.write(f"  類似度詳細: {cluster['similarities']}\n")
            f.write("\n")
        
        f.write("=== 個別マッピング詳細 ===\n")
        for old_id, new_ids in old_to_new.items():
            if new_ids:
                f.write(f"{old_id} -> {sorted(list(new_ids))} (1:{len(new_ids)})\n")
    
    # クラスターをグループに変換
    groups = []
    used_old = set()
    used_new = set()
    
    for cluster in clusters:
        old_ids = cluster['old']
        new_ids = cluster['new']
        
        # 既に使用されたチャンクを除外
        available_old = [oid for oid in old_ids if oid not in used_old]
        available_new = [nid for nid in new_ids if nid not in used_new]
        
        if available_old and available_new:
            group_type = determine_group_type(len(available_old), len(available_new))
            
            groups.append({
                "old": available_old,
                "new": available_new,
                "type": group_type,
                "strength": cluster['strength'],
                "similarities": {k: v for k, v in cluster['similarities'].items() 
                               if k[0] in available_old and k[1] in available_new}
            })
            
            used_old.update(available_old)
            used_new.update(available_new)
            
            logging.info(f"{group_type}マッチ: {available_old} -> {available_new} (強度: {cluster['strength']:.4f})")

    # 完全孤立チャンク（どことも関連しない、または使用されなかった）
    deleted = [cid for cid in old_chunk_ids if cid not in used_old]
    added = [cid for cid in new_chunk_ids if cid not in used_new]
    
    logging.info(f"処理結果: グループ{len(groups)}個, 削除{len(deleted)}個, 追加{len(added)}個")
    
    return groups, added, deleted 