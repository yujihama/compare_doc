import logging
import os
from typing import List, Dict, Tuple
from datetime import datetime
from .similarity import find_similar_groups, get_chunk_hierarchy_key, can_merge_by_hierarchy
from .structured_gpt_analysis import analyze_chunk_group_structured
from .structured_report import create_structured_comparison_report
from .structured_models import AnalysisResult
from .config import SIMILARITY_THRESHOLDS

def process_document_comparison(
    old_chunks: List[Dict], 
    new_chunks: List[Dict], 
    old_vecs: List[List[float]], 
    new_vecs: List[List[float]], 
    threshold: float = None,
    max_groups: int = 50,
    refinement_mode: str = "auto",
    bypass_threshold: float = None,
    force_clustering: bool = True,
    initial_clustering_mode: str = "strict",
    structural_integration: bool = True,
    perfect_match_threshold: float = None,
    use_hierarchy_constraints: bool = True
) -> Tuple[object, Dict]:
    """
    文書比較のメイン処理ロジック（構造化出力）
    
    Args:
        perfect_match_threshold: 完全一致とみなす閾値（この値以上は完全一致として除外）
        use_hierarchy_constraints: 階層制約を使用するかどうか
    """
    # デフォルト値を設定ファイルから取得
    if threshold is None:
        threshold = SIMILARITY_THRESHOLDS["default"]
    if bypass_threshold is None:
        bypass_threshold = SIMILARITY_THRESHOLDS["bypass"]
    
    logging.info(f"強く結合されたチャンククラスター抽出（細分化モード: {refinement_mode}, 強制クラスター化: {force_clustering}, 初期形成: {initial_clustering_mode}, 構造的組み入れ: {structural_integration}, 階層制約: {use_hierarchy_constraints}）")
    groups, added, deleted = find_similar_groups(
        old_chunks, new_chunks, old_vecs, new_vecs, 
        threshold=threshold, refinement_mode=refinement_mode,
        force_clustering=force_clustering,
        initial_clustering_mode=initial_clustering_mode,
        structural_integration=structural_integration,
        perfect_match_threshold=perfect_match_threshold,
        use_hierarchy_constraints=use_hierarchy_constraints
    )
    logging.info(f"グループ: {len(groups)} 追加: {len(added)} 削除: {len(deleted)}")
    
    # グループタイプ別の統計
    group_type_stats = {}
    for g in groups:
        group_type = g.get("type", "unknown")
        if group_type not in group_type_stats:
            group_type_stats[group_type] = 0
        group_type_stats[group_type] += 1
    
    logging.info(f"グループタイプ詳細統計: {group_type_stats}")
    
    group_1_to_1 = group_type_stats.get("1:1", 0)
    group_1_to_n = group_type_stats.get("1:N", 0)
    group_n_to_1 = group_type_stats.get("N:1", 0)
    group_n_to_n = group_type_stats.get("N:N", 0)
    
    # 細分化関連統計
    refinement_stats = {
        'hierarchical_core': sum(1 for g in groups if g.get("refinement_method") == "hierarchical_core"),
        'hierarchical_remaining': sum(1 for g in groups if g.get("refinement_method") == "hierarchical_remaining"),
        'semantic': sum(1 for g in groups if g.get("refinement_method") == "semantic"),
        'original': sum(1 for g in groups if g.get("refinement_method") == "original"),
    }
    total_refined = sum(refinement_stats.values()) - refinement_stats['original']
    
    # バイパス処理：高強度クラスターの検出
    bypassed_groups = []
    analysis_groups = []
    
    for group in groups:
        strength = group.get("strength", 0.0)
        is_forced = group.get("forced_clustering", False)
        is_structural = '+structural_integration' in group.get("refinement_method", "")
        
        # 内容類似度を取得（バイパス判定用）
        content_similarities = group.get("content_similarities", {})
        if content_similarities:
            # 内容類似度の最大値を取得
            max_content_similarity = max(content_similarities.values()) if content_similarities else 0.0
        else:
            # 階層制約未使用の場合は通常の類似度を使用
            similarities = group.get("similarities", {})
            max_content_similarity = max(similarities.values()) if similarities else 0.0
        
        # 構造的組み入れや強制クラスター化されたグループはバイパス禁止
        if is_forced or is_structural:
            analysis_groups.append(group)
            logging.info(f"強制分析対象 (強制={is_forced}, 構造的={is_structural}, 最終強度={strength:.4f}, 内容類似度={max_content_similarity:.4f}): {group['old']} -> {group['new']}")
        elif max_content_similarity < bypass_threshold:
            # 内容類似度がバイパス閾値未満なのでLLM分析対象
            analysis_groups.append(group)
        else:
            # 内容類似度がバイパス閾値以上なのでバイパス対象
            bypassed_groups.append(group)
            logging.debug(f"バイパス処理 (内容類似度={max_content_similarity:.4f} >= {bypass_threshold}, 最終強度={strength:.4f}): {group['old']} -> {group['new']}")
    
    logging.info(f"グループタイプ内訳: 1:1={group_1_to_1}, 1:N={group_1_to_n}, N:1={group_n_to_1}, N:N={group_n_to_n}")
    logging.info(f"細分化統計: 階層コア={refinement_stats['hierarchical_core']}, 階層残り={refinement_stats['hierarchical_remaining']}, 意味的={refinement_stats['semantic']}, 元のまま={refinement_stats['original']}")
    logging.info(f"バイパス統計: {len(bypassed_groups)}個のクラスターをバイパス（内容類似度閾値: {bypass_threshold}）")
    
    # LLM分析対象の1:1グループを階層ごとに結合
    if analysis_groups:
        logging.debug("LLM分析対象の1:1グループの階層結合を開始")
        analysis_groups = merge_analysis_groups_by_hierarchy(analysis_groups, old_chunks, new_chunks)
        logging.debug(f"階層結合後のLLM分析グループ数: {len(analysis_groups)}個")
    
    # バイパス対象とLLM分析対象の詳細をファイルに出力（LLM分析実行前）
    save_llm_analysis_details(bypassed_groups, analysis_groups, old_chunks, new_chunks)
    
    # GPT分析処理（構造化出力版）
    gpt_analyzed_groups = []
    
    # GPT分析可能な最大数を計算（バイパス済みを除く）
    remaining_analysis_capacity = max(0, max_groups - len(bypassed_groups))
    analysis_target_groups = analysis_groups[:remaining_analysis_capacity] if remaining_analysis_capacity > 0 else analysis_groups[:max_groups]
    
    logging.info(f"構造化GPT分析対象: {len(analysis_target_groups)}個（バイパス: {len(bypassed_groups)}個, 上限: {max_groups}個）")
    
    for i, group_meta_for_gpt in enumerate(analysis_target_groups):
        logging.info(f"構造化GPT分析 {i+1}/{len(analysis_target_groups)} (強度={group_meta_for_gpt.get('strength', 0.0):.4f})")
        
        try:
            old_content_for_gpt = []
            old_ids_for_gpt = []
            for old_id in group_meta_for_gpt["old"]:
                chunk = next((c for c in old_chunks if c["id"] == old_id), None)
                if chunk:
                    content = chunk.get("text", chunk.get("content", ""))
                    old_content_for_gpt.append(content)
                    old_ids_for_gpt.append(old_id)
            
            new_content_for_gpt = []
            new_ids_for_gpt = []
            for new_id in group_meta_for_gpt["new"]:
                chunk = next((c for c in new_chunks if c["id"] == new_id), None)
                if chunk:
                    content = chunk.get("text", chunk.get("content", ""))
                    new_content_for_gpt.append(content)
                    new_ids_for_gpt.append(new_id)
            
            # グループ情報を再構築して構造化GPT分析に渡す
            group_for_analysis = {
                "old": old_ids_for_gpt,
                "new": new_ids_for_gpt,
                "type": group_meta_for_gpt.get("type", "1:1"),
                "similarities": group_meta_for_gpt.get("similarities", {})
            }
            
            # 構造化分析を実行
            try:
                structured_analysis_gpt = analyze_chunk_group_structured(
                    old_chunks, new_chunks, group_for_analysis
                )
                
                # 分析結果の妥当性確認
                if structured_analysis_gpt is None:
                    logging.warning(f"グループ {i+1}: 構造化分析が None を返しました")
                    structured_analysis_gpt = AnalysisResult(
                        change_type="変更",
                        summary="分析結果が取得できませんでした",
                        detailed_analysis="構造化分析処理でNoneが返されました",
                        main_changes=["分析エラー"],
                        correspondence_details="分析エラーのため対応関係を特定できませんでした"
                    )
                
                logging.info(f"グループ {i+1} 分析完了: 変更タイプ={structured_analysis_gpt.change_type}")
                
            except Exception as analysis_error:
                logging.error(f"グループ {i+1} の構造化分析でエラー: {analysis_error}")
                logging.error(f"エラー詳細 - 旧チャンク: {old_ids_for_gpt}, 新チャンク: {new_ids_for_gpt}")
                
                # フォールバック分析結果を生成
                structured_analysis_gpt = AnalysisResult(
                    change_type="変更",
                    summary=f"分析エラー: {str(analysis_error)[:100]}",
                    detailed_analysis=f"構造化分析中にエラーが発生しました。エラー内容: {str(analysis_error)}",
                    main_changes=[f"分析エラー: {type(analysis_error).__name__}"],
                    correspondence_details="エラーのため対応関係を特定できませんでした"
                )
            
            group_result = {
                "group_number": 0,
                "type": group_meta_for_gpt.get("type", "1:1"),
                "old_chunks": old_ids_for_gpt,
                "new_chunks": new_ids_for_gpt,
                "strength": group_meta_for_gpt.get("strength", 0.0),
                "refinement_method": group_meta_for_gpt.get("refinement_method", "original"),
                "structured_analysis": structured_analysis_gpt,
                "processing_method": "gpt_analyzed",
                "old_hierarchy": get_chunk_hierarchy_info(old_ids_for_gpt[0], old_chunks) if old_ids_for_gpt else None,
                "new_hierarchies": [get_chunk_hierarchy_info(new_id, new_chunks) for new_id in new_ids_for_gpt if get_chunk_hierarchy_info(new_id, new_chunks)]
            }
            gpt_analyzed_groups.append(group_result)
            
        except Exception as e:
            logging.error(f"グループ {i+1} の処理全体でエラー: {e}")
            logging.error(f"グループ詳細: {group_meta_for_gpt}")
            # エラーがあっても処理を継続
            continue
    
    processed_groups_list = []
    # バイパスされたグループを処理
    for group_meta_bypassed in bypassed_groups:
        # create_bypass_group_result を呼び出して、structured_analysis を持つ整形済み辞書を取得
        processed_bypassed_group = create_bypass_group_result(group_meta_bypassed, old_chunks, new_chunks)
        processed_groups_list.append(processed_bypassed_group)
    
    # GPT分析されたグループを追加
    processed_groups_list.extend(gpt_analyzed_groups)
    
    # 強度順に再ソート
    processed_groups_list.sort(key=lambda x: x["strength"], reverse=True)
    
    # グループ番号を再振り
    for i, group_item in enumerate(processed_groups_list, 1):
        group_item["group_number"] = i
    
    # 構造化レポート生成
    structured_report = create_structured_comparison_report(
        processed_groups_list, added, deleted, old_chunks, new_chunks
    )
    
    # 統計情報
    stats = {
        "processed_groups": len(processed_groups_list),
        "total_groups": len(groups),
        "bypassed_groups": len(bypassed_groups),
        "gpt_analyzed_groups": len(gpt_analyzed_groups),
        "added_chunks": len(added),
        "deleted_chunks": len(deleted),
        "group_1_to_1": group_1_to_1,
        "group_1_to_n": group_1_to_n,
        "group_n_to_1": group_n_to_1,
        "group_n_to_n": group_n_to_n,
        "refinement_mode": refinement_mode,
        "total_refined": total_refined,
        "refinement_stats": refinement_stats,
        "bypass_threshold": bypass_threshold,
        "processed_groups_detail": processed_groups_list,
        "structured_report": structured_report
    }
    
    return structured_report, stats

def create_bypass_group_result(group: Dict, old_chunks: List[Dict], new_chunks: List[Dict]) -> Dict:
    """
    バイパス処理用のグループ結果を作成（構造化版）
    """
    old_ids = []
    new_ids = []
    
    for old_id in group["old"]:
        chunk = next((c for c in old_chunks if c["id"] == old_id), None)
        if chunk:
            old_ids.append(old_id)
    
    for new_id in group["new"]:
        chunk = next((c for c in new_chunks if c["id"] == new_id), None)
        if chunk:
            new_ids.append(new_id)
    
    # 構造化された分析結果
    structured_analysis = AnalysisResult(
        change_type="変更なし",
        summary=f"高い類似度により変更なしと判定",
        detailed_analysis=f"バイパス処理により、設定された閾値を上回る類似度が検出されたため、詳細な分析をスキップして「変更なし」と判定しました。強度: {group.get('strength', 0.0):.4f}",
        main_changes=["実質的な変更なし"],
        correspondence_details="自動バイパス処理により、内容は維持されていると判定"
    )
    
    return {
        "group_number": 0,  # 後で再振り
        "type": group.get("type", "1:1"),
        "old_chunks": old_ids,
        "new_chunks": new_ids,
        "strength": group.get("strength", 0.0),
        "refinement_method": group.get("refinement_method", "original"),
        "structured_analysis": structured_analysis,
        "processing_method": "bypassed",
        "old_hierarchy": get_chunk_hierarchy_info(old_ids[0], old_chunks) if old_ids else None,
        "new_hierarchies": [get_chunk_hierarchy_info(new_id, new_chunks) for new_id in new_ids if get_chunk_hierarchy_info(new_id, new_chunks)]
    }

def get_chunk_hierarchy_info(chunk_id: str, chunks: List[Dict]) -> str:
    """チャンクの階層情報を取得（ダミー：常にNoneを返す）"""
    return None 

def merge_analysis_groups_by_hierarchy(groups: List[Dict], old_chunks: List[Dict], new_chunks: List[Dict]) -> List[Dict]:
    """
    LLM分析対象の1:1グループを階層ごとに結合
    """
    # 階層キーごとにグループを分類
    hierarchy_combinations = {}
    non_mergeable_groups = []
    
    for group in groups:
        # 1:1グループで階層情報があるもののみ結合対象
        if (group.get('type') == '1:1' and 
            len(group.get('old', [])) == 1 and 
            len(group.get('new', [])) == 1):
            
            old_chunk_id = group['old'][0]
            new_chunk_id = group['new'][0]
            old_chunk = next((c for c in old_chunks if c['id'] == old_chunk_id), None)
            new_chunk = next((c for c in new_chunks if c['id'] == new_chunk_id), None)
            
            if old_chunk and new_chunk:
                old_hierarchy_key = get_chunk_hierarchy_key(old_chunk)
                new_hierarchy_key = get_chunk_hierarchy_key(new_chunk)
                
                # 新旧の階層組み合わせをキーとして使用
                if old_hierarchy_key and new_hierarchy_key:
                    # 新旧階層の組み合わせパターンをキーとする
                    hierarchy_combination_key = f"{old_hierarchy_key} -> {new_hierarchy_key}"
                    
                    hierarchy_combinations.setdefault(hierarchy_combination_key, []).append(group)
                else:
                    # 階層情報がない場合は結合しない
                    non_mergeable_groups.append(group)
            else:
                non_mergeable_groups.append(group)
        else:
            # 1:1以外のグループは結合しない
            non_mergeable_groups.append(group)
        
    # 階層ごとにグループを結合
    merged_groups = []
    
    for hierarchy_key, groups_in_hierarchy in hierarchy_combinations.items():
        if len(groups_in_hierarchy) == 1:
            # 1つしかない場合はそのまま
            merged_groups.append(groups_in_hierarchy[0])
        else:
            # 複数ある場合は結合
            merged_group = merge_multiple_analysis_groups(groups_in_hierarchy, hierarchy_key)
            merged_groups.append(merged_group)
    
    # 結合不可能なグループと結合されたグループをマージ
    final_groups = non_mergeable_groups + merged_groups
    
    logging.debug(f"LLM分析グループ階層結合結果: 元{len(groups)}個 -> 最終{len(final_groups)}個 (結合可能{len(merged_groups)}個, 結合不可{len(non_mergeable_groups)}個)")
    
    return final_groups

def merge_multiple_analysis_groups(groups: List[Dict], hierarchy_key: str) -> Dict:
    """
    同じ階層の複数のLLM分析対象グループを1つに結合
    """
    # 結合されたチャンクリストを作成
    merged_old_chunks = []
    merged_new_chunks = []
    merged_strength_sum = 0.0
    refinement_methods = set()
    merged_similarities = {}
    merged_content_similarities = {}
    
    for group in groups:
        merged_old_chunks.extend(group.get('old', []))
        merged_new_chunks.extend(group.get('new', []))
        merged_strength_sum += group.get('strength', 0.0)
        
        refinement_method = group.get('refinement_method', 'original')
        refinement_methods.add(refinement_method)
        
        # 類似度情報をマージ
        group_similarities = group.get('similarities', {})
        merged_similarities.update(group_similarities)
        
        # 内容類似度情報をマージ
        group_content_similarities = group.get('content_similarities', {})
        merged_content_similarities.update(group_content_similarities)
    
    # 平均強度を計算
    average_strength = merged_strength_sum / len(groups) if groups else 0.0
    
    # 結合されたグループタイプを決定
    old_count = len(merged_old_chunks)
    new_count = len(merged_new_chunks)
    
    if old_count == 1 and new_count == 1:
        group_type = "1:1"
    elif old_count == 1 and new_count > 1:
        group_type = "1:N"
    elif old_count > 1 and new_count == 1:
        group_type = "N:1"
    else:
        group_type = "N:N"
    
    return {
        'old': merged_old_chunks,
        'new': merged_new_chunks,
        'type': group_type,
        'strength': average_strength,
        'refinement_method': "+".join(sorted(refinement_methods)) + "+hierarchy_merged",
        'similarities': merged_similarities,
        'content_similarities': merged_content_similarities,  # バイパス判定用を保持
        'hierarchy_key': hierarchy_key,
        'merged_group_count': len(groups),
        'forced_clustering': any(group.get('forced_clustering', False) for group in groups)
    } 

def save_llm_analysis_details(bypassed_groups: List[Dict], analysis_groups: List[Dict], old_chunks: List[Dict], new_chunks: List[Dict]):
    """
    LLM分析対象とバイパス対象の詳細情報をファイルに出力
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # LLM分析詳細ログの出力先をlogディレクトリ配下に変更
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log')
    os.makedirs(log_dir, exist_ok=True)
    llm_log_path = os.path.join(log_dir, 'llm_analysis_details.log')
    
    with open(llm_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== LLM分析対象グループ詳細 ===\n")
        f.write(f"生成時刻: {current_time}\n\n")
        
        # バイパスグループの詳細
        f.write(f"=== バイパスグループ ({len(bypassed_groups)}個) ===\n")
        f.write("※これらのグループはLLM分析をスキップ（高い類似度により変更なしと判定）\n\n")
        
        for i, group in enumerate(bypassed_groups, 1):
            f.write(f"バイパスグループ {i}:\n")
            f.write(f"  処理方法: {group.get('processing_method', 'unknown')}\n")
            f.write(f"  グループタイプ: {group.get('type', 'unknown')}\n")
            f.write(f"  強度: {group.get('strength', 0.0):.4f}\n")
            f.write(f"  細分化方法: {group.get('refinement_method', 'unknown')}\n")
            
            # 階層情報がある場合
            if 'hierarchy_key' in group:
                f.write(f"  階層キー: {group['hierarchy_key']}\n")
                f.write(f"  結合グループ数: {group.get('merged_group_count', 1)}\n")
            
            f.write(f"  旧チャンク ({len(group.get('old_chunks', []))}個):\n")
            for old_chunk_id in group.get('old_chunks', []):
                chunk = next((c for c in old_chunks if c['id'] == old_chunk_id), None)
                if chunk:
                    content_preview = chunk.get('text', chunk.get('content', ''))[:100].replace('\n', ' ')
                    f.write(f"    - {old_chunk_id}: {content_preview}...\n")
                else:
                    f.write(f"    - {old_chunk_id}: [チャンク見つからず]\n")
            
            f.write(f"  新チャンク ({len(group.get('new_chunks', []))}個):\n")
            for new_chunk_id in group.get('new_chunks', []):
                chunk = next((c for c in new_chunks if c['id'] == new_chunk_id), None)
                if chunk:
                    content_preview = chunk.get('text', chunk.get('content', ''))[:100].replace('\n', ' ')
                    f.write(f"    - {new_chunk_id}: {content_preview}...\n")
                else:
                    f.write(f"    - {new_chunk_id}: [チャンク見つからず]\n")
            f.write("\n")
        
        # LLM分析対象グループの詳細
        f.write(f"=== LLM分析対象グループ ({len(analysis_groups)}個) ===\n")
        f.write("※これらのグループは実際にLLM分析を実行予定\n\n")
        
        for i, group in enumerate(analysis_groups, 1):
            f.write(f"LLM分析グループ {i}:\n")
            f.write(f"  グループタイプ: {group.get('type', 'unknown')}\n")
            f.write(f"  強度: {group.get('strength', 0.0):.4f}\n")
            f.write(f"  細分化方法: {group.get('refinement_method', 'unknown')}\n")
            
            # 階層情報がある場合
            if 'hierarchy_key' in group:
                f.write(f"  階層キー: {group['hierarchy_key']}\n")
                f.write(f"  結合グループ数: {group.get('merged_group_count', 1)}\n")
            
            # 類似度情報
            content_similarities = group.get('content_similarities', {})
            if content_similarities:
                max_content_similarity = max(content_similarities.values())
                f.write(f"  最大内容類似度: {max_content_similarity:.4f}\n")
            
            f.write(f"  旧チャンク ({len(group.get('old', []))}個):\n")
            for old_chunk_id in group.get('old', []):
                chunk = next((c for c in old_chunks if c['id'] == old_chunk_id), None)
                if chunk:
                    content_preview = chunk.get('text', chunk.get('content', ''))[:100].replace('\n', ' ')
                    f.write(f"    - {old_chunk_id}: {content_preview}...\n")
                else:
                    f.write(f"    - {old_chunk_id}: [チャンク見つからず]\n")
            
            f.write(f"  新チャンク ({len(group.get('new', []))}個):\n")
            for new_chunk_id in group.get('new', []):
                chunk = next((c for c in new_chunks if c['id'] == new_chunk_id), None)
                if chunk:
                    content_preview = chunk.get('text', chunk.get('content', ''))[:100].replace('\n', ' ')
                    f.write(f"    - {new_chunk_id}: {content_preview}...\n")
                else:
                    f.write(f"    - {new_chunk_id}: [チャンク見つからず]\n")
            f.write("\n")
        
        # 統計サマリー
        f.write(f"=== 統計サマリー ===\n")
        f.write(f"総グループ数: {len(bypassed_groups) + len(analysis_groups)}\n")
        f.write(f"バイパス数: {len(bypassed_groups)} ({len(bypassed_groups)/(len(bypassed_groups) + len(analysis_groups))*100:.1f}%)\n")
        f.write(f"LLM分析数: {len(analysis_groups)} ({len(analysis_groups)/(len(bypassed_groups) + len(analysis_groups))*100:.1f}%)\n")
        
        # 階層結合の効果
        hierarchy_merged_bypass = sum(1 for g in bypassed_groups if 'hierarchy_merged' in g.get('refinement_method', ''))
        hierarchy_merged_llm = sum(1 for g in analysis_groups if 'hierarchy_merged' in g.get('refinement_method', ''))
        
        if hierarchy_merged_bypass > 0 or hierarchy_merged_llm > 0:
            f.write(f"\n=== 階層結合効果 ===\n")
            f.write(f"階層結合されたバイパスグループ: {hierarchy_merged_bypass}個\n")
            f.write(f"階層結合されたLLM分析グループ: {hierarchy_merged_llm}個\n")
            
            # 結合による節約効果を計算
            total_merged_groups = sum(g.get('merged_group_count', 0) for g in bypassed_groups if 'merged_group_count' in g)
            if total_merged_groups > 0:
                saved_groups = total_merged_groups - hierarchy_merged_bypass
                f.write(f"結合により削減されたグループ数: {saved_groups}個\n")
        
        f.write(f"\n生成完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logging.info(f"LLM分析詳細をファイルに出力しました: {llm_log_path}") 