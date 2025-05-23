import logging
from typing import List, Dict, Tuple
from .similarity import find_similar_groups
from .gpt_analysis import analyze_chunk_group
from .report import compile_markdown_report

def process_document_comparison(
    old_chunks: List[Dict], 
    new_chunks: List[Dict], 
    old_vecs: List[List], 
    new_vecs: List[List], 
    threshold: float,
    max_groups: int,
    refinement_mode: str = "auto",
    bypass_threshold: float = 0.98
) -> Tuple[str, Dict]:
    """
    文書比較のメイン処理ロジック（クラスターベース + 細分化 + バイパス）
    bypass_threshold: この強度以上のクラスターは「変更なし」として自動処理
    """
    logging.info(f"強く結合されたチャンククラスター抽出（細分化モード: {refinement_mode}）")
    groups, added, deleted = find_similar_groups(
        old_chunks, new_chunks, old_vecs, new_vecs, 
        threshold=threshold, refinement_mode=refinement_mode
    )
    logging.info(f"グループ: {len(groups)} 追加: {len(added)} 削除: {len(deleted)}")
    
    # グループタイプ別の統計
    group_1_to_1 = sum(1 for g in groups if g.get("type") == "1:1")
    group_1_to_n = sum(1 for g in groups if g.get("type") == "1:N")
    group_n_to_1 = sum(1 for g in groups if g.get("type") == "N:1")
    group_n_to_n = sum(1 for g in groups if g.get("type") == "N:N")
    
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
        if strength >= bypass_threshold:
            # バイパス：変更なしとして処理
            bypassed_groups.append(create_bypass_group_result(group, old_chunks, new_chunks))
            logging.info(f"バイパス処理 (強度={strength:.4f}): {group['old']} -> {group['new']}")
        else:
            analysis_groups.append(group)
    
    logging.info(f"グループタイプ内訳: 1:1={group_1_to_1}, 1:N={group_1_to_n}, N:1={group_n_to_1}, N:N={group_n_to_n}")
    logging.info(f"細分化統計: 階層コア={refinement_stats['hierarchical_core']}, 階層残り={refinement_stats['hierarchical_remaining']}, 意味的={refinement_stats['semantic']}, 元のまま={refinement_stats['original']}")
    logging.info(f"バイパス統計: {len(bypassed_groups)}個のクラスターをバイパス（閾値: {bypass_threshold}）")
    
    # GPT分析処理（バイパス以外のグループのみ、制限つき）
    gpt_analyzed_groups = []
    
    # GPT分析可能な最大数を計算（バイパス済みを除く）
    remaining_analysis_capacity = max(0, max_groups - len(bypassed_groups))
    analysis_target_groups = analysis_groups[:remaining_analysis_capacity] if remaining_analysis_capacity > 0 else analysis_groups[:max_groups]
    
    logging.info(f"GPT分析対象: {len(analysis_target_groups)}個（バイパス: {len(bypassed_groups)}個, 上限: {max_groups}個）")
    
    for i, group in enumerate(analysis_target_groups):
        logging.info(f"GPT分析 {i+1}/{len(analysis_target_groups)} (強度={group.get('strength', 0.0):.4f})")
        
        try:
            old_content = []
            old_ids = []
            for old_id in group["old"]:
                chunk = next((c for c in old_chunks if c["id"] == old_id), None)
                if chunk:
                    # textフィールドを優先し、なければcontentフィールドを使用
                    content = chunk.get("text", chunk.get("content", ""))
                    old_content.append(content)
                    old_ids.append(old_id)
            
            new_content = []
            new_ids = []
            for new_id in group["new"]:
                chunk = next((c for c in new_chunks if c["id"] == new_id), None)
                if chunk:
                    # textフィールドを優先し、なければcontentフィールドを使用
                    content = chunk.get("text", chunk.get("content", ""))
                    new_content.append(content)
                    new_ids.append(new_id)
            
            # グループ情報を再構築してGPT分析に渡す
            group_for_analysis = {
                "old": old_ids,
                "new": new_ids,
                "type": group.get("type", "1:1"),
                "similarities": group.get("similarities", {})
            }
            
            analysis_result = analyze_chunk_group(
                old_chunks, new_chunks, group_for_analysis
            )
            
            group_result = {
                "group_number": len(bypassed_groups) + i + 1,
                "type": group.get("type", "1:1"),
                "old_chunks": old_ids,
                "new_chunks": new_ids,
                "strength": group.get("strength", 0.0),
                "refinement_method": group.get("refinement_method", "original"),
                "analysis": analysis_result,
                "processing_method": "gpt_analyzed"
            }
            gpt_analyzed_groups.append(group_result)
            
        except Exception as e:
            logging.error(f"グループ {i+1} の分析でエラー: {e}")
            # エラーがあっても処理を継続
            continue
    
    # 全処理済みグループをマージ（バイパス + GPT分析）
    processed_groups = bypassed_groups + gpt_analyzed_groups
    
    # 強度順に再ソート
    processed_groups.sort(key=lambda x: x["strength"], reverse=True)
    
    # グループ番号を再振り
    for i, group in enumerate(processed_groups, 1):
        group["group_number"] = i
    
    # レポート生成
    report = compile_markdown_report(processed_groups, added, deleted)
    
    # 統計情報
    stats = {
        "processed_groups": len(processed_groups),
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
        "processed_groups_detail": processed_groups  # フィルタ機能用に詳細情報を追加
    }
    
    return report, stats

def create_bypass_group_result(group: Dict, old_chunks: List[Dict], new_chunks: List[Dict]) -> Dict:
    """
    バイパス処理用のグループ結果を作成（変更なしとして扱う）
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
    
    # 統一された分析結果フォーマット
    analysis_result = f"""## 変更の種類
**変更なし**

## 変更前
対応チャンク: {', '.join(old_ids)}

## 変更後
対応チャンク: {', '.join(new_ids)}

## 変更内容の概要
非常に高い類似度（強度={group.get('strength', 0.0):.4f}）により、実質的な変更はないと判定されました。内容は維持されています。

**判定理由**: 自動バイパス処理により、設定された閾値を上回る類似度が検出されたため、詳細なAI分析をスキップして「変更なし」と判定しました。"""
    
    return {
        "group_number": 0,  # 後で再振り
        "type": group.get("type", "1:1"),
        "old_chunks": old_ids,
        "new_chunks": new_ids,
        "strength": group.get("strength", 0.0),
        "refinement_method": group.get("refinement_method", "original"),
        "analysis": analysis_result,
        "processing_method": "bypassed"
    } 