from typing import List, Dict

def compile_markdown_report(processed_groups: List[Dict], added_chunks: List[str], deleted_chunks: List[str]) -> str:
    """
    処理済みグループ、追加チャンク、削除チャンクから完全なマークダウンレポートを生成
    """
    md = ["# ドキュメント比較結果\n"]
    
    # サマリー
    md.append("## 概要")
    md.append(f"- **処理済みグループ**: {len(processed_groups)}個")
    md.append(f"- **追加チャンク**: {len(added_chunks)}個")
    md.append(f"- **削除チャンク**: {len(deleted_chunks)}個")
    
    # 処理方法別の統計
    bypassed_count = sum(1 for g in processed_groups if g.get("processing_method") == "bypassed")
    gpt_analyzed_count = sum(1 for g in processed_groups if g.get("processing_method") == "gpt_analyzed")
    
    if bypassed_count > 0:
        md.append(f"- **バイパス処理**: {bypassed_count}個（変更なし自動判定）")
        md.append(f"- **GPT詳細分析**: {gpt_analyzed_count}個")
    
    md.append("")
    
    # 各グループの詳細分析
    if processed_groups:
        md.append("## 詳細分析")
        md.append("")
        
        for i, group in enumerate(processed_groups):
            group_num = group.get("group_number", 0)
            group_type = group.get("type", "unknown")
            strength = group.get("strength", 0.0)
            old_chunks = group.get("old_chunks", [])
            new_chunks = group.get("new_chunks", [])
            processing_method = group.get("processing_method", "unknown")
            refinement_method = group.get("refinement_method", "original")
            
            # グループヘッダー
            method_text = "自動判定" if processing_method == "bypassed" else "AI分析"
            
            # クラスター番号と基本情報を小さな見出しで
            md.append(f"#### クラスター {group_num}")
            md.append(f"**比較タイプ**: {group_type} | **強度**: {strength:.4f} | **処理**: {method_text} | **細分化**: {refinement_method}")
            md.append(f"**対象**: {', '.join(old_chunks)} → {', '.join(new_chunks)}")
            md.append("")
            
            # 分析結果
            analysis = group.get("analysis", "分析結果なし")
            md.append(analysis)
            md.append("")
            
            # クラスター間の明確な区切り
            if i < len(processed_groups) - 1:  # 最後のグループでない場合
                md.append("---")
                md.append("")
    
    # 追加チャンク
    if added_chunks:
        md.append("---")
        md.append("")
        md.append("## 追加チャンク")
        md.append("")
        for chunk_id in added_chunks:
            md.append(f"- **{chunk_id}**: 新しく追加されたチャンク")
        md.append("")
    
    # 削除チャンク
    if deleted_chunks:
        md.append("---")
        md.append("")
        md.append("## 削除チャンク")
        md.append("")
        for chunk_id in deleted_chunks:
            md.append(f"- **{chunk_id}**: 削除されたチャンク")
        md.append("")
    
    return "\n".join(md) 