from typing import List, Dict
from .structured_models import GroupAnalysisResult, ChunkInfo, ComparisonReport, AnalysisResult, CorrespondenceInfo

def create_structured_comparison_report(
    processed_groups: List[Dict], 
    added_chunks: List[str], 
    deleted_chunks: List[str],
    old_chunks: List[Dict],
    new_chunks: List[Dict]
) -> ComparisonReport:
    """
    処理済みグループから構造化された比較レポートを作成
    """
    
    # 統計情報の準備
    bypassed_count = sum(1 for g in processed_groups if g.get("processing_method") == "bypassed")
    gpt_analyzed_count = sum(1 for g in processed_groups if g.get("processing_method") == "gpt_analyzed")
    
    summary = {
        "processed_groups": len(processed_groups),
        "bypassed_groups": bypassed_count,
        "gpt_analyzed_groups": gpt_analyzed_count,
        "added_chunks": len(added_chunks),
        "deleted_chunks": len(deleted_chunks)
    }
    
    # チャンク情報の作成
    def get_chunk_info(chunk_id: str, chunks: List[Dict]) -> ChunkInfo:
        for chunk in chunks:
            if chunk["id"] == chunk_id:
                return ChunkInfo(
                    id=chunk_id,
                    content=chunk.get("text", chunk.get("content", "")),
                    heading=chunk.get("heading", chunk.get("title"))
                )
        return ChunkInfo(id=chunk_id, content="チャンクが見つかりません")
    
    # 追加・削除チャンクの情報作成
    added_chunk_infos = [get_chunk_info(chunk_id, new_chunks) for chunk_id in added_chunks]
    deleted_chunk_infos = [get_chunk_info(chunk_id, old_chunks) for chunk_id in deleted_chunks]
    
    # グループ分析結果の構造化
    structured_groups = []
    for group in processed_groups:
        # 旧チャンクと新チャンクの情報を取得
        old_chunk_infos = [get_chunk_info(chunk_id, old_chunks) for chunk_id in group.get("old_chunks", [])]
        new_chunk_infos = [get_chunk_info(chunk_id, new_chunks) for chunk_id in group.get("new_chunks", [])]
        
        # 分析結果の構造化（既存のanalysisフィールドから抽出、またはstructured_analysisがあればそれを使用）
        if "structured_analysis" in group and isinstance(group["structured_analysis"], AnalysisResult):
            analysis = group["structured_analysis"]
        else:
            # 既存のtext-basedな分析結果から情報を抽出
            analysis = extract_analysis_from_text(group.get("analysis", ""))
        
        # 対応関係情報の作成（CorrespondenceInfoクラスのインスタンスとして）
        correspondence = CorrespondenceInfo(
            old_chunk_ids=group.get("old_chunks", []),
            new_chunk_ids=group.get("new_chunks", []),
            correspondence_type=group.get("type", "unknown")
        )
        
        structured_group = GroupAnalysisResult(
            group_number=group.get("group_number", 0),
            group_type=group.get("type", "unknown"),
            old_chunks=old_chunk_infos,
            new_chunks=new_chunk_infos,
            strength=group.get("strength", 0.0),
            refinement_method=group.get("refinement_method", "original"),
            processing_method=group.get("processing_method", "unknown"),
            analysis=analysis,
            correspondence=correspondence,
            old_hierarchy=group.get("old_hierarchy"),
            new_hierarchies=group.get("new_hierarchies", [])
        )
        structured_groups.append(structured_group)
    
    return ComparisonReport(
        summary=summary,
        groups=structured_groups,
        added_chunks=added_chunk_infos,
        deleted_chunks=deleted_chunk_infos
    )

def extract_analysis_from_text(analysis_text: str) -> AnalysisResult:
    """
    既存のテキストベースの分析結果から構造化された情報を抽出
    """
    lines = analysis_text.split('\n')
    
    # 変更の種類を抽出（文字列として直接指定）
    change_type = "変更"  # デフォルト
    for i, line in enumerate(lines):
        if line.strip() == "## 変更の種類":
            if i + 1 < len(lines):
                type_line = lines[i + 1].strip().replace("**", "")
                if "変更なし" in type_line:
                    change_type = "変更なし"
                elif "追加" in type_line:
                    change_type = "追加"
                elif "削除" in type_line:
                    change_type = "削除"
                elif "変更" in type_line:
                    change_type = "変更"
                break
    
    # 概要の抽出
    summary = ""
    detailed_analysis = analysis_text
    
    for i, line in enumerate(lines):
        if "## 変更内容の概要" in line:
            # 次のセクションまでの内容を取得
            summary_lines = []
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("##") or lines[j].startswith("#"):
                    break
                summary_lines.append(lines[j])
            summary = "\n".join(summary_lines).strip()[:200]  # 200文字制限
            break
    
    if not summary:
        # 簡単な概要を作成
        summary = analysis_text[:200].replace("\n", " ").strip()
    
    # 主な変更点を抽出（簡易版）
    main_changes = []
    if "主な変更点" in analysis_text:
        # 主な変更点セクションを探す
        for i, line in enumerate(lines):
            if "主な変更点" in line:
                for j in range(i + 1, min(i + 6, len(lines))):  # 最大5行
                    if lines[j].strip().startswith("-") or lines[j].strip().startswith("*"):
                        main_changes.append(lines[j].strip().lstrip("-*").strip())
                break
    
    if not main_changes:
        main_changes = ["詳細は分析結果を参照"]
    
    # 対応関係の詳細（簡易版）
    correspondence_details = "チャンク間の対応関係が分析されています" if "対応関係" in analysis_text else "対応関係の詳細は分析結果を参照"
    
    return AnalysisResult(
        change_type=change_type,
        summary=summary,
        detailed_analysis=detailed_analysis,
        main_changes=main_changes,
        correspondence_details=correspondence_details
    ) 