import os
import datetime
import json
import csv
from typing import Optional, Dict, Any
from .structured_models import ComparisonReport
from .config import EXPORT_CONFIG


def create_output_directory(output_dir: str = "output") -> str:
    """出力ディレクトリを作成"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def generate_filename(prefix: str = "comparison_report", extension: str = "md") -> str:
    """タイムスタンプ付きのファイル名を生成"""
    timestamp = datetime.datetime.now().strftime(EXPORT_CONFIG["timestamp_format"])
    return f"{prefix}_{timestamp}.{extension}"


def _format_chunk_content(content: str) -> str:
    """チャンク内容の共通フォーマット処理"""
    max_length = EXPORT_CONFIG["max_content_length"]
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content


def _format_heading(heading: str) -> str:
    """見出しの共通フォーマット処理"""
    if not heading or heading == "見出しなし":
        return "見出しなし"
    
    max_length = EXPORT_CONFIG["max_heading_length"]
    if len(heading) > max_length:
        return heading[:max_length] + "..."
    return heading


def format_chunk_info(chunk, chunk_type: str) -> str:
    """チャンク情報をMarkdown形式でフォーマット"""
    formatted_heading = _format_heading(chunk.heading)
    formatted_content = _format_chunk_content(chunk.content)
    
    return f"""
### {chunk_type}チャンク: {formatted_heading}

**ID:** {chunk.id}

**内容:**
```
{formatted_content}
```
"""


def format_group_analysis(group) -> str:
    """グループ分析結果をMarkdown形式でフォーマット"""
    old_chunks_text = "\n".join([f"- {_format_heading(chunk.heading)} ({chunk.id})" for chunk in group.old_chunks])
    new_chunks_text = "\n".join([f"- {_format_heading(chunk.heading)} ({chunk.id})" for chunk in group.new_chunks])
    
    correspondence_info = ""
    if group.correspondence:
        correspondence_info = f"""
**対応関係の種類:** {group.correspondence.correspondence_type}

**対応の詳細:**
- 旧チャンク: {', '.join(group.correspondence.old_chunk_ids)}
- 新チャンク: {', '.join(group.correspondence.new_chunk_ids)}
"""
    
    return f"""
## グループ {group.group_number}: {group.group_type}

**変更の種類:** {group.analysis.change_type}  
**類似度強度:** {group.strength:.4f}  
**処理方法:** {group.processing_method}  
**細分化手法:** {group.refinement_method}

### 分析サマリー
{group.analysis.summary}

### 詳細分析
{group.analysis.detailed_analysis}

### 主な変更点
{chr(10).join([f"- {change}" for change in group.analysis.main_changes])}

### 対応チャンク
**旧チャンク:**
{old_chunks_text}

**新チャンク:**
{new_chunks_text}
{correspondence_info}

---
"""


def _calculate_statistics(report: ComparisonReport) -> Dict[str, Dict[str, int]]:
    """統計情報を計算する共通関数"""
    # グループタイプ別の統計
    group_type_counts = {}
    change_type_counts = {}
    processing_method_counts = {}
    
    for group in report.groups:
        group_type_counts[group.group_type] = group_type_counts.get(group.group_type, 0) + 1
        change_type_counts[group.analysis.change_type] = change_type_counts.get(group.analysis.change_type, 0) + 1
        processing_method_counts[group.processing_method] = processing_method_counts.get(group.processing_method, 0) + 1
    
    return {
        "group_types": group_type_counts,
        "change_types": change_type_counts,
        "processing_methods": processing_method_counts
    }


def export_to_markdown(structured_report: ComparisonReport, 
                      output_dir: str = "output", 
                      filename: Optional[str] = None) -> str:
    """構造化レポートをMarkdownファイルとして出力"""
    
    # 出力ディレクトリの作成
    output_path = create_output_directory(output_dir)
    
    # ファイル名の決定
    if filename is None:
        filename = generate_filename("comparison_report", "md")
    
    full_path = os.path.join(output_path, filename)
    
    # 統計情報を計算
    statistics = _calculate_statistics(structured_report)
    
    # Markdownコンテンツの生成
    markdown_content = f"""# ドキュメント比較レポート

**生成日時:** {datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## サマリー

- **処理グループ数:** {structured_report.summary.get('processed_groups', 0)}
- **追加チャンク数:** {structured_report.summary.get('added_chunks', 0)}
- **削除チャンク数:** {structured_report.summary.get('deleted_chunks', 0)}
- **バイパスグループ数:** {structured_report.summary.get('bypassed_groups', 0)}
- **GPT分析グループ数:** {structured_report.summary.get('gpt_analyzed_groups', 0)}

## 統計情報

### グループ別統計
"""
    
    # 統計情報を追加
    markdown_content += "\n**グループタイプ別:**\n"
    for group_type, count in statistics["group_types"].items():
        markdown_content += f"- {group_type}: {count}個\n"
    
    markdown_content += "\n**変更タイプ別:**\n"
    for change_type, count in statistics["change_types"].items():
        markdown_content += f"- {change_type}: {count}個\n"
    
    # グループ分析結果の追加
    markdown_content += "\n# グループ分析結果\n"
    
    for group in structured_report.groups:
        markdown_content += format_group_analysis(group)
    
    # 追加・削除チャンクの情報
    if structured_report.added_chunks:
        markdown_content += "\n# 追加されたチャンク\n"
        for chunk in structured_report.added_chunks:
            markdown_content += format_chunk_info(chunk, "追加")
    
    if structured_report.deleted_chunks:
        markdown_content += "\n# 削除されたチャンク\n"
        for chunk in structured_report.deleted_chunks:
            markdown_content += format_chunk_info(chunk, "削除")
    
    # ファイルへの書き込み
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return full_path


def export_summary_to_markdown(structured_report: ComparisonReport, 
                              output_dir: str = "output") -> str:
    """サマリー版のMarkdownレポートを出力"""
    
    output_path = create_output_directory(output_dir)
    filename = generate_filename("comparison_summary", "md")
    full_path = os.path.join(output_path, filename)
    
    # サマリーMarkdownコンテンツの生成
    markdown_content = f"""# ドキュメント比較サマリー

**生成日時:** {datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 概要

- **処理グループ数:** {structured_report.summary.get('processed_groups', 0)}
- **追加チャンク数:** {structured_report.summary.get('added_chunks', 0)}
- **削除チャンク数:** {structured_report.summary.get('deleted_chunks', 0)}

## 主要な変更点

"""
    
    # 重要な変更のみをピックアップ
    important_groups = []
    for group in structured_report.groups:
        if group.analysis.change_type in ["変更", "追加", "削除"] and group.strength < 0.9:
            important_groups.append(group)
    
    for group in important_groups[:10]:  # 上位10件のみ
        heading = _format_heading(group.old_chunks[0].heading if group.old_chunks else group.new_chunks[0].heading)
        markdown_content += f"""
### {group.analysis.change_type}: {heading}

**変更概要:** {group.analysis.summary}

**主な変更点:**
{chr(10).join([f"- {change}" for change in group.analysis.main_changes[:3]])}

---
"""
    
    # 追加・削除チャンクのサマリー
    if structured_report.added_chunks:
        markdown_content += "\n## 追加されたセクション\n"
        for chunk in structured_report.added_chunks[:5]:  # 上位5件のみ
            markdown_content += f"- {_format_heading(chunk.heading)}\n"
    
    if structured_report.deleted_chunks:
        markdown_content += "\n## 削除されたセクション\n"
        for chunk in structured_report.deleted_chunks[:5]:  # 上位5件のみ
            markdown_content += f"- {_format_heading(chunk.heading)}\n"
    
    # ファイルへの書き込み
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return full_path


def export_to_json(structured_report: ComparisonReport, 
                  output_dir: str = "output", 
                  filename: Optional[str] = None) -> str:
    """構造化レポートをJSONファイルとして出力"""
    
    output_path = create_output_directory(output_dir)
    
    if filename is None:
        filename = generate_filename("comparison_report", "json")
    
    full_path = os.path.join(output_path, filename)
    
    # JSONシリアライズ可能な形式に変換
    def serialize_chunk(chunk):
        return {
            "id": chunk.id,
            "heading": _format_heading(chunk.heading),
            "content": _format_chunk_content(chunk.content)
        }
    
    def serialize_correspondence(correspondence):
        if correspondence is None:
            return None
        return {
            "old_chunk_ids": correspondence.old_chunk_ids,
            "new_chunk_ids": correspondence.new_chunk_ids,
            "correspondence_type": correspondence.correspondence_type
        }
    
    def serialize_analysis(analysis):
        return {
            "change_type": analysis.change_type,
            "summary": analysis.summary[:EXPORT_CONFIG["max_summary_length"]],
            "detailed_analysis": analysis.detailed_analysis,
            "main_changes": analysis.main_changes,
            "correspondence_details": analysis.correspondence_details
        }
    
    def serialize_group(group):
        return {
            "group_number": group.group_number,
            "group_type": group.group_type,
            "strength": group.strength,
            "refinement_method": group.refinement_method,
            "processing_method": group.processing_method,
            "old_chunks": [serialize_chunk(chunk) for chunk in group.old_chunks],
            "new_chunks": [serialize_chunk(chunk) for chunk in group.new_chunks],
            "analysis": serialize_analysis(group.analysis),
            "correspondence": serialize_correspondence(group.correspondence)
        }
    
    json_data = {
        "generation_time": datetime.datetime.now().isoformat(),
        "summary": structured_report.summary,
        "groups": [serialize_group(group) for group in structured_report.groups],
        "added_chunks": [serialize_chunk(chunk) for chunk in structured_report.added_chunks],
        "deleted_chunks": [serialize_chunk(chunk) for chunk in structured_report.deleted_chunks]
    }
    
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return full_path


def export_to_csv(structured_report: ComparisonReport, 
                 output_dir: str = "output") -> str:
    """構造化レポートをCSVファイルとして出力（グループサマリー）"""
    
    output_path = create_output_directory(output_dir)
    filename = generate_filename("comparison_summary", "csv")
    full_path = os.path.join(output_path, filename)
    
    with open(full_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "グループ番号", "グループタイプ", "変更種類", "類似度強度", 
            "処理方法", "細分化手法", "旧チャンク数", "新チャンク数",
            "旧チャンク見出し", "新チャンク見出し", "変更サマリー"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for group in structured_report.groups:
            old_headings = " | ".join([_format_heading(chunk.heading) for chunk in group.old_chunks])
            new_headings = " | ".join([_format_heading(chunk.heading) for chunk in group.new_chunks])
            
            writer.writerow({
                "グループ番号": group.group_number,
                "グループタイプ": group.group_type,
                "変更種類": group.analysis.change_type,
                "類似度強度": round(group.strength, 4),
                "処理方法": group.processing_method,
                "細分化手法": group.refinement_method,
                "旧チャンク数": len(group.old_chunks),
                "新チャンク数": len(group.new_chunks),
                "旧チャンク見出し": old_headings,
                "新チャンク見出し": new_headings,
                "変更サマリー": group.analysis.summary[:EXPORT_CONFIG["max_summary_length"]]
            })
    
    return full_path


def export_statistics_to_csv(structured_report: ComparisonReport, 
                           output_dir: str = "output") -> str:
    """統計情報をCSVファイルとして出力"""
    
    output_path = create_output_directory(output_dir)
    filename = generate_filename("comparison_stats", "csv")
    full_path = os.path.join(output_path, filename)
    
    # 統計データを収集
    statistics = _calculate_statistics(structured_report)
    
    with open(full_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー
        writer.writerow(["統計カテゴリ", "項目", "件数"])
        
        # サマリー統計
        writer.writerow(["サマリー", "処理グループ数", structured_report.summary.get('processed_groups', 0)])
        writer.writerow(["サマリー", "追加チャンク数", structured_report.summary.get('added_chunks', 0)])
        writer.writerow(["サマリー", "削除チャンク数", structured_report.summary.get('deleted_chunks', 0)])
        writer.writerow(["サマリー", "バイパスグループ数", structured_report.summary.get('bypassed_groups', 0)])
        writer.writerow(["サマリー", "GPT分析グループ数", structured_report.summary.get('gpt_analyzed_groups', 0)])
        
        # グループタイプ別統計
        for group_type, count in statistics["group_types"].items():
            writer.writerow(["グループタイプ", group_type, count])
        
        # 変更タイプ別統計
        for change_type, count in statistics["change_types"].items():
            writer.writerow(["変更タイプ", change_type, count])
        
        # 処理方法別統計
        for processing_method, count in statistics["processing_methods"].items():
            writer.writerow(["処理方法", processing_method, count])
    
    return full_path


def export_all_formats(structured_report: ComparisonReport, 
                      output_dir: str = "output") -> Dict[str, str]:
    """すべての形式でエクスポート"""
    
    results = {}
    
    # Markdown（詳細版）
    results["markdown_full"] = export_to_markdown(structured_report, output_dir)
    
    # Markdown（サマリー版）
    results["markdown_summary"] = export_summary_to_markdown(structured_report, output_dir)
    
    # JSON
    results["json"] = export_to_json(structured_report, output_dir)
    
    # CSV（グループ情報）
    results["csv_groups"] = export_to_csv(structured_report, output_dir)
    
    # CSV（統計情報）
    results["csv_stats"] = export_statistics_to_csv(structured_report, output_dir)
    
    return results 