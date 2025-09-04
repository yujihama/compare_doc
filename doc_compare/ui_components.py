import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from typing import List, Literal, Dict, Optional
from .structured_models import GroupAnalysisResult, ChunkInfo, ComparisonReport
from .config import UI_COLORS

def get_chart_colors(categories: List[str], chart_type: str = "change_types") -> List[str]:
    """カテゴリリストに対応する色のリストを返す"""
    if chart_type == "processing_types":
        # 処理方法別の色設定
        color_map = UI_COLORS["processing_types"]
        return [color_map.get(category, "#d1d5db") for category in categories]
    elif chart_type == "basic_stats":
        # 基本統計の色設定
        basic_color_map = {
            "処理グループ": "#059669",
            "追加チャンク": "#60a5fa", 
            "削除チャンク": "#f87171"
        }
        return [basic_color_map.get(category, "#d1d5db") for category in categories]
    else:
        # 変更タイプ別の色設定（デフォルト）
        return [UI_COLORS["change_types"].get(category, UI_COLORS["change_types"]["その他"]) for category in categories]

def _render_section_header(title: str, color: str, icon: str = "") -> str:
    """セクションヘッダーの共通生成"""
    return f"""
    <div style="
        background-color: {color};
        border-left: 4px solid {color};
        padding: 0.8rem 1rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: bold;
        color: white;
        font-size: 1.3rem;
        border-radius: 5px;
    ">
        {icon} {title}
    </div>
    """

def _render_chunk_header(title: str, color: str, icon: str = "📊") -> str:
    """チャンクヘッダーの共通生成"""
    return f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-radius: 5px;
    ">
        {icon} {title}
    </div>
    """

def _render_info_box(title: str, color: str) -> str:
    """情報ボックスの共通生成"""
    return f"""
    <div style="
        background-color: {color}20;
        border-left: 4px solid {color};
        padding: 0.5rem 1rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: bold;
        color: {color};
    ">
        {title}
    </div>
    """

@st.cache_data(show_spinner=False)
def create_pie_chart(data: Dict[str, int], title: str, chart_type: str = "change_types") -> go.Figure:
    """統一された円グラフを作成する共通関数"""
    # 0以外の値のみを表示
    filtered_data = {k: v for k, v in data.items() if v > 0}
    
    if not filtered_data:
        return None
    
    # 円グラフを作成
    fig = px.pie(
        values=list(filtered_data.values()),
        names=list(filtered_data.keys()),
        title=title
    )
    
    # 色を明示的に設定
    colors = get_chart_colors(list(filtered_data.keys()), chart_type)
    fig.update_traces(marker=dict(colors=colors))
    
    # レイアウト設定（スタイリッシュなデザイン）
    fig.update_layout(
        showlegend=True,
        height=350,
        title_font_size=16,
        title_x=0.3,
        title_font_family="Arial, sans-serif",
        title_font_color="#374151",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11, color="#374151")
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # テキスト表示設定（読みやすく）
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        textfont_size=10,
        textfont_color="white",
        textfont_family="Arial, sans-serif",
        hole=0.3,  # ドーナツチャートに変更（より現代的）
        marker=dict(
            line=dict(color="white", width=2)  # セクション間に白い境界線
        )
    )
    
    return fig

def render_chunk_info(chunk: ChunkInfo, label: str):
    """チャンク情報を整理して表示"""
    # チャンクIDと見出し（色付きラベル）
    if chunk.heading:
        st.markdown(f' `{chunk.id}` - {chunk.heading}', unsafe_allow_html=True)
    else:
        st.markdown(f' `{chunk.id}`', unsafe_allow_html=True)
    
    # 内容をコードブロック形式で表示（改行を適切に処理）
    content = chunk.content
    # 長文は初期表示をトリミングし、全文はエクスパンダで表示
    max_chars = 800
    if isinstance(content, str) and len(content) > max_chars:
        st.code(content[:max_chars] + "\n...", language=None)
        with st.expander("全文を表示"):
            st.code(content, language=None)
    else:
        st.code(content, language=None)

def _render_pagination_controls(total_items: int, key_prefix: str, default_per_page: int = 10) -> tuple[int, int]:
    """ページネーションのUI（1ページ件数とページ移動）を表示し、(page, per_page)を返す"""
    if total_items <= 0:
        return 1, default_per_page
    per_page = st.selectbox(
        "1ページあたりの表示件数",
        options=[5, 10, 20, 50, 100],
        index=[5, 10, 20, 50, 100].index(default_per_page) if default_per_page in [5, 10, 20, 50, 100] else 1,
        key=f"{key_prefix}_per_page"
    )
    total_pages = max(1, math.ceil(total_items / per_page))
    current_page = st.session_state.get(f"{key_prefix}_page", 1)
    cols = st.columns([1, 1, 3, 2])
    with cols[0]:
        if st.button("前へ", key=f"{key_prefix}_prev", use_container_width=True) and current_page > 1:
            current_page -= 1
    with cols[1]:
        if st.button("次へ", key=f"{key_prefix}_next", use_container_width=True) and current_page < total_pages:
            current_page += 1
    with cols[2]:
        current_page = st.number_input(
            "ページ番号",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            key=f"{key_prefix}_page_input"
        )
    with cols[3]:
        st.caption(f"全{total_items}件 / {total_pages}ページ")
    st.session_state[f"{key_prefix}_page"] = int(current_page)
    return int(current_page), int(per_page)

def render_change_type_badge(change_type: str) -> str:
    """変更の種類に応じたシンプルなバッジを生成"""
    emoji_map = {
        "変更": "🔄",
        "追加": "➕",
        "削除": "➖",
        "変更なし": "✅"
    }
    
    emoji = emoji_map.get(change_type, "❓")
    return f"{emoji} **{change_type}**"

def render_group_analysis(group: GroupAnalysisResult):
    """個別のグループ分析結果を表示"""
    
    # グループの種類を判定
    is_forced = hasattr(group, 'forced_clustering') or 'forced' in group.refinement_method.lower()
    is_structural = 'structural_integration' in group.refinement_method.lower()
    
    # ヘッダーの色とタイトルを種類に応じて変更
    if is_structural:
        header_color = "#004a55" 
        header_title = f"クラスター {group.group_number} "
        header_icon = ""
    elif is_forced:
        header_color = "#004a55" 
        header_title = f"クラスター {group.group_number} "
        header_icon = ""
    else:
        header_color = "#004a55" 
        header_title = f"クラスター {group.group_number}"
        header_icon = ""
    
    # 共通ヘッダー生成を使用
    header_html = _render_chunk_header(header_title, header_color, header_icon)
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 構造的組み入れの詳細情報を表示
    if is_structural and hasattr(group, 'integration_info'):
        integration_info = group.integration_info
        with st.expander(f"🔗 構造的組み入れ詳細", expanded=False):
            reason = integration_info.get('integration_reason', '不明')
            direction = integration_info.get('neighbor_direction', '不明')
            offset = integration_info.get('neighbor_offset', '不明')
            group_id = integration_info.get('neighbor_group_id', '不明')
            
            st.write(f"**組み入れ理由**: {reason}")
            
            # 完全孤立の場合は特別な表示
            if direction == 'isolated':
                st.write(f"**組み入れタイプ**: 🏝️ 完全孤立チャンク（個別処理）")
                st.write(f"**状況**: 前後10チャンク範囲内に既存クラスターなし")
            elif direction.endswith('_far'):
                st.write(f"**組み入れタイプ**: 🔍 広域検索による組み入れ")
                st.write(f"**近隣方向**: {direction.replace('_far', '')} (遠距離)")
                st.write(f"**距離**: {offset}チャンク")
                st.write(f"**参照グループ**: グループ{group_id}")
            elif direction == 'largest':
                st.write(f"**組み入れタイプ**: 📊 最大グループへの強制組み入れ")
                st.write(f"**参照グループ**: グループ{group_id} (最大サイズグループ)")
            else:
                st.write(f"**近隣方向**: {direction}")
                st.write(f"**距離**: {offset}チャンク")
                st.write(f"**参照グループ**: グループ{group_id}")
    
    # 基本情報を2列で表示
    refinement_display = group.refinement_method
    if is_forced:
        # 強制クラスター化の方法を日本語化
        force_method_map = {
            'low_threshold_forced': '低閾値強制マッチング',
            'cluster_expansion_forced': 'クラスター拡張',
            'nearest_neighbor_forced': '最近傍マッチング',
            'orphan_consolidation': '孤立チャンク統合'
        }
        refinement_display = force_method_map.get(group.refinement_method, group.refinement_method)
    
    info_html = f"""
    <div style="
        text-align: right;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #666;
    ">
        <strong>比較タイプ</strong>: {group.group_type} | 
        <strong>強度</strong>: {group.strength:.4f} 
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)
    
    # 階層情報を表示
    if hasattr(group, 'old_hierarchy') and group.old_hierarchy:
        hierarchy_info_box = _render_info_box("上位階層情報", "#004a55")
        st.markdown(hierarchy_info_box, unsafe_allow_html=True)
        
        # 旧文書の階層
        st.markdown(f"**変更前階層**: {group.old_hierarchy}")
        
        # 新文書の階層（複数の場合があるため）
        if hasattr(group, 'new_hierarchies') and group.new_hierarchies:
            if len(group.new_hierarchies) == 1:
                st.markdown(f"**変更後階層**: {group.new_hierarchies[0]}")
            else:
                st.markdown(f"**変更後階層**: {', '.join(group.new_hierarchies)}")
        
        st.markdown("---")  # 区切り線
    
    # 変更の種類をシンプルに表示（共通情報ボックスを使用）
    change_info_box = _render_info_box("変更の種類", "#004a55")
    st.markdown(change_info_box, unsafe_allow_html=True)
    
    # シンプルなバッジ
    change_badge = render_change_type_badge(group.analysis.change_type)
    st.markdown(change_badge)
    
    # 概要
    if group.analysis.summary:
        st.markdown(f"**概要**: {group.analysis.summary}")
    
    # 主な変更点
    if group.analysis.main_changes:
        main_changes_box = _render_info_box("主な変更点の詳細", "#004a55")
        st.markdown(main_changes_box, unsafe_allow_html=True)
        for change in group.analysis.main_changes:
            st.markdown(f"- {change}")
    
    # 対応関係情報（初期は折りたたみ）
    with st.expander("対応チャンクの詳細", expanded=False):
        correspondence_col1, correspondence_col2 = st.columns(2)
        with correspondence_col1:
            old_info_box = _render_info_box("変更前", "#004a55")
            st.markdown(old_info_box, unsafe_allow_html=True)
            for chunk in group.old_chunks:
                render_chunk_info(chunk, "旧")
        with correspondence_col2:
            new_info_box = _render_info_box("変更後", "#004a55")
            st.markdown(new_info_box, unsafe_allow_html=True)
            for chunk in group.new_chunks:
                render_chunk_info(chunk, "新")
    
    # 詳細分析（展開可能）
    with st.expander("詳細分析を表示"):
        st.markdown(group.analysis.detailed_analysis)
        
        if group.analysis.correspondence_details:
            st.markdown("**対応関係の詳細**")
            st.markdown(group.analysis.correspondence_details)

def render_added_chunk(chunk: ChunkInfo, index: int):
    """追加チャンクを表示"""
    # 共通ヘッダー生成を使用
    header_html = _render_chunk_header(f"追加チャンク {index + 1}", UI_COLORS["change_types"]["追加"], "➕")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 共通情報ボックスを使用
    change_info_box = _render_info_box("変更の種類", UI_COLORS["change_types"]["追加"])
    st.markdown(change_info_box, unsafe_allow_html=True)
    
    # シンプルなバッジ
    change_badge = render_change_type_badge("追加")
    st.markdown(change_badge)
        
    # チャンク内容表示
    content_info_box = _render_info_box("追加内容", UI_COLORS["change_types"]["追加"])
    st.markdown(content_info_box, unsafe_allow_html=True)
    render_chunk_info(chunk, "追加")

def render_deleted_chunk(chunk: ChunkInfo, index: int):
    """削除チャンクを表示"""
    # 共通ヘッダー生成を使用
    header_html = _render_chunk_header(f"削除チャンク {index + 1}", UI_COLORS["change_types"]["削除"], "➖")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 共通情報ボックスを使用
    change_info_box = _render_info_box("変更の種類", UI_COLORS["change_types"]["削除"])
    st.markdown(change_info_box, unsafe_allow_html=True)
    
    # シンプルなバッジ
    change_badge = render_change_type_badge("削除")
    st.markdown(change_badge)
    
    # チャンク内容表示
    content_info_box = _render_info_box("削除内容", UI_COLORS["change_types"]["削除"])
    st.markdown(content_info_box, unsafe_allow_html=True)
    render_chunk_info(chunk, "削除")

def render_comparison_report(report: ComparisonReport):
    """構造化された比較レポート全体を表示"""
    
    # 概要統計の表示
    render_summary_statistics(report)
    
    # グループ分析詳細
    if report.groups:
        section_header = _render_section_header("比較結果詳細", "#059669", "")
        st.markdown(section_header, unsafe_allow_html=True)
        total_groups = len(report.groups)
        page, per_page = _render_pagination_controls(total_groups, key_prefix="groups", default_per_page=10)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        visible_groups = report.groups[start_idx:end_idx]
        for i, group in enumerate(visible_groups):
            render_group_analysis(group)
            st.markdown("---")
        st.caption(f"このページに {len(visible_groups)} 件を表示（全 {total_groups} 件中）")
    
    # 追加チャンク（統一されたレイアウト）
    if report.added_chunks:
        st.markdown("---")
        with st.expander(f"追加チャンク（{len(report.added_chunks)}件）", expanded=False):
            total_added = len(report.added_chunks)
            page_a, per_page_a = _render_pagination_controls(total_added, key_prefix="added", default_per_page=10)
            start_a = (page_a - 1) * per_page_a
            end_a = start_a + per_page_a
            for i, chunk in enumerate(report.added_chunks[start_a:end_a]):
                render_added_chunk(chunk, start_a + i)
                if start_a + i < total_added - 1:
                    st.markdown("---")
            st.caption(f"このページに {min(per_page_a, max(0, total_added - start_a))} 件を表示（全 {total_added} 件中）")
    
    # 削除チャンク（統一されたレイアウト）
    if report.deleted_chunks:
        st.markdown("---")
        with st.expander(f"削除チャンク（{len(report.deleted_chunks)}件）", expanded=False):
            total_deleted = len(report.deleted_chunks)
            page_d, per_page_d = _render_pagination_controls(total_deleted, key_prefix="deleted", default_per_page=10)
            start_d = (page_d - 1) * per_page_d
            end_d = start_d + per_page_d
            for i, chunk in enumerate(report.deleted_chunks[start_d:end_d]):
                render_deleted_chunk(chunk, start_d + i)
                if start_d + i < total_deleted - 1:
                    st.markdown("---")
            st.caption(f"このページに {min(per_page_d, max(0, total_deleted - start_d))} 件を表示（全 {total_deleted} 件中）")

def filter_report_by_change_types(report: ComparisonReport, selected_types: List[str]) -> ComparisonReport:
    """選択された変更の種類に基づいてレポートをフィルタ"""
    if "すべて" in selected_types:
        return report
    
    # 選択された変更の種類に対応するグループのみを抽出
    filtered_groups = []
    for group in report.groups:
        change_type_value = group.analysis.change_type
        if change_type_value in selected_types:
            filtered_groups.append(group)
    
    # 追加・削除チャンクのフィルタ
    filtered_added = report.added_chunks if "追加" in selected_types else []
    filtered_deleted = report.deleted_chunks if "削除" in selected_types else []
    
    # 元のレポートのコピーを作成
    import copy
    filtered_report = copy.deepcopy(report)
    
    # フィルタされたデータで置き換え
    filtered_report.groups = filtered_groups
    filtered_report.added_chunks = filtered_added
    filtered_report.deleted_chunks = filtered_deleted
    
    # サマリーを更新
    filtered_report.summary["processed_groups"] = len(filtered_groups)
    filtered_report.summary["added_chunks"] = len(filtered_added)
    filtered_report.summary["deleted_chunks"] = len(filtered_deleted)
    
    # バイパスとLLM分析の数を再計算
    bypassed_count = sum(1 for g in filtered_groups if g.processing_method == "bypassed")
    gpt_count = len(filtered_groups) - bypassed_count
    filtered_report.summary["bypassed_groups"] = bypassed_count
    filtered_report.summary["gpt_analyzed_groups"] = gpt_count
    
    return filtered_report 

def render_summary_statistics(report: ComparisonReport):
    """概要統計を表示"""
    # 概要ヘッダー
    section_header = _render_section_header("比較結果概要", "#059669", "")
    st.markdown(section_header, unsafe_allow_html=True)
     
    # メイン統計の円グラフ（3つ横並び）
    col1, col2, col3 = st.columns(3)
     
    with col1:
        # 基本統計の円グラフ
        basic_data = {
            "処理グループ": report.summary.get('processed_groups', 0),
            "追加チャンク": report.summary.get('added_chunks', 0),
            "削除チャンク": report.summary.get('deleted_chunks', 0)
        }
        basic_fig = create_pie_chart(basic_data, "基本統計", "basic_stats")
        if basic_fig:
            st.plotly_chart(basic_fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # 処理方法別統計
        processing_data = {
            "バイパス": report.summary.get('bypassed_groups', 0),
            "LLM分析": report.summary.get('gpt_analyzed_groups', 0)
        }
        processing_fig = create_pie_chart(processing_data, "処理方法別", "processing_types")
        if processing_fig:
            st.plotly_chart(processing_fig, use_container_width=True, config={'displayModeBar': False})
    
    with col3:
        # 変更タイプ別統計
        change_type_counts = {}
        for group in report.groups:
            change_type = group.analysis.change_type
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        change_fig = create_pie_chart(change_type_counts, "変更タイプ別", "change_types")
        if change_fig:
            st.plotly_chart(change_fig, use_container_width=True, config={'displayModeBar': False})

def render_detailed_statistics(report: ComparisonReport):
    """詳細統計情報を表示"""
    section_header = _render_section_header("詳細統計", "#6366f1", "")
    st.markdown(section_header, unsafe_allow_html=True)
    
    # 統計テーブルの作成
    stats_data = []
    
    # 基本統計
    stats_data.append(["基本統計", "処理グループ数", report.summary.get('processed_groups', 0)])
    stats_data.append(["基本統計", "追加チャンク数", report.summary.get('added_chunks', 0)])
    stats_data.append(["基本統計", "削除チャンク数", report.summary.get('deleted_chunks', 0)])
    
    # 処理方法別統計
    stats_data.append(["処理方法", "バイパス処理", report.summary.get('bypassed_groups', 0)])
    stats_data.append(["処理方法", "LLM分析", report.summary.get('gpt_analyzed_groups', 0)])
    
    # グループタイプ別統計
    group_type_counts = {}
    for group in report.groups:
        group_type = group.group_type
        group_type_counts[group_type] = group_type_counts.get(group_type, 0) + 1
    
    for group_type, count in group_type_counts.items():
        stats_data.append(["グループタイプ", group_type, count])
    
    # 変更タイプ別統計
    change_type_counts = {}
    for group in report.groups:
        change_type = group.analysis.change_type
        change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
    
    for change_type, count in change_type_counts.items():
        stats_data.append(["変更タイプ", change_type, count])
    
    # DataFrameとして表示
    df = pd.DataFrame(stats_data, columns=["カテゴリ", "項目", "件数"])
    st.dataframe(df, use_container_width=True) 