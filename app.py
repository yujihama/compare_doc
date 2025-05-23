import streamlit as st
import os
import re
from doc_compare.pdf_util import extract_chunks_by_headings
from doc_compare.preprocess import split_sentences
from doc_compare.vectorize import get_embeddings
from doc_compare.main_processor import process_document_comparison
from doc_compare.cache_util import save_embeddings_cache, load_embeddings_cache, clear_cache, get_cache_info
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_groups_debug(groups):
    """グループ情報をデバッグログに書き込み"""
    lines = [f"Total groups: {len(groups)}"]
    for i, group in enumerate(groups, 1):
        lines.append(f"Group {i}:")
        lines.append(f"  old: {', '.join(group['old'])}")
        lines.append(f"  new: {', '.join(group['new'])}")
    with open("group_debug.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_or_create_embeddings(chunks, prefix):
    """エンベディングをファイルキャッシュから取得または新規作成"""
    cached_data = load_embeddings_cache(chunks, prefix)
    if cached_data is not None:
        logging.info(f"{prefix}のエンベディングをファイルキャッシュから読み込み")
        return cached_data
    
    logging.info(f"{prefix}の文分割とエンベディング開始")
    sent_lists = [split_sentences(c["text"]) for c in chunks]
    embeddings = [get_embeddings(sents) if sents else [] for sents in sent_lists]
    
    save_embeddings_cache(chunks, sent_lists, embeddings, prefix)
    return sent_lists, embeddings

def extract_change_types_from_report(report, processed_groups):
    """レポートと処理済みグループから変更の種類を抽出"""
    change_types = {}
    
    for group in processed_groups:
        group_num = group.get("group_number", 0)
        analysis = group.get("analysis", "")
        processing_method = group.get("processing_method", "")
        
        # 変更の種類を判定
        if processing_method == "bypassed":
            change_type = "変更なし"
        else:
            # まず構造化された「## 変更の種類」セクションを探す
            change_type = "その他"  # デフォルト
            
            # 「## 変更の種類」セクションから直接取得を試行
            lines = analysis.split('\n')
            for i, line in enumerate(lines):
                if line.strip() == "## 変更の種類":
                    if i + 1 < len(lines):
                        extracted_type = lines[i + 1].strip()
                        
                        # 太字マークダウンを除去して内容を取得
                        extracted_type = extracted_type.replace("**", "")
                        
                        # 標準的な変更の種類にマッピング
                        if any(word in extracted_type for word in ["変更なし", "変更無し", "変更はなし"]):
                            change_type = "変更なし"
                        elif any(word in extracted_type for word in ["追加", "新規"]):
                            change_type = "追加"
                        elif any(word in extracted_type for word in ["削除", "除去"]):
                            change_type = "削除"
                        elif any(word in extracted_type for word in ["統合", "分散", "再編成", "部分変更", "変更", "修正"]):
                            change_type = "変更"
                        break
            
            # 構造化セクションが見つからない場合は従来のキーワード検索（フォールバック）
            if change_type == "その他":
                analysis_lower = analysis.lower()
                
                if any(word in analysis for word in ["変更なし", "変更はない", "実質的な変更はない", "変更されていません"]):
                    change_type = "変更なし"
                elif any(word in analysis for word in ["追加", "新規", "新しく", "新たに", "追加された"]):
                    change_type = "追加"
                elif any(word in analysis for word in ["削除", "除去", "削除された", "なくなった", "除かれ"]):
                    change_type = "削除"
                elif any(word in analysis for word in ["変更", "修正", "改定", "改訂", "更新", "変化", "異なる", "変わっ"]):
                    change_type = "変更"
        
        change_types[group_num] = change_type
    
    return change_types

def filter_report_by_change_type(report, processed_groups, selected_types):
    """選択された変更の種類に基づいてレポートをフィルタ"""
    if "すべて" in selected_types:
        return report
    
    change_types = extract_change_types_from_report(report, processed_groups)
    
    # レポートを行単位で分割
    lines = report.split('\n')
    filtered_lines = []
    current_group_num = None
    include_current_section = True
    
    for line in lines:
        # グループヘッダーの検出（#### クラスター で始まる行）
        if line.startswith('#### クラスター '):
            # グループ番号を抽出
            match = re.search(r'#### クラスター (\d+)', line)
            if match:
                current_group_num = int(match.group(1))
                include_current_section = change_types.get(current_group_num, "その他") in selected_types
            else:
                include_current_section = True
        
        # 追加・削除チャンクセクションの処理
        elif line.startswith('## 追加チャンク'):
            include_current_section = "追加" in selected_types
        elif line.startswith('## 削除チャンク'):
            include_current_section = "削除" in selected_types
        elif line.startswith('## 概要') or line.startswith('# ドキュメント比較結果') or line.startswith('## 詳細分析'):
            include_current_section = True  # タイトル、概要、詳細分析ヘッダーは常に表示
        
        if include_current_section:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

# ページ設定
st.set_page_config(page_title="ドキュメント比較ツール", layout="wide")
st.title("ドキュメント比較ツール")
st.caption("PDFファイルの見出しベース比較ツール（観点別細分化対応）")

# 設定パネル
with st.sidebar:
    st.header("設定")
    
    st.subheader("キャッシュ管理")
    old_cache_info = get_cache_info("old")
    new_cache_info = get_cache_info("new")
    
    if old_cache_info["exists"]:
        st.success(f"旧文書キャッシュ: {old_cache_info['file_size_mb']}MB")
    else:
        st.info("旧文書キャッシュなし")
    
    if new_cache_info["exists"]:
        st.success(f"新文書キャッシュ: {new_cache_info['file_size_mb']}MB")
    else:
        st.info("新文書キャッシュなし")
    
    if st.button("キャッシュクリア"):
        clear_cache("old")
        clear_cache("new")
        st.success("キャッシュをクリアしました")
        st.rerun()

# ファイルアップロード
col1, col2 = st.columns(2)

with col1:
    st.subheader("旧ドキュメント")
    old_pdf = st.file_uploader("旧ドキュメント（PDF）", type=["pdf"], key="old_pdf")
    old_chunks = []
    if old_pdf:
        if "old_pdf_file_id" not in st.session_state or st.session_state["old_pdf_file_id"] != old_pdf.file_id:
            st.session_state["old_pdf_chunks"] = extract_chunks_by_headings(old_pdf, "old")
            st.session_state["old_pdf_file_id"] = old_pdf.file_id
        old_chunks = st.session_state["old_pdf_chunks"]
        st.success(f"{len(old_chunks)}チャンクを抽出")

with col2:
    st.subheader("新ドキュメント")
    new_pdf = st.file_uploader("新ドキュメント（PDF）", type=["pdf"], key="new_pdf")
    new_chunks = []
    if new_pdf:
        if "new_pdf_file_id" not in st.session_state or st.session_state["new_pdf_file_id"] != new_pdf.file_id:
            st.session_state["new_pdf_chunks"] = extract_chunks_by_headings(new_pdf, "new")
            st.session_state["new_pdf_file_id"] = new_pdf.file_id
        new_chunks = st.session_state["new_pdf_chunks"]
        st.success(f"{len(new_chunks)}チャンクを抽出")

# 分析設定
if old_chunks and new_chunks:
    st.subheader("分析設定")
    
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    
    with col_settings1:
        st.write("**基本設定**")
        threshold = st.slider("類似度閾値", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
        max_groups = st.number_input("最大分析グループ数", min_value=1, max_value=100, value=20, step=1)
        
    with col_settings2:
        st.write("**効率化設定**")
        bypass_threshold = st.slider(
            "バイパス閾値", 
            min_value=0.90, 
            max_value=1.0, 
            value=0.98, 
            step=0.01,
            help="この強度以上のクラスターは「変更なし」として自動処理"
        )
    
    with col_settings3:
        st.write("**細分化設定**")
        refinement_mode = st.selectbox(
            "大きなクラスターの細分化モード",
            options=["auto", "hierarchical", "semantic", "none"],
            index=0,
            help="auto: 自動選択, hierarchical: 階層的細分化, semantic: 意味的細分化, none: 細分化なし"
        )

    # 比較実行
    if st.button("比較実行", type="primary", use_container_width=True):
        if not old_chunks or not new_chunks:
            st.error("両方のPDFファイルをアップロードしてください。")
        else:
            with st.spinner("比較処理中..."):
                try:
                    # エンベディング取得
                    old_sent_lists, old_vecs = get_or_create_embeddings(old_chunks, "old")
                    new_sent_lists, new_vecs = get_or_create_embeddings(new_chunks, "new")
                    
                    # メイン処理
                    report, stats = process_document_comparison(
                        old_chunks, new_chunks, old_vecs, new_vecs, 
                        threshold, max_groups, refinement_mode, bypass_threshold
                    )
                    
                    st.session_state["comparison_result"] = report
                    st.session_state["comparison_stats"] = stats
                    # 処理済みグループの詳細をセッション状態に保存（フィルタ機能用）
                    st.session_state["processed_groups"] = stats.get("processed_groups_detail", [])
                    st.success("比較が完了しました")
                    
                except Exception as e:
                    st.error(f"比較処理中にエラーが発生しました: {e}")
                    st.exception(e)

# 結果表示
if "comparison_result" in st.session_state and "comparison_stats" in st.session_state:
    stats = st.session_state["comparison_stats"]
    
    st.subheader("比較結果")
    
    # フィルタ設定
    col_filter1, col_filter2 = st.columns([3, 1])
    with col_filter1:
        # 変更の種類別の件数を計算
        processed_groups_detail = st.session_state.get("processed_groups", [])
        change_types = extract_change_types_from_report(st.session_state["comparison_result"], processed_groups_detail)
        
        type_counts = {}
        for change_type in change_types.values():
            type_counts[change_type] = type_counts.get(change_type, 0) + 1
        
        # 追加・削除チャンクの件数も追加
        type_counts["追加"] = type_counts.get("追加", 0) + stats["added_chunks"]
        type_counts["削除"] = type_counts.get("削除", 0) + stats["deleted_chunks"]
        
        # フィルタ選択肢を件数付きで作成
        filter_options = ["すべて"]
        for change_type in ["変更", "追加", "削除", "変更なし", "その他"]:
            count = type_counts.get(change_type, 0)
            if count > 0:
                filter_options.append(f"{change_type} ({count})")
            else:
                filter_options.append(change_type)
        
        selected_filters = st.multiselect(
            "変更の種類でフィルタ",
            filter_options,
            default=["すべて"],
            help="表示する変更の種類を選択してください"
        )
        
        # 選択されたフィルタから実際の変更の種類を抽出
        change_type_filter = []
        for filter_item in selected_filters:
            if filter_item == "すべて":
                change_type_filter.append("すべて")
            else:
                # " (数)" の部分を除去
                change_type = filter_item.split(" (")[0]
                change_type_filter.append(change_type)
    
    with col_filter2:
        if st.button("フィルタリセット"):
            st.rerun()
    
    # 基本統計
    col1_result, col2_result, col3_result, col4_result = st.columns(4)
    with col1_result:
        st.metric("処理済みグループ", stats["processed_groups"])
    with col2_result:
        st.metric("追加チャンク", stats["added_chunks"])
    with col3_result:
        st.metric("削除チャンク", stats["deleted_chunks"])
    with col4_result:
        bypassed = stats.get("bypassed_groups", 0)
        gpt_analyzed = stats.get("gpt_analyzed_groups", 0)
        efficiency_rate = bypassed / max(stats["total_groups"], 1) * 100 if stats.get("total_groups", 0) > 0 else 0
        st.metric("効率化率", f"{efficiency_rate:.1f}%")
    
    # グループタイプ統計
    if stats["processed_groups"] > 0:
        st.write("**グループタイプ内訳**")
        group_col1, group_col2, group_col3, group_col4 = st.columns(4)
        with group_col1:
            st.metric("1:1", stats['group_1_to_1'])
        with group_col2:
            st.metric("1:N", stats['group_1_to_n'])
        with group_col3:
            st.metric("N:1", stats['group_n_to_1'])
        with group_col4:
            st.metric("N:N", stats['group_n_to_n'])
    
    # 処理統計
    if stats.get("bypassed_groups", 0) > 0 or stats.get("gpt_analyzed_groups", 0) > 0:
        st.write("**処理統計**")
        process_col1, process_col2, process_col3 = st.columns(3)
        with process_col1:
            st.metric("バイパス処理", stats["bypassed_groups"])
        with process_col2:
            st.metric("GPT分析", stats["gpt_analyzed_groups"])
        with process_col3:
            st.metric("バイパス閾値", f"{stats.get('bypass_threshold', 0.98):.2f}")
    
    # 細分化統計
    if stats.get("total_refined", 0) > 0:
        st.write("**細分化統計**")
        ref_col1, ref_col2, ref_col3, ref_col4 = st.columns(4)
        with ref_col1:
            st.metric("階層コア", stats["refinement_stats"]["hierarchical_core"])
        with ref_col2:
            st.metric("階層残り", stats["refinement_stats"]["hierarchical_remaining"])
        with ref_col3:
            st.metric("意味的", stats["refinement_stats"]["semantic"])
        with ref_col4:
            st.metric("元のまま", stats["refinement_stats"]["original"])
        
        st.caption(f"細分化モード: {stats.get('refinement_mode', 'unknown')} | 細分化されたクラスター: {stats['total_refined']}個")
    
    # フィルタされたレポート表示
    st.write("---")
    
    filtered_report = filter_report_by_change_type(
        st.session_state["comparison_result"], 
        processed_groups_detail, 
        change_type_filter
    )
    
    if filtered_report.strip():
        # CSSスタイルを追加してクラスター表示を改善
        st.markdown("""
        <style>
        /* クラスターヘッダーのスタイル */
        div[data-testid="stMarkdownContainer"] h4 {
            background-color: #f0f2f6;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            font-size: 1.1rem;
        }
        
        /* 分析結果の小見出しスタイル */
        div[data-testid="stMarkdownContainer"] h2 {
            color: #1f77b4;
            font-size: 1rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid #1f77b4;
            padding-left: 0.5rem;
            background-color: #f8f9fa;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        /* 区切り線のスタイル */
        div[data-testid="stMarkdownContainer"] hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #e6e9ef;
            background: linear-gradient(to right, #e6e9ef, transparent);
        }
        
        /* メインタイトルのスタイル */
        div[data-testid="stMarkdownContainer"] h1 {
            color: #262730;
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 0.5rem;
        }
        
        /* 強調テキストのスタイル */
        div[data-testid="stMarkdownContainer"] strong {
            color: #1f77b4;
        }
        
        /* コードブロックのスタイル */
        div[data-testid="stMarkdownContainer"] code {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 0.25rem;
            padding: 0.125rem 0.25rem;
            font-size: 0.875rem;
        }
        
        /* テーブルのスタイル */
        div[data-testid="stMarkdownContainer"] table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        div[data-testid="stMarkdownContainer"] th,
        div[data-testid="stMarkdownContainer"] td {
            border: 1px solid #dee2e6;
            padding: 0.5rem;
            text-align: left;
        }
        
        div[data-testid="stMarkdownContainer"] th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        /* 全体的なフォント調整 */
        div[data-testid="stMarkdownContainer"] {
            font-family: 'Source Sans Pro', sans-serif;
            line-height: 1.6;
        }
        
        /* リストアイテムのスタイル */
        div[data-testid="stMarkdownContainer"] ul li {
            margin-bottom: 0.25rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(filtered_report)
    else:
        st.info("選択したフィルタに該当する結果がありません。")

# デバッグ機能
with st.expander("デバッグ情報"):
    col_debug1, col_debug2, col_debug3 = st.columns(3)
    
    with col_debug1:
        if st.button("グループデバッグログ"):
            try:
                with open("group_debug.log", "r", encoding="utf-8") as f:
                    st.text_area("group_debug.log", f.read(), height=200)
            except FileNotFoundError:
                st.warning("ログファイルが見つかりません。")
    
    with col_debug2:
        if st.button("類似度デバッグログ"):
            try:
                with open("similarity_debug.log", "r", encoding="utf-8") as f:
                    st.text_area("similarity_debug.log", f.read(), height=300)
            except FileNotFoundError:
                st.warning("類似度デバッグログが見つかりません。")
    
    with col_debug3:
        if st.button("変更種類デバッグ") and "processed_groups" in st.session_state:
            processed_groups_detail = st.session_state.get("processed_groups", [])
            change_types = extract_change_types_from_report(st.session_state["comparison_result"], processed_groups_detail)
            
            st.write("**変更の種類判定結果:**")
            for group_num, change_type in change_types.items():
                st.write(f"グループ {group_num}: {change_type}")
            
            st.write("**処理済みグループの詳細:**")
            for group in processed_groups_detail[:3]:  # 最初の3個のみ表示
                st.write(f"グループ {group.get('group_number', 0)}:")
                st.write(f"  処理方法: {group.get('processing_method', 'unknown')}")
                st.write(f"  分析結果の先頭部分: {group.get('analysis', '')[:200]}...")
                st.write("---")
            
            # フィルタ機能のデバッグ
            st.write("**フィルタ機能テスト:**")
            test_filters = ["変更", "追加", "削除", "変更なし"]
            for test_filter in test_filters:
                filtered_report = filter_report_by_change_type(
                    st.session_state["comparison_result"], 
                    processed_groups_detail, 
                    [test_filter]
                )
                section_count = filtered_report.count("#### クラスター ")
                st.write(f"{test_filter}フィルタ: {section_count}セクション")
            
            # 生の分析結果の「## 変更の種類」セクションを表示
            st.write("**構造化された変更の種類セクション:**")
            for group in processed_groups_detail[:2]:
                analysis = group.get('analysis', '')
                lines = analysis.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == "## 変更の種類":
                        if i + 1 < len(lines):
                            raw_type = lines[i + 1].strip()
                            clean_type = raw_type.replace("**", "")
                            st.write(f"グループ {group.get('group_number', 0)}: '{raw_type}' → '{clean_type}'")
                        break 