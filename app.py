import streamlit as st
import os
import re
from doc_compare.pdf_util import extract_chunks_by_headings
from doc_compare.text_processing import split_sentences, get_embeddings, process_chunks_to_chunk_embeddings
from doc_compare.main_processor import process_document_comparison
from doc_compare.ui_components import render_comparison_report, filter_report_by_change_types
from doc_compare.cache_util import save_embeddings_cache, load_embeddings_cache, clear_cache, get_cache_info, calculate_text_hash
from doc_compare.markdown_exporter import export_all_formats, export_to_markdown, export_summary_to_markdown, export_to_json, export_to_csv
from doc_compare.config import SIMILARITY_THRESHOLDS
import logging
import numpy as np
import json
from datetime import datetime
from doc_compare.config import CACHE_CONFIG
import traceback

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# OpenAI APIキーの確認
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    # キャッシュ情報を確認
    cache_info = get_cache_info(prefix)
    if cache_info["exists"]:
        try:
            # キャッシュディレクトリの設定
            from doc_compare.config import CACHE_CONFIG
            cache_dir = CACHE_CONFIG["cache_directory"]
            
            # ファイルパスの構築
            def get_cache_path(filename):
                return os.path.join(cache_dir, filename)
            
            # 文分割結果読み込み
            sentences_file = get_cache_path(f"{prefix}_sentences.json")
            with open(sentences_file, "r", encoding="utf-8") as f:
                sent_lists = json.load(f)
            
            # エンベディング読み込み
            embeddings_file = get_cache_path(f"{prefix}_embeddings.npz")
            cache_data = np.load(embeddings_file)
            embeddings_array = cache_data["embeddings"]
            indices = cache_data["indices"]
            
            # エンベディングを元の構造に復元
            embeddings = [[] for _ in chunks]
            for i, (chunk_idx, sent_idx) in enumerate(indices):
                if chunk_idx < len(embeddings):
                    embeddings[chunk_idx].append(embeddings_array[i].tolist())
            
            return sent_lists, embeddings
            
        except Exception as e:
            logging.warning(f"{prefix}のキャッシュ読み込みエラー: {e}")
            # キャッシュ読み込みに失敗した場合は新規作成
    
    # キャッシュがない場合は新規作成
    logging.info(f"{prefix}の文分割とエンベディング開始")
    sent_lists = [split_sentences(c["text"]) for c in chunks]
    embeddings = [get_embeddings(sents) if sents else [] for sents in sent_lists]
    
    save_embeddings_cache(chunks, sent_lists, embeddings, prefix)
    return sent_lists, embeddings

def get_or_create_chunk_embeddings(chunks, prefix):
    """チャンク全体のエンベディングを取得または新規作成 (キャッシュ対応・存在すれば常に使用)"""
    if not chunks:
        logging.info(f"{prefix}: チャンクが空のため、チャンクエンベディング処理をスキップします。")
        return []

    cache_dir = CACHE_CONFIG.get("cache_directory", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{prefix}_chunk_embeddings.npz")

    try:
        if os.path.exists(cache_file):
            logging.info(f"'{prefix}' のチャンク全体エンベディングキャッシュファイルが見つかりました。内容の検証をスキップして使用します: {cache_file}")
            cache_data = np.load(cache_file, allow_pickle=True)
            # キャッシュファイルに 'chunk_embeddings' キーが存在することだけを期待する
            if 'chunk_embeddings' in cache_data:
                return cache_data['chunk_embeddings'].tolist() if isinstance(cache_data['chunk_embeddings'], np.ndarray) else cache_data['chunk_embeddings']
            else:
                logging.warning(f"'{prefix}' のチャンクエンベディングキャッシュファイルに 'chunk_embeddings' キーが存在しません。再作成します: {cache_file}")
        
    except Exception as e:
        logging.warning(f"'{prefix}' のチャンクエンベディングキャッシュ読み込み/検証エラー: {e}。再作成します。")

    logging.info(f"'{prefix}' のチャンク全体エンベディングを新規作成します。")
    chunk_embeddings_data = process_chunks_to_chunk_embeddings(chunks)
    
    if not isinstance(chunk_embeddings_data, np.ndarray):
        if chunk_embeddings_data is None or (isinstance(chunk_embeddings_data, list) and not chunk_embeddings_data):
            logging.warning(f"'{prefix}' の process_chunks_to_chunk_embeddings の結果が空またはNoneです。キャッシュは作成されません。")
            return []
        try:
            chunk_embeddings_array = np.array(chunk_embeddings_data)
        except Exception as e:
            logging.error(f"'{prefix}' のチャンクエンベディングをNumPy配列に変換中にエラー: {e}。キャッシュは作成されません。")
            return chunk_embeddings_data
    else:
        chunk_embeddings_array = chunk_embeddings_data
    
    if chunk_embeddings_array.size == 0:
        logging.warning(f"'{prefix}' のチャンクエンベディング配列が空です。キャッシュは作成されません。")
        return chunk_embeddings_array.tolist() if isinstance(chunk_embeddings_array, np.ndarray) else chunk_embeddings_array

    try:
        np.savez_compressed(cache_file, chunk_embeddings=chunk_embeddings_array)
        logging.info(f"'{prefix}' のチャンク全体エンベディングをキャッシュに保存しました: {cache_file}")
    except Exception as e:
        logging.error(f"'{prefix}' のチャンクエンベディングキャッシュ保存エラー: {e}")
        
    return chunk_embeddings_array.tolist() if isinstance(chunk_embeddings_array, np.ndarray) else chunk_embeddings_array

# ページ設定
st.set_page_config(
    page_title="約款比較ツール",
    page_icon="",
    layout="wide"
)

# ログファイルの初期化（アプリ起動時）
if 'log_initialized' not in st.session_state:
    st.session_state['log_initialized'] = True

# CSSスタイルの適用
def apply_custom_css():
    """カスタムCSSスタイルを適用"""
    css = """
    <style>
    /* メインコンテナのパディング調整 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1200px;
    }
    
    /* 全体的な文字色を濃くして読みやすくする */
    .stMarkdown {
        color: #333;
    }
    
    /* サイドバーのスタイル */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* ボタンのシンプルなスタイル */
    .stButton > button {
        background-color: #004a55;
        color: white;
        border: 1px solid #004a55;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #00646f;
        border: 1px solid #00646f;
    }
    
    /* スライダーのスタイル */
    .stSlider .css-1cpxqw2 {
        background-color: #004a55;
    }
    
    /* セレクトボックスのスタイル */
    .stSelectbox > div > div {
        border: 1px solid #e0e0e0;
    }
    
    /* ファイルアップローダーのスタイル */
    .stFileUploader > section > div {
        border: 2px dashed #004a55;
        padding: 1rem;
    }
    
    /* マルチセレクトのスタイル */
    .stMultiSelect > div > div {
        border: 1px solid #e0e0e0;
    }
    
    /* 数値入力のスタイル */
    .stNumberInput > div > div {
        border: 1px solid #e0e0e0;
    }
    
    /* プログレスバーのスタイル */
    .stProgress .css-1cpxqw2 {
        background-color: #004a55;
    }
    
    /* スピナーのスタイル */
    .stSpinner > div {
        border-top-color: #004a55;
    }
    
    /* リンクのスタイル */
    a {
        color: #004a55;
        text-decoration: none;
    }
    
    a:hover {
        color: #00646f;
        text-decoration: underline;
    }
    
    /* フッターを隠す（Streamlitデフォルト） */
    footer {
        visibility: hidden;
    }
    
    /* ヘッダーツールバーを隠す */
    .stApp > header {
        background-color: transparent;
    }
    
    /* メニューボタンを隠す */
    .stApp > div[data-testid="stDecoration"] {
        background-image: none;
    }
    
    /* 引用ブロックのスタイル調整 */
    blockquote {
        border-left: 4px solid #004a55;
        margin: 1rem 0;
        padding-left: 1rem;
        color: #666;
        font-style: italic;
    }
    
    /* コードブロックのスタイル */
    code {
        background-color: #f5f5f5;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-family: monospace;
        color: #004a55;
        font-weight: bold;
    }
    
    /* セクション区切り線のシンプルなスタイル */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 2rem 0;
    }
    
    /* メトリクス表示の改善 */
    .metric-container div[data-testid="metric-container"] {
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
        border-radius: 4px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# カスタムCSSを適用
apply_custom_css()

st.title("約款比較ツール")

# 設定パネル
with st.sidebar:
    st.header("設定")
    
    # マッチングアルゴリズムの選択
    st.subheader("アルゴリズム設定")
    
    # 利用可能なアルゴリズムのリスト
    available_algorithms = ["従来の閾値ベース"]
    
    matching_algorithm = st.selectbox(
        "マッチングアルゴリズム",
        available_algorithms,
        index=0,
        help="従来の閾値ベース: 既存のアルゴリズム"
    )
    
    st.subheader("キャッシュ管理")
    old_cache_info = get_cache_info("old")
    new_cache_info = get_cache_info("new")
    
    # --- チャンクエンベディングキャッシュ情報の表示を追加 ---
    st.markdown("---") # 区切り線
    st.write("**チャンクエンベディングキャッシュ:**")
    cache_dir_display = CACHE_CONFIG.get("cache_directory", ".cache")
    for prefix_label, prefix_val in [("旧文書", "old"), ("新文書", "new")]:
        chunk_emb_cache_file = os.path.join(cache_dir_display, f"{prefix_val}_chunk_embeddings.npz")
        if os.path.exists(chunk_emb_cache_file):
            try:
                file_size_bytes = os.path.getsize(chunk_emb_cache_file)
                file_size_mb = file_size_bytes / (1024 * 1024)
                st.success(f"- {prefix_label}: {file_size_mb:.2f}MB (`{os.path.basename(chunk_emb_cache_file)}`)")
            except Exception as e:
                st.warning(f"- {prefix_label}: 情報取得エラー ({e})")
        else:
            st.info(f"- {prefix_label}: キャッシュなし")
    st.markdown("---") # 区切り線
    # --- ここまで追加 ---

    st.write("**基本キャッシュ (文分割/文エンベディング等):**") # 表示を明確化
    if old_cache_info["exists"]:
        st.success(f"- 旧文書: {old_cache_info['file_size_mb']}MB (例: `old_sentences.json`, `old_embeddings.npz`)")
    else:
        st.info("- 旧文書: キャッシュなし")
    
    if new_cache_info["exists"]:
        st.success(f"- 新文書: {new_cache_info['file_size_mb']}MB (例: `new_sentences.json`, `new_embeddings.npz`)")
    else:
        st.info("- 新文書: キャッシュなし")
    
    if st.button("すべてのキャッシュをクリア"): 
        cache_cleared_messages = []
        error_messages = []

        files_to_delete_patterns = [
            "old_sentences.json", "old_embeddings.npz", "old_cache_metadata.json", "old_chunks.txt",
            "new_sentences.json", "new_embeddings.npz", "new_cache_metadata.json", "new_chunks.txt",
            "old_chunk_embeddings.npz", "new_chunk_embeddings.npz"
        ]
        
        cache_dir_for_clear = CACHE_CONFIG.get("cache_directory", ".cache")
        if not os.path.isdir(cache_dir_for_clear):
            st.info("キャッシュディレクトリが見つかりません。")
        else:
            for pattern in files_to_delete_patterns:
                file_path = os.path.join(cache_dir_for_clear, pattern)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        cache_cleared_messages.append(f"削除: `{pattern}`")
                        logging.info(f"キャッシュファイル削除: {file_path}")
                    except Exception as e:
                        error_messages.append(f"エラー ({pattern}): {str(e)}")
                        logging.warning(f"キャッシュファイル削除エラー ({file_path}): {e}")
            
            if cache_cleared_messages:
                for msg in cache_cleared_messages:
                    st.success(msg)
            if error_messages:
                for msg in error_messages:
                    st.error(msg)
            
            if not cache_cleared_messages and not error_messages:
                st.info("クリア対象のキャッシュファイルが見つかりませんでした。")
            else:
                st.success("指定されたキャッシュファイルのクリア処理が完了しました。")
        
        st.rerun()

def check_and_load_from_cache(pdf_file, prefix):
    """キャッシュをチェックして、存在する場合はキャッシュからチャンクを復元"""
    if not pdf_file:
        return None

    try:
        from doc_compare.config import CACHE_CONFIG # 関数内でimport
        import os # 関数内でimport
        cache_dir_path = CACHE_CONFIG.get("cache_directory", ".cache")
        
        if not os.path.isdir(cache_dir_path):
            logging.warning(f"({prefix}): キャッシュディレクトリが存在しません: {cache_dir_path}")
            
        expected_metadata_file = os.path.join(cache_dir_path, f"{prefix}_cache_metadata.json")
        expected_chunks_file = os.path.join(cache_dir_path, f"{prefix}_chunks.txt")
            
    except ImportError:
        logging.error(f"({prefix}): CACHE_CONFIG を import できませんでした。doc_compare.config を確認してください。")
    except Exception as e:
        logging.error(f"({prefix}): キャッシュディレクトリ/ファイル存在確認中に予期せぬエラー: {e}")
    
    try:
        cache_info = get_cache_info(prefix)
        
        if not cache_info.get("exists", False):
            return None
        
        try:
            import json 
            import os   
            from doc_compare.config import CACHE_CONFIG 
            
            cache_dir = CACHE_CONFIG.get("cache_directory", ".cache") 
            if not cache_dir: 
                logging.error(f"({prefix}): CACHE_CONFIG から cache_directory が取得できませんでした。")
                return None
            
            metadata_file = os.path.join(cache_dir, f"{prefix}_cache_metadata.json")
            
            if not os.path.exists(metadata_file):
                return None

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            chunks_file_path = os.path.join(cache_dir, f"{prefix}_chunks.txt")

            if os.path.exists(chunks_file_path):
                chunks = []
                with open(chunks_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                chunk_sections = content.split("#id: ")[1:]
                if not chunk_sections: 
                    logging.warning(f"({prefix}): {prefix}_chunks.txt をsplitしましたが、有効なセクションがありませんでした。")
                
                for section_idx, section in enumerate(chunk_sections):
                    lines = section.split("\n")
                    if lines:
                        chunk_id_line = lines[0].strip()
                        chunk_id_match = re.match(r"^([a-zA-Z0-9_]+)", chunk_id_line)
                        if not chunk_id_match:
                            logging.warning(f"({prefix}): セクション {{section_idx}} でchunk_idがパースできませんでした: {{chunk_id_line}}")
                            continue
                        chunk_id = chunk_id_match.group(1)
                        
                        text_lines = []
                        for line_idx, line_content in enumerate(lines[1:]):
                            if line_content.strip() == "--------------------------------":
                                break
                            text_lines.append(line_content)
                        chunk_text = "\n".join(text_lines).strip()
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text
                        })
                
                if chunks:
                    logging.info(f"({prefix}): {prefix}_chunks.txt から {len(chunks)}個のチャンクを復元しました。")
                    return chunks
            
            logging.info(f"({prefix}): {prefix}_chunks.txt がないか有効なチャンクなし。メタデータから基本情報のみ復元試行。")
            chunks_info = metadata.get("chunks_info", [])
            if not chunks_info:
                logging.warning(f"({prefix}): メタデータに chunks_info がないか空です。キャッシュからの復元に失敗しました。")
                return None
                
            chunks = []
            for chunk_info_item in chunks_info:
                chunks.append({
                    "id": chunk_info_item.get("chunk_id", "unknown_id"),
                    "text": f"[キャッシュから復元(メタデータのみ)] {{chunk_info_item.get('chunk_id', 'unknown_id')}}"
                })
            logging.info(f"({prefix}): メタデータからチャンク情報（IDのみ）を {len(chunks)}個復元しました。")
            return chunks
            
        except Exception as e:
            logging.error(f"({prefix}): キャッシュ復元中に致命的なエラー: {e}\n{traceback.format_exc()}")
            return None
        
    except Exception as e:
        logging.error(f"({prefix}): キャッシュチェック処理の予期せぬエラー: {e}\n{traceback.format_exc()}")
        return None

# ファイルアップロード
col1, col2 = st.columns(2)

with col1:
    st.subheader("旧ドキュメント")
    old_pdf = st.file_uploader("旧ドキュメント（PDF）", type=["pdf"], key="old_pdf")
    old_chunks = []
    if old_pdf:
        if "old_pdf_file_id" not in st.session_state: 
            cached_chunks = check_and_load_from_cache(old_pdf, "old")
            if cached_chunks:
                st.session_state["old_pdf_chunks"] = cached_chunks
                st.success(f"✅ キャッシュから{len(cached_chunks)}チャンクを復元（チャンク抽出処理をスキップ）") # メッセージ修正
            else:
                st.session_state["old_pdf_chunks"] = extract_chunks_by_headings(old_pdf, "old")
                st.info(f"🔄 新規に{len(st.session_state['old_pdf_chunks'])}チャンクを抽出")
                # --- ここから簡易メタデータ保存処理を追加 ---
                try:
                    current_chunks = st.session_state["old_pdf_chunks"]
                    if current_chunks: # チャンクが空でない場合のみメタデータを保存
                        text_hash = calculate_text_hash(current_chunks) # cache_utilからimportした関数を使用
                        metadata = {
                            "prefix": "old",
                            "text_hash": text_hash,
                            "total_chunks": len(current_chunks),
                            "total_sentences": 0, # この段階では不明なので0またはNone
                            "created_at": datetime.now().isoformat(),
                            "chunks_info": [
                                {"chunk_id": chunk["id"], "sentence_count": 0, "embedding_count": 0}
                                for chunk in current_chunks
                            ] # ダミーのchunks_info構造
                        }
                        cache_dir = CACHE_CONFIG.get("cache_directory", ".")
                        if cache_dir != "." and not os.path.exists(cache_dir):
                            os.makedirs(cache_dir, exist_ok=True) # cache_util.pyの保存処理に合わせる
                        
                        metadata_file_path = os.path.join(cache_dir, "old_cache_metadata.json")
                        with open(metadata_file_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                        logging.info(f"簡易メタデータ (old_cache_metadata.json) を保存しました: {metadata_file_path}")
                    else:
                        logging.warning("抽出されたチャンクが空のため、old_cache_metadata.json の保存をスキップしました。")
                except Exception as e:
                    logging.error(f"簡易メタデータ (old_cache_metadata.json) の保存中にエラー: {e}")
                    st.warning(f"⚠️ チャンクキャッシュのメタデータ保存に失敗しました (old): {e}")
                # --- ここまで簡易メタデータ保存処理を追加 ---
            st.session_state["old_pdf_file_id"] = old_pdf.file_id
        old_chunks = st.session_state["old_pdf_chunks"]

with col2:
    st.subheader("新ドキュメント")
    new_pdf = st.file_uploader("新ドキュメント（PDF）", type=["pdf"], key="new_pdf")
    new_chunks = []
    if new_pdf:
        if "new_pdf_file_id" not in st.session_state: 
            cached_chunks = check_and_load_from_cache(new_pdf, "new")
            if cached_chunks:
                st.session_state["new_pdf_chunks"] = cached_chunks
                st.success(f"✅ キャッシュから{len(cached_chunks)}チャンクを復元（チャンク抽出処理をスキップ）") # メッセージ修正
            else:
                st.session_state["new_pdf_chunks"] = extract_chunks_by_headings(new_pdf, "new")
                st.info(f"🔄 新規に{len(st.session_state['new_pdf_chunks'])}チャンクを抽出")
                # --- ここから簡易メタデータ保存処理を追加 ---
                try:
                    current_chunks = st.session_state["new_pdf_chunks"]
                    if current_chunks:
                        text_hash = calculate_text_hash(current_chunks)
                        metadata = {
                            "prefix": "new",
                            "text_hash": text_hash,
                            "total_chunks": len(current_chunks),
                            "total_sentences": 0, 
                            "created_at": datetime.now().isoformat(),
                            "chunks_info": [
                                {"chunk_id": chunk["id"], "sentence_count": 0, "embedding_count": 0}
                                for chunk in current_chunks
                            ]
                        }
                        cache_dir = CACHE_CONFIG.get("cache_directory", ".")
                        if cache_dir != "." and not os.path.exists(cache_dir):
                            os.makedirs(cache_dir, exist_ok=True)
                            
                        metadata_file_path = os.path.join(cache_dir, "new_cache_metadata.json")
                        with open(metadata_file_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                        logging.info(f"簡易メタデータ (new_cache_metadata.json) を保存しました: {metadata_file_path}")
                    else:
                        logging.warning("抽出されたチャンクが空のため、new_cache_metadata.json の保存をスキップしました。")
                except Exception as e:
                    logging.error(f"簡易メタデータ (new_cache_metadata.json) の保存中にエラー: {e}")
                    st.warning(f"⚠️ チャンクキャッシュのメタデータ保存に失敗しました (new): {e}")
                # --- ここまで簡易メタデータ保存処理を追加 ---
            st.session_state["new_pdf_file_id"] = new_pdf.file_id
        new_chunks = st.session_state["new_pdf_chunks"]

# 比較設定（折りたたみ可能）
if old_chunks and new_chunks:
    with st.expander("比較設定", expanded=True):
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        
        with col_settings1:
            st.write("**基本設定**")
            threshold = st.slider(
                "類似度閾値", 
                min_value=0.5, 
                max_value=1.0, 
                value=SIMILARITY_THRESHOLDS["default"], 
                step=0.05
            )
            max_groups = st.number_input("最大分析グループ数", min_value=1, max_value=100, value=20, step=1)
            
        with col_settings2:
            st.write("**効率化設定**")
            bypass_threshold = st.slider(
                "バイパス閾値", 
                min_value=0.90, 
                max_value=1.0, 
                value=SIMILARITY_THRESHOLDS["bypass"], 
                step=0.01,
                help="この強度以上のクラスターは「変更なし」として自動処理"
            )
            
            # 強制クラスター化オプションを追加
            force_clustering = st.checkbox(
                "孤立チャンクの強制クラスター化",
                value=True,
                help="孤立したチャンクを強制的にクラスター化する"
            )
            if force_clustering:
                structural_integration = st.checkbox(
                    "構造的組み入れ (推奨)",
                    value=True,
                    help="前後のチャンクが所属するクラスターに孤立チャンクを組み入れる"
                )
            else:
                structural_integration = False
        
        with col_settings3:
            st.write("**細分化設定**")
            refinement_mode = st.selectbox(
                "大きなクラスターの細分化モード",
                options=["auto", "hierarchical", "semantic", "none"],
                index=0,
                help="auto: 自動選択, hierarchical: 階層的細分化, semantic: 意味的細分化, none: 細分化なし"
            )
            
            # 初期クラスター形成モードを追加
            initial_clustering_mode = st.selectbox(
                "初期クラスター形成モード",
                options=["strict", "adaptive", "relaxed"],
                index=0,
                help="strict: 厳格な形成(推奨), adaptive: 適応的形成, relaxed: 従来方式"
            )
            
            # 階層制約オプションを追加
            use_hierarchy_constraints = st.checkbox(
                "階層制約を使用",
                value=True,
                help="上位階層情報を使用してマッチング精度を向上させる"
            )

    # 比較実行
    if st.button("比較実行", type="primary", use_container_width=True):
        if not old_chunks or not new_chunks:
            st.error("両方のPDFファイルをアップロードしてください。")
            st.stop() # PDFが両方ない場合は処理停止
        else:
            with st.spinner("比較処理中..."):
                try:
                    if old_chunks is None:
                        logging.error("old_chunks が None です！Langgraph以外の処理に進めません。")
                        st.error("旧ドキュメントのチャンクが正しく読み込めていません。ファイルを再アップロードしてください。") # ユーザー向けエラー表示
                        st.stop() # return の代わりに st.stop() を使用
                    elif not old_chunks: # 空のリストの場合
                        logging.warning("old_chunks は空のリストです。Langgraph以外の処理に進めません。")
                        st.warning("旧ドキュメントからチャンクが抽出できませんでした。ファイル内容を確認してください。") # ユーザー向けエラー表示
                        st.stop() # return の代わりに st.stop() を使用
                    
                    # new_chunks も同様にチェック
                    if new_chunks is None:
                        logging.error("new_chunks が None です！Langgraph以外の処理に進めません。")
                        st.error("新ドキュメントのチャンクが正しく読み込めていません。ファイルを再アップロードしてください。")
                        st.stop() # return の代わりに st.stop() を使用
                    elif not new_chunks:
                        logging.warning("new_chunks は空のリストです。Langgraph以外の処理に進めません。")
                        st.warning("新ドキュメントからチャンクが抽出できませんでした。ファイル内容を確認してください。")
                        st.stop() # return の代わりに st.stop() を使用

                    old_vecs = get_or_create_chunk_embeddings(old_chunks, "old")
                    new_vecs = get_or_create_chunk_embeddings(new_chunks, "new")

                    # 構造化メイン処理
                    structured_report, stats = process_document_comparison(
                        old_chunks, new_chunks, old_vecs, new_vecs, 
                        threshold, max_groups, refinement_mode, bypass_threshold,
                        force_clustering=force_clustering,
                        initial_clustering_mode=initial_clustering_mode,
                        structural_integration=structural_integration,
                        use_hierarchy_constraints=use_hierarchy_constraints
                    )

                    # 結果の検証
                    if structured_report is None:
                        st.error("比較結果が正常に生成されませんでした。")
                    else:
                        st.session_state["structured_report"] = structured_report
                        st.session_state["comparison_stats"] = stats
                        st.session_state["processed_groups"] = stats.get("processed_groups_detail", [])
                        
                        # エラーがあったグループの件数を表示
                        error_groups = [g for g in stats.get("processed_groups_detail", []) 
                                        if "エラー" in str(g.get("structured_analysis", ""))]
                        if error_groups:
                            st.warning(f"⚠️ {len(error_groups)}個のグループで分析エラーが発生しました。詳細はデバッグ情報を確認してください。")
                        
                        st.success("比較が完了しました")
                        
                        # 自動出力機能
                        if st.session_state.get("auto_export", False):
                            try:
                                auto_export_path = export_summary_to_markdown(structured_report)
                                st.info(f"📄 自動出力完了: {os.path.basename(auto_export_path)}")
                            except Exception as e:
                                st.warning(f"⚠️ 自動出力エラー: {str(e)}")
                except Exception as e:
                    st.error(f"比較処理中にエラーが発生しました: {str(e)}")
                    logging.error(f"アプリケーションレベルエラー: {e}")
                    with st.expander("エラー詳細"):
                        st.code(str(e))
                        st.write("**エラータイプ:**", type(e).__name__)
                        import traceback
                        st.code(traceback.format_exc())

# 結果表示（折りたたみ可能）
if "structured_report" in st.session_state and "comparison_stats" in st.session_state:
    structured_report = st.session_state["structured_report"]
    stats = st.session_state["comparison_stats"]
    
    # ファイル出力セクション
    with st.expander("📁 ファイル出力", expanded=False):
        st.subheader("比較結果をファイルに出力")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            st.write("**単一形式出力**")
            if st.button("📄 Markdown詳細版", use_container_width=True):
                try:
                    file_path = export_to_markdown(structured_report)
                    st.success(f"✅ 出力完了: {file_path}")
                except Exception as e:
                    st.error(f"❌ 出力エラー: {str(e)}")
            
            if st.button("📋 Markdownサマリー", use_container_width=True):
                try:
                    file_path = export_summary_to_markdown(structured_report)
                    st.success(f"✅ 出力完了: {file_path}")
                except Exception as e:
                    st.error(f"❌ 出力エラー: {str(e)}")
        
        with col_export2:
            st.write("**データ形式出力**")
            if st.button("📊 JSON形式", use_container_width=True):
                try:
                    file_path = export_to_json(structured_report)
                    st.success(f"✅ 出力完了: {file_path}")
                except Exception as e:
                    st.error(f"❌ 出力エラー: {str(e)}")
            
            if st.button("📈 CSV形式", use_container_width=True):
                try:
                    file_path = export_to_csv(structured_report)
                    st.success(f"✅ 出力完了: {file_path}")
                except Exception as e:
                    st.error(f"❌ 出力エラー: {str(e)}")
        
        with col_export3:
            st.write("**一括出力**")
            if st.button("🎯 すべての形式で出力", type="primary", use_container_width=True):
                try:
                    with st.spinner("各形式でファイルを出力中..."):
                        results = export_all_formats(structured_report)
                        
                        st.success("✅ すべての形式で出力が完了しました！")
                        
                        # 出力されたファイルの一覧を表示
                        st.write("**出力されたファイル:**")
                        for format_type, file_path in results.items():
                            file_name = os.path.basename(file_path)
                            st.write(f"- **{format_type}**: `{file_name}`")
                        
                        st.info(f"📁 出力フォルダ: `output/`")
                        
                except Exception as e:
                    st.error(f"❌ 一括出力エラー: {str(e)}")
        
        # 出力オプション
        st.markdown("---")
        col_options1, col_options2 = st.columns(2)
        
        with col_options1:
            st.write("**出力オプション**")
            auto_export = st.checkbox(
                "比較完了時に自動でMarkdownサマリーを出力", 
                value=False,
                help="比較処理が完了すると自動的にMarkdownサマリーファイルを出力します"
            )
            
            if auto_export:
                st.session_state["auto_export"] = True
            else:
                st.session_state["auto_export"] = False
        
        with col_options2:
            st.write("**出力フォルダ情報**")
            if os.path.exists("output"):
                output_files = [f for f in os.listdir("output") if f.endswith(('.md', '.json', '.csv'))]
                st.info(f"📁 output フォルダ: {len(output_files)}個のファイル")
                
                if st.button("🗂️ 出力フォルダを開く"):
                    try:
                        import subprocess
                        subprocess.run(["explorer", os.path.abspath("output")], shell=True)
                    except:
                        st.info("フォルダを手動で開いてください: `output/`")
            else:
                st.info("📁 output フォルダはまだ作成されていません")
    
    # フィルタ設定
    col_filter1, col_filter2 = st.columns([3, 1])
    with col_filter1:
        # 変更の種類別の件数を計算
        change_type_counts = {}
        for group in structured_report.groups:
            change_type = group.analysis.change_type
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        # 追加・削除チャンクの件数も追加
        change_type_counts["追加"] = change_type_counts.get("追加", 0) + len(structured_report.added_chunks)
        change_type_counts["削除"] = change_type_counts.get("削除", 0) + len(structured_report.deleted_chunks)
        
        # フィルタ選択肢を件数付きで作成
        filter_options = ["すべて"]
        for change_type in ["変更", "追加", "削除", "変更なし", "その他"]:
            count = change_type_counts.get(change_type, 0)
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

    # 詳細分析結果（概要セクション外）
    if structured_report:
        filtered_report = filter_report_by_change_types(structured_report, change_type_filter)
        render_comparison_report(filtered_report)
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
        if st.button("グループタイプデバッグ"):
            if "structured_report" in st.session_state:
                structured_report = st.session_state["structured_report"]
                
                st.write("**グループタイプの詳細情報:**")
                group_type_counts = {}
                
                for group in structured_report.groups:
                    group_type = group.group_type
                    group_type_counts[group_type] = group_type_counts.get(group_type, 0) + 1
                    
                    # 最初の5個のグループについて詳細を表示
                    if group.group_number <= 5:
                        st.write(f"グループ {group.group_number}:")
                        st.write(f"  - グループタイプ: {group.group_type}")
                        st.write(f"  - 旧チャンク数: {len(group.old_chunks)}")
                        st.write(f"  - 新チャンク数: {len(group.new_chunks)}")
                        st.write(f"  - 変更の種類: {group.analysis.change_type}")
                        st.write(f"  - 類似度強度: {group.strength:.4f}")
                        st.write("---")
                
                st.write("**グループタイプ別集計:**")
                for group_type, count in group_type_counts.items():
                    st.write(f"- {group_type}: {count}個")
                
                st.write("**期待されるグループタイプとその条件:**")
                st.write("- 1:1: 旧チャンク1個 → 新チャンク1個")
                st.write("- 1:N: 旧チャンク1個 → 新チャンク複数個")
                st.write("- N:1: 旧チャンク複数個 → 新チャンク1個")
                st.write("- N:N: 旧チャンク複数個 → 新チャンク複数個")
            else:
                st.info("結果がまだ生成されていません。")
    
    st.markdown("---")
    col_debug4, col_debug5 = st.columns(2)
    
    with col_debug4:
        if st.button("グループタイプテスト"):
            # テスト用のサンプルレポートを作成
            from doc_compare.structured_models import GroupAnalysisResult, ChunkInfo, ComparisonReport, AnalysisResult, CorrespondenceInfo
            from doc_compare.visualization import create_sankey_diagram
            
            # サンプルチャンク情報
            sample_old_1 = ChunkInfo(id="old_chunk_1", content="サンプル旧チャンク1", heading="見出し1")
            sample_old_2 = ChunkInfo(id="old_chunk_2", content="サンプル旧チャンク2", heading="見出し2")
            sample_old_3 = ChunkInfo(id="old_chunk_3", content="サンプル旧チャンク3", heading="見出し3")
            
            sample_new_1 = ChunkInfo(id="new_chunk_1", content="サンプル新チャンク1", heading="見出し1")
            sample_new_2 = ChunkInfo(id="new_chunk_2", content="サンプル新チャンク2", heading="見出し2")
            sample_new_3 = ChunkInfo(id="new_chunk_3", content="サンプル新チャンク3", heading="見出し3")
            sample_new_4 = ChunkInfo(id="new_chunk_4", content="サンプル新チャンク4", heading="見出し4")
            
            # サンプル分析結果
            sample_analysis = AnalysisResult(
                change_type="変更",
                summary="テスト用の分析結果",
                detailed_analysis="詳細な分析内容",
                main_changes=["変更点1", "変更点2"],
                correspondence_details="対応関係の詳細"
            )
            
            # サンプル対応関係
            sample_correspondence_1_1 = CorrespondenceInfo(
                old_chunk_ids=["old_chunk_1"],
                new_chunk_ids=["new_chunk_1"],
                correspondence_type="1:1"
            )
            
            sample_correspondence_1_n = CorrespondenceInfo(
                old_chunk_ids=["old_chunk_2"],
                new_chunk_ids=["new_chunk_2", "new_chunk_3"],
                correspondence_type="1:N"
            )
            
            sample_correspondence_n_1 = CorrespondenceInfo(
                old_chunk_ids=["old_chunk_2", "old_chunk_3"],
                new_chunk_ids=["new_chunk_2"],
                correspondence_type="N:1"
            )
            
            sample_correspondence_n_n = CorrespondenceInfo(
                old_chunk_ids=["old_chunk_2", "old_chunk_3"],
                new_chunk_ids=["new_chunk_3", "new_chunk_4"],
                correspondence_type="N:N"
            )
            
            # テスト用グループを作成
            test_groups = [
                GroupAnalysisResult(
                    group_number=1,
                    group_type="1:1",
                    old_chunks=[sample_old_1],
                    new_chunks=[sample_new_1],
                    strength=0.95,
                    refinement_method="original",
                    processing_method="bypassed",
                    analysis=sample_analysis,
                    correspondence=sample_correspondence_1_1
                ),
                GroupAnalysisResult(
                    group_number=2,
                    group_type="1:N",
                    old_chunks=[sample_old_2],
                    new_chunks=[sample_new_2, sample_new_3],
                    strength=0.85,
                    refinement_method="hierarchical_core",
                    processing_method="gpt_analyzed",
                    analysis=sample_analysis,
                    correspondence=sample_correspondence_1_n
                ),
                GroupAnalysisResult(
                    group_number=3,
                    group_type="N:1",
                    old_chunks=[sample_old_2, sample_old_3],
                    new_chunks=[sample_new_2],
                    strength=0.78,
                    refinement_method="semantic",
                    processing_method="gpt_analyzed",
                    analysis=sample_analysis,
                    correspondence=sample_correspondence_n_1
                ),
                GroupAnalysisResult(
                    group_number=4,
                    group_type="N:N",
                    old_chunks=[sample_old_2, sample_old_3],
                    new_chunks=[sample_new_3, sample_new_4],
                    strength=0.72,
                    refinement_method="hierarchical_remaining",
                    processing_method="gpt_analyzed",
                    analysis=sample_analysis,
                    correspondence=sample_correspondence_n_n
                )
            ]
            
            # テスト用レポート作成
            test_report = ComparisonReport(
                summary={"processed_groups": 4, "added_chunks": 0, "deleted_chunks": 0, "bypassed_groups": 1, "gpt_analyzed_groups": 3},
                groups=test_groups,
                added_chunks=[],
                deleted_chunks=[]
            )
            
            st.write("**テスト用サンキーダイアグラム（全グループタイプ含む）:**")
            test_fig = create_sankey_diagram(test_report)
            st.plotly_chart(test_fig, use_container_width=True, config={'displayModeBar': False})
            
            st.write("**テストグループの詳細:**")
            for group in test_groups:
                st.write(f"- グループ {group.group_number}: {group.group_type} (旧:{len(group.old_chunks)}, 新:{len(group.new_chunks)})")
    
    with col_debug5:
        if st.button("変更種類デバッグ"):
            if "structured_report" in st.session_state:
                structured_report = st.session_state["structured_report"]
                
                st.write("**構造化分析結果の変更種類:**")
                for group in structured_report.groups:
                    st.write(f"グループ {group.group_number}: {group.analysis.change_type}")
                
                st.write("**フィルタ機能テスト:**")
                test_filters = ["変更", "追加", "削除", "変更なし"]
                for test_filter in test_filters:
                    filtered_report = filter_report_by_change_types(structured_report, [test_filter])
                    st.write(f"{test_filter}フィルタ: {len(filtered_report.groups)}グループ")
            else:
                st.info("結果がまだ生成されていません。")
        
        if st.button("現在の状態情報"):
            if "structured_report" in st.session_state:
                st.write("**構造化レポート**: 利用可能")
            else:
                st.write("**構造化レポート**: まだ生成されていません")
