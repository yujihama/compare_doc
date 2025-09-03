"""設定値の一元管理"""
import os
from dotenv import load_dotenv

# LLM/Embeddings 切替のための読み込み
try:
    # 依存は requirements.txt に含まれている想定
    from langchain_openai import (
        ChatOpenAI,
        OpenAIEmbeddings,
        AzureChatOpenAI,
        AzureOpenAIEmbeddings,
    )
except Exception:
    # 実行環境により初期ロードで失敗しても、使用時に ImportError を報告させる
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    AzureChatOpenAI = None  # type: ignore
    AzureOpenAIEmbeddings = None  # type: ignore

load_dotenv()

# 類似度計算設定
SIMILARITY_THRESHOLDS = {
    "default": 0.75,
    "strict": 0.95,
    "loose": 0.6,
    "bypass": 1.00
}

# クラスタリング設定
CLUSTERING_CONFIG = {
    "max_cluster_size": 15,
    "min_cluster_size": 2,
    "refinement_threshold": 0.85,
    "large_size_threshold": 10,
    "min_remaining_for_split": 15
}

# UI色設定
UI_COLORS = {
    "change_types": {
        "要判定": "#10d9c4",
        "追加": "#60a5fa",
        "削除": "#f87171",
        "変更": "#8b5cf6",
        "変更なし": "#9ca3af",
        "その他": "#d1d5db"
    },
    "group_types": {
        "1:1": "rgba(60, 179, 113, 0.7)",
        "1:N": "rgba(30, 144, 255, 0.7)",
        "N:1": "rgba(255, 165, 0, 0.7)",
        "N:N": "rgba(255, 99, 71, 0.7)",
        "unknown": "rgba(128, 128, 128, 0.7)"
    },
    "processing_types": {
        "バイパス": "#fbbf24",
        "LLM分析": "#a78bfa"
    },
    "chunk_types": {
        "old": "#87CEEB",      # スカイブルー
        "new": "#98FB98",      # ペールグリーン
        "deleted": "#FFA07A",  # ライトサーモン
        "added": "#FFB6C1"     # ライトピンク
    }
}

# エクスポート設定
EXPORT_CONFIG = {
    "max_content_length": 500,
    "timestamp_format": "%Y%m%d_%H%M%S",
    "max_heading_length": 20,
    "max_summary_length": 200
}

# PDF処理設定
PDF_CONFIG = {
    "chunk_size_range": (100, 500),
    "max_chunk_processing": 100,
    "text_normalization": {
        "min_line_length": 3,           # 意味のある行の最小文字数
        "max_single_char_combine": 50,  # 1文字行結合の最大文字数
        "combine_short_lines": True,    # 短い行を前の行と結合するか
    },
    "heading_patterns": [
        r'^\d+\.\s+.+',           # "1. 概要"
        r'^第\d+章\s+.+',         # "第1章 はじめに"  
        r'^\d+\.\d+\s+.+',        # "1.1 目的"
        r'^\d+\.\d+\.\d+\s+.+',   # "1.1.1 詳細"
        r'^[A-Z]\.\s+.+',         # "A. 整備方針"
        r'^\(\d+\)\s+.+',         # "(1) 概要"
        r'^■.+',                  # "■重要事項"
        r'^【.+】',                # "【注意事項】"
        r'^◆.+',                  # "◆ポイント"
        r'^●.+',                  # "●重要"
        r'^第\d+条\s*.+',         # "第1条 定義"
        r'^第\d+節\s*.+',         # "第1節 総則"
        r'^附則\s*\d*\s*.+',      # "附則 経過措置"
    ]
}

# ログ設定
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# OpenAI API設定
OPENAI_CONFIG = {
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "embedding_model": "text-embedding-3-small"
}

# キャッシュ設定
CACHE_CONFIG = {
    "enable_cache": True,
    "cache_directory": "cache",
    "max_cache_age_days": 7
} 

# =============================
# LLM/Embedding ファクトリ
# =============================

def _get_provider() -> str:
    """.env から LLM プロバイダを取得 (openai | azure)。既定は openai。
    Returns:
        str: "openai" or "azure"
    """
    return (os.getenv("LLM_PROVIDER", "openai") or "openai").strip().lower()


def get_chat_llm(model: str | None = None, temperature: float | None = None):
    """Chat用 LLM インスタンスを返す（OpenAI / Azure を .env で切替）。

    .env (例):
      LLM_PROVIDER=openai|azure

      # OpenAI
      OPENAI_API_KEY=...

      # Azure OpenAI
      AZURE_OPENAI_API_KEY=...
      AZURE_OPENAI_ENDPOINT=...        # 例: https://your-resource.openai.azure.com/ または https://models.azure.com/
      AZURE_OPENAI_API_VERSION=...     # 例: 2024-02-15-preview
      AZURE_OPENAI_CHAT_DEPLOYMENT=... # デプロイ名 (通常は必須)
    """

    provider = _get_provider()
    mdl = model or OPENAI_CONFIG["model"]
    temp = OPENAI_CONFIG["temperature"] if temperature is None else temperature

    if provider == "azure":
        if AzureChatOpenAI is None:
            raise ImportError("AzureChatOpenAI が読み込めません。'langchain_openai' がインストール済みか確認してください。")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
        azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # 通常は azure_deployment 指定を推奨
        if azure_deployment:
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=azure_deployment,
                temperature=temp,
            )
        # models.azure.com (グローバル) 等で model 指定を使う場合のフォールバック
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            model=mdl,
            temperature=temp,
        )

    # OpenAI (デフォルト)
    if ChatOpenAI is None:
        raise ImportError("ChatOpenAI が読み込めません。'langchain_openai' がインストール済みか確認してください。")
    return ChatOpenAI(model=mdl, temperature=temp, api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings_client(model: str | None = None):
    """Embeddings クライアントを返す（OpenAI / Azure を .env で切替）。

    .env (例):
      LLM_PROVIDER=openai|azure

      # OpenAI
      OPENAI_API_KEY=...

      # Azure OpenAI
      AZURE_OPENAI_API_KEY=...
      AZURE_OPENAI_ENDPOINT=...
      AZURE_OPENAI_API_VERSION=...
      AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=... # デプロイ名 (通常は必須)
    """
    provider = _get_provider()
    emb_model = model or OPENAI_CONFIG["embedding_model"]

    if provider == "azure":
        if AzureOpenAIEmbeddings is None:
            raise ImportError("AzureOpenAIEmbeddings が読み込めません。'langchain_openai' がインストール済みか確認してください。")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

        if azure_deployment:
            return AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=azure_deployment,
            )
        # models.azure.com 等で model 指定を使う場合
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            model=emb_model,
        )

    # OpenAI (デフォルト)
    if OpenAIEmbeddings is None:
        raise ImportError("OpenAIEmbeddings が読み込めません。'langchain_openai' がインストール済みか確認してください。")
    return OpenAIEmbeddings(model=emb_model, api_key=os.getenv("OPENAI_API_KEY"))