# ドキュメント比較ツール

PDFドキュメント間の変更点を自動的に検出・分析するStreamlitアプリケーションです。

## 🚀 概要

本アプリはPDFドキュメント間の差分比較を行うシンプルなツールです。現行実装は「従来の閾値ベース」アルゴリズムを使用し、結果のインタラクティブ表示とMarkdown/JSON/CSVへの出力を提供します。

### 利用可能なマッチングアルゴリズム

1. **従来の閾値ベース** - TF-IDF（エンベディング）＋類似度に基づくグルーピングとLLM補助分析

## 機能

- **PDFファイルの自動解析**: 見出しベースでのチャンク分割
- **マッチングアルゴリズム**: 現在は従来の閾値ベースのみ
- **インタラクティブな結果表示**: 変更タイプ別フィルタリング
- **多様な出力形式**: Markdown、JSON、CSV形式での結果出力
- **キャッシュ機能**: エンベディングの高速化
- **詳細なログ機能**: 処理過程の可視化

## インストール

### 必要な環境

- Python 3.8以上
- OpenAI または Azure OpenAI のAPIキー（GPT補助分析・埋め込み利用時）

### セットアップ

1. リポジトリのクローン:
```bash
git clone <repository-url>
cd compare_doc
```

2. 仮想環境の作成と有効化:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

4. 環境変数の設定:
```bash
# .envファイルを作成
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 使用方法

### Streamlitアプリケーションの起動

```bash
streamlit run app.py
```

### 基本的な使用手順

1. **ファイルアップロード**: 旧ドキュメントと新ドキュメント（PDF）をアップロード
2. **設定調整**: 類似度閾値、細分化モード、初期クラスター形成モードなどを必要に応じて調整
3. **比較実行**: ボタンクリックで分析開始
4. **結果確認**: インタラクティブな結果表示とエクスポート

### エクスポート

- Markdown（詳細/サマリー）: `doc_compare/markdown_exporter.py` の `export_to_markdown` / `export_summary_to_markdown`
- JSON: `export_to_json`
- CSV: `export_to_csv`（グループサマリー）, `export_statistics_to_csv`（統計）

## 技術仕様

### アーキテクチャ

- **フロントエンド**: Streamlit
- **バックエンド**: Python
- **AI/ML**: OpenAI/ Azure OpenAI（Chat: `gpt-4.1-mini` 既定, Embedding: `text-embedding-3-small` 既定）
- **フレームワーク**: LangChain, LangGraph
- **データ処理**: pandas, numpy, scikit-learn

### LLM/埋め込みの切替

`.env` の `LLM_PROVIDER` により OpenAI / Azure を切替可能です。`doc_compare/config.py` の `get_chat_llm` / `get_embeddings_client` を参照してください。

### ワークフロー（従来アルゴリズム）

1. **ドキュメント読み込み**: PDFから見出しベースでチャンク抽出（`doc_compare/pdf_util.py`）
2. **埋め込み生成**: 文/チャンクの埋め込み作成（`doc_compare/text_processing.py`）
3. **グルーピング**: 類似度と階層制約を用いたグループ化（`doc_compare/similarity.py`）
4. **LLM補助分析**: グループ毎の変更タイプ推定（`doc_compare/structured_gpt_analysis.py`）
5. **レポート生成**: 構造化モデルに整形（`doc_compare/structured_report.py`）→ 各種エクスポート

## ファイル構成

```
compare_doc/
├── app.py                              # Streamlitメインアプリ
├── doc_compare/
│   ├── main_processor.py               # 従来アルゴリズム
│   ├── pdf_util.py                     # PDF処理
│   ├── text_processing.py              # テキスト処理
│   └── ...
├── requirements.txt                     # 依存関係
└── README.md                           # このファイル
```

## ログとデバッグ

### ログファイル

- `log/llm_analysis_details.log`：LLM分析対象/バイパス対象の詳細（`doc_compare/main_processor.py`）
- `group_debug.log`：デバッグ用グループ情報（`app.py`）

### テスト実行

（現行実装に対応した統合テストスクリプトは未提供です）

## トラブルシューティング

### よくある問題

1. **OpenAI/Azure APIキーエラー**:
   ```bash
   # 環境変数を確認
   echo $OPENAI_API_KEY  # Linux/macOS
   echo %OPENAI_API_KEY%  # Windows
   ```

2. **インポートエラー**:
   ```bash
   # 依存関係を再インストール
   pip install -r requirements.txt --upgrade
   ```

3. **メモリエラー**:
   - 大きなPDFファイルの場合、チャンク数を制限
   - システムメモリを確認

### パフォーマンス最適化

- **キャッシュ活用**: 同じドキュメントの再処理時間短縮
- **チャンク制限**: 大規模PDFではチャンク数を制限
- **キャッシュの再利用**: 既存のエンベディングキャッシュを優先利用

## 貢献

プルリクエストやイシューの報告を歓迎します。

## ライセンス

MIT License

## 更新履歴

### v1.1.0（最新）
- ドキュメント構造化レポートとCSV統計出力を強化
- LLM補助分析（構造化出力）に対応

### v1.0.0
- 初期リリース（従来の閾値ベースアルゴリズム）