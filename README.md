# ドキュメント比較ツール

PDFドキュメント間の変更点を自動的に検出・分析するStreamlitアプリケーションです。

## 🚀 新機能: Langgraph Prebuilt Agent

最新のLanggraphのprebuilt `create_react_agent`を使用した、より効率的で保守しやすいドキュメント比較機能を追加しました。

### 利用可能なマッチングアルゴリズム

1. **従来の閾値ベース** - 既存のTF-IDF + コサイン類似度アルゴリズム
2. **Langgraph Reactエージェント** - カスタムStateGraphによる詳細制御
3. **Langgraph Prebuilt Agent** ⭐ **NEW** - Prebuilt create_react_agentによる最適化された分析

### Langgraph Prebuilt Agentの特徴

- 🤖 **最新のReactパターン**: Langgraphのprebuilt `create_react_agent`を使用
- 🧠 **シンプルで効率的**: より少ないコードで高い性能を実現
- 💾 **メモリ機能**: 会話の継続とコンテキスト保持
- 🛠️ **4つの専用ツール**:
  - 文字列検索ツール
  - ベクトル類似度検索ツール
  - チャンク詳細取得ツール
  - 前後チャンク取得ツール
- 📋 **詳細ログ**: エージェント対話、ツール実行、比較結果の詳細ログ

## 機能

- **PDFファイルの自動解析**: 見出しベースでのチャンク分割
- **複数のマッチングアルゴリズム**: 用途に応じて選択可能
- **インタラクティブな結果表示**: 変更タイプ別フィルタリング
- **多様な出力形式**: Markdown、JSON、CSV形式での結果出力
- **キャッシュ機能**: エンベディングの高速化
- **詳細なログ機能**: 処理過程の可視化

## インストール

### 必要な環境

- Python 3.8以上
- OpenAI APIキー（Langgraph機能使用時）

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
2. **アルゴリズム選択**: 
   - 従来の閾値ベース（高速、設定可能）
   - Langgraph Reactエージェント（詳細制御）
   - **Langgraph Prebuilt Agent**（推奨、最新）
3. **設定調整**: 類似度閾値やログレベルの調整
4. **比較実行**: ボタンクリックで分析開始
5. **結果確認**: インタラクティブな結果表示と出力

### Langgraph Prebuilt Agentの使用

1. サイドバーで「Langgraph Prebuilt Agent」を選択
2. 必要に応じて設定を調整:
   - ログレベル（INFO推奨）
   - 類似度閾値（0.75推奨）
   - 詳細ログ表示（ON推奨）
3. 比較実行でAIエージェントが自動分析

## 技術仕様

### アーキテクチャ

- **フロントエンド**: Streamlit
- **バックエンド**: Python
- **AI/ML**: OpenAI GPT-4o-mini, text-embedding-3-large
- **フレームワーク**: LangChain, LangGraph
- **データ処理**: pandas, numpy, scikit-learn

### Langgraph Prebuilt Agentの技術詳細

```python
# 基本的な使用例
from doc_compare.langgraph_prebuilt_matcher import LanggraphPrebuiltDocumentMatcher

matcher = LanggraphPrebuiltDocumentMatcher()
result = matcher.compare_documents(old_chunks, new_chunks)
```

### ワークフロー

1. **ドキュメント読み込み**: チャンク形式の統一
2. **ツール作成**: 4つの専用ツールの初期化
3. **Reactエージェント作成**: create_react_agentによる初期化
4. **チャンク分析**: 各チャンクを独立したスレッドで分析
5. **結果生成**: Markdown形式での結果出力

## ファイル構成

```
compare_doc/
├── app.py                              # Streamlitメインアプリ
├── doc_compare/
│   ├── langgraph_prebuilt_matcher.py   # 🆕 Prebuilt Agent
│   ├── langgraph_matcher.py            # カスタムReact Agent
│   ├── main_processor.py               # 従来アルゴリズム
│   ├── pdf_util.py                     # PDF処理
│   ├── text_processing.py              # テキスト処理
│   └── ...
├── test_prebuilt_matcher.py            # 🆕 テストスクリプト
├── requirements.txt                     # 依存関係
└── README.md                           # このファイル
```

## ログとデバッグ

### 詳細ログ機能

Langgraph Prebuilt Agentは詳細なログ機能を提供:

- `log/session_YYYYMMDD_HHMMSS/agent_interactions.jsonl` - エージェント対話ログ
- `log/session_YYYYMMDD_HHMMSS/tool_executions.jsonl` - ツール実行ログ
- `log/session_YYYYMMDD_HHMMSS/comparison_results.jsonl` - 比較結果ログ
- `log/session_YYYYMMDD_HHMMSS/session_summary.md` - セッションサマリー

### テスト実行

```bash
python test_prebuilt_matcher.py
```

## トラブルシューティング

### よくある問題

1. **OpenAI APIキーエラー**:
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
- **チャンク制限**: テストモードで最大5チャンクに制限
- **並列処理**: ツール実行の並列化

## 貢献

プルリクエストやイシューの報告を歓迎します。

## ライセンス

MIT License

## 更新履歴

### v2.1.0 (最新)
- 🆕 Langgraph Prebuilt Agent追加
- 🔧 create_react_agentによる最適化
- 📋 詳細ログ機能強化
- 💾 メモリ機能追加

### v2.0.0
- Langgraph Reactエージェント追加
- カスタムStateGraphによる詳細制御

### v1.0.0
- 初期リリース
- 従来の閾値ベースアルゴリズム 