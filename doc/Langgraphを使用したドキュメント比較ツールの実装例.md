# Langgraphを使用したドキュメント比較ツールの実装例

以下は、Langgraphを使用して2つのドキュメントを比較するツールの実装例です。この例では、GPT-4.1-miniとOpenAIの埋め込みモデルを使用して、チャンク化されたドキュメント間の意味的な比較を行います。

```python
import os
from typing import List, Dict, Optional, TypedDict, Literal, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

# 環境変数の設定
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 状態の型定義
class ComparisonState(TypedDict):
    # 入力ドキュメント
    document_a: List[Dict]
    document_b: List[Dict]
    
    # 現在処理中のチャンク
    current_chunk_a: Optional[Dict]
    
    # 比較結果の蓄積
    comparison_results: List[Dict]
    
    # 処理済みチャンクの追跡
    processed_chunks_a: List[str]
    
    # エージェントの思考ログ
    agent_thoughts: List[str]
    
    # 現在のステップ
    current_step: str

# モデルの初期化
llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-large")

# ツール1: 文字列検索ツール
def string_search_tool(query: str, document: List[Dict]) -> List[Dict]:
    """
    指定された語句で文字列検索した結果を取得する
    
    Args:
        query: 検索語句
        document: 検索対象ドキュメント
        
    Returns:
        マッチしたチャンクのリスト
    """
    results = []
    keywords = query.split() if isinstance(query, str) else query
    
    for chunk in document:
        content = chunk["content"].lower()
        matches = [keyword for keyword in keywords if keyword.lower() in content]
        if matches:
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "matched_keywords": matches,
                "match_count": len(matches)
            })
    
    # マッチ数の多い順にソート
    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results

# ツール2: ベクトル類似度検索ツール
def vector_similarity_tool(
    query: str, 
    document: List[Dict], 
    threshold: float = 0.7, 
    max_results: int = 5
) -> List[Dict]:
    """
    指定された語句・文章とベクトル類似度が高いチャンクを取得する
    
    Args:
        query: クエリテキスト
        document: 検索対象ドキュメント
        threshold: 類似度閾値
        max_results: 最大結果数
        
    Returns:
        類似度スコア付きのチャンクリスト
    """
    # クエリのベクトル化
    query_embedding = embeddings.embed_query(query)
    
    results = []
    for chunk in document:
        # チャンク内容のベクトル化
        chunk_embedding = embeddings.embed_query(chunk["content"])
        
        # コサイン類似度の計算
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        
        if similarity >= threshold:
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "similarity": float(similarity)
            })
    
    # 類似度の高い順にソート
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]

# ツール3: 比較元チャンクの文脈取得ツール
def get_context_tool_a(chunk_id: str, document: List[Dict]) -> Dict:
    """
    比較元チャンクの文脈情報を取得する
    
    Args:
        chunk_id: チャンクID
        document: ドキュメント
        
    Returns:
        チャンクの詳細情報と関連メタデータ
    """
    for chunk in document:
        if chunk["id"] == chunk_id:
            return chunk
    return None

# ツール4: 比較先チャンクの文脈取得ツール
def get_context_tool_b(chunk_id: str, document: List[Dict]) -> Dict:
    """
    比較先チャンクの文脈情報を取得する
    
    Args:
        chunk_id: チャンクID
        document: ドキュメント
        
    Returns:
        チャンクの詳細情報と関連メタデータ
    """
    for chunk in document:
        if chunk["id"] == chunk_id:
            return chunk
    return None

# ツール5: 前後チャンク取得ツール
def get_adjacent_chunks_tool(
    chunk_id: str, 
    document: List[Dict], 
    direction: Literal["prev", "next"], 
    count: int = 1
) -> List[Dict]:
    """
    特定のチャンクの前、または後に続くチャンクを指定された数だけ取得
    
    Args:
        chunk_id: チャンクID
        document: ドキュメント
        direction: 取得方向（"prev"または"next"）
        count: 取得数
        
    Returns:
        指定された数の前後チャンクリスト
    """
    # チャンクのインデックスを特定
    chunk_index = -1
    for i, chunk in enumerate(document):
        if chunk["id"] == chunk_id:
            chunk_index = i
            break
    
    if chunk_index == -1:
        return []
    
    if direction == "prev":
        start_idx = max(0, chunk_index - count)
        return document[start_idx:chunk_index]
    else:  # direction == "next"
        end_idx = min(len(document), chunk_index + count + 1)
        return document[chunk_index + 1:end_idx]

# キーワード抽出関数
def extract_keywords(text: str) -> List[str]:
    """
    テキストから重要なキーワードを抽出する
    
    Args:
        text: 入力テキスト
        
    Returns:
        抽出されたキーワードのリスト
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは与えられたテキストから重要なキーワードを抽出するアシスタントです。"),
        ("user", "以下のテキストから重要なキーワードを5-10個抽出してください。キーワードのみをカンマ区切りで返してください。\n\n{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    keywords = chain.invoke({"text": text})
    return [kw.strip() for kw in keywords.split(",")]

# 比較関数
def compare_chunks(chunk_a: Dict, candidates_b: List[Dict]) -> Dict:
    """
    チャンクAと候補チャンクBを比較し、差分を特定する
    
    Args:
        chunk_a: 比較元チャンク
        candidates_b: 比較先の候補チャンクリスト
        
    Returns:
        比較結果（追加/削除/変更）と詳細情報
    """
    SIMILARITY_THRESHOLD = 0.75
    
    if not candidates_b:
        # 類似チャンクが見つからない場合は削除と判断
        return {
            "type": "削除",
            "chunk_a": chunk_a,
            "chunk_b": None,
            "details": "ドキュメントBに対応するチャンクが見つかりません"
        }
    
    # 最も類似度の高い候補を特定
    best_match = candidates_b[0]
    
    # 類似度スコアが閾値以上の場合
    if "similarity" in best_match and best_match["similarity"] >= SIMILARITY_THRESHOLD:
        # 内容が完全に一致する場合
        if chunk_a["content"] == best_match["content"]:
            return {
                "type": "一致",
                "chunk_a": chunk_a,
                "chunk_b": best_match,
                "details": "内容が完全に一致しています"
            }
        else:
            # 内容が類似しているが完全一致ではない場合は変更と判断
            # 変更内容の詳細を取得
            prompt = ChatPromptTemplate.from_messages([
                ("system", "あなたは2つのテキスト間の変更点を特定するアシスタントです。"),
                ("user", "以下の2つのテキストを比較し、変更点を箇条書きで詳細に説明してください。\n\nテキストA:\n{text_a}\n\nテキストB:\n{text_b}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            changes = chain.invoke({
                "text_a": chunk_a["content"],
                "text_b": best_match["content"]
            })
            
            return {
                "type": "変更",
                "chunk_a": chunk_a,
                "chunk_b": best_match,
                "details": changes,
                "similarity": best_match.get("similarity", 0)
            }
    else:
        # 類似度が低い場合は削除と判断し、最も近いものを参考情報として提供
        return {
            "type": "削除",
            "chunk_a": chunk_a,
            "chunk_b": best_match if "similarity" in best_match else None,
            "details": f"類似度が低いため削除と判断 (類似度: {best_match.get('similarity', 0)})",
            "similarity": best_match.get("similarity", 0)
        }

# Reactエージェントの実装
def react_agent(state: ComparisonState) -> ComparisonState:
    """
    Reactエージェントの実装
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された比較状態
    """
    # 現在のチャンクを取得
    current_chunk = state["current_chunk_a"]
    
    # 思考ログを初期化
    thoughts = []
    
    # 観察: 現在の状態を観察
    thoughts.append(f"現在処理中のチャンク: {current_chunk['id']}")
    thoughts.append(f"チャンク内容: {current_chunk['content'][:100]}...")
    
    # 思考: 比較戦略を立てる
    thoughts.append("重要なキーワードを抽出して検索を行います")
    
    # キーワード抽出
    keywords = extract_keywords(current_chunk["content"])
    thoughts.append(f"抽出したキーワード: {', '.join(keywords)}")
    
    # ツール選択と実行: キーワード検索
    thoughts.append("ツール1（文字列検索）を使用してキーワード検索を実行します")
    keyword_results = string_search_tool(keywords, state["document_b"])
    
    if not keyword_results:
        thoughts.append("キーワード検索では結果が得られませんでした。ベクトル類似度検索を試みます")
        # ベクトル類似度検索
        similarity_results = vector_similarity_tool(
            current_chunk["content"], 
            state["document_b"],
            threshold=0.7,
            max_results=5
        )
        candidates = similarity_results
    else:
        candidates = keyword_results
        
    thoughts.append(f"候補チャンク数: {len(candidates)}")
    
    # 文脈取得
    if candidates:
        thoughts.append("候補チャンクの文脈を取得します")
        for i, candidate in enumerate(candidates[:3]):  # 上位3件のみ処理
            context = get_context_tool_b(candidate["id"], state["document_b"])
            if context:
                candidates[i]["context"] = context
            
        # 前後チャンク取得（必要に応じて）
        if len(candidates) > 0:
            best_candidate = candidates[0]
            thoughts.append(f"最良候補チャンク {best_candidate['id']} の前後チャンクを取得します")
            prev_chunks = get_adjacent_chunks_tool(
                best_candidate["id"], 
                state["document_b"], 
                direction="prev", 
                count=1
            )
            next_chunks = get_adjacent_chunks_tool(
                best_candidate["id"], 
                state["document_b"], 
                direction="next", 
                count=1
            )
            
    # 比較結果の決定
    comparison_result = compare_chunks(current_chunk, candidates)
    thoughts.append(f"比較結果: {comparison_result['type']}")
    
    # 状態の更新
    state["comparison_results"].append(comparison_result)
    state["processed_chunks_a"].append(current_chunk["id"])
    state["agent_thoughts"].extend(thoughts)
    
    return state

# 初期化関数
def initialize_comparison(state: Optional[ComparisonState] = None) -> ComparisonState:
    """
    比較状態を初期化する
    
    Args:
        state: 初期状態（オプション）
        
    Returns:
        初期化された比較状態
    """
    if state is None:
        state = ComparisonState(
            document_a=[],
            document_b=[],
            current_chunk_a=None,
            comparison_results=[],
            processed_chunks_a=[],
            agent_thoughts=[],
            current_step="initialize"
        )
    
    state["current_step"] = "select_next_chunk"
    return state

# 次のチャンク選択関数
def select_next_chunk_from_doc_a(state: ComparisonState) -> Tuple[ComparisonState, str]:
    """
    ドキュメントAから次の処理対象チャンクを選択する
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された状態と次のノード名
    """
    # 処理済みでないチャンクを探す
    for chunk in state["document_a"]:
        if chunk["id"] not in state["processed_chunks_a"]:
            state["current_chunk_a"] = chunk
            state["current_step"] = "react_agent"
            return state, "has_more_chunks"
    
    # すべてのチャンクが処理済みの場合
    state["current_chunk_a"] = None
    state["current_step"] = "format_output"
    return state, "no_more_chunks"

# 比較結果の処理関数
def process_comparison_results(state: ComparisonState) -> ComparisonState:
    """
    比較結果を処理する
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された比較状態
    """
    # 必要に応じて結果の後処理を行う
    state["current_step"] = "select_next_chunk"
    return state

# 出力整形関数
def format_output_as_markdown_table(state: ComparisonState) -> ComparisonState:
    """
    比較結果を表形式のマークダウンに整形する
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された比較状態（出力結果を含む）
    """
    # テーブルヘッダー
    markdown = "# ドキュメント比較結果\n\n"
    markdown += "## 概要\n\n"
    
    # 統計情報
    total = len(state["comparison_results"])
    changes = sum(1 for r in state["comparison_results"] if r["type"] == "変更")
    additions = sum(1 for r in state["comparison_results"] if r["type"] == "追加")
    deletions = sum(1 for r in state["comparison_results"] if r["type"] == "削除")
    matches = sum(1 for r in state["comparison_results"] if r["type"] == "一致")
    
    markdown += f"- 総チャンク数: {total}\n"
    markdown += f"- 変更: {changes}\n"
    markdown += f"- 追加: {additions}\n"
    markdown += f"- 削除: {deletions}\n"
    markdown += f"- 一致: {matches}\n\n"
    
    # 詳細テーブル
    markdown += "## 詳細比較結果\n\n"
    markdown += "| No. | 変更タイプ | ドキュメントA | ドキュメントB | 詳細 |\n"
    markdown += "|-----|------------|--------------|--------------|------|\n"
    
    for i, result in enumerate(state["comparison_results"], 1):
        # 変更タイプのアイコン
        type_icon = {
            "変更": "🔄 変更",
            "追加": "➕ 追加",
            "削除": "❌ 削除",
            "一致": "✓ 一致"
        }.get(result["type"], result["type"])
        
        # ドキュメントAの内容
        doc_a_content = result.get("chunk_a", {}).get("content", "-") if result["type"] != "追加" else "-"
        if len(doc_a_content) > 100:
            doc_a_content = doc_a_content[:97] + "..."
            
        # ドキュメントBの内容
        doc_b_content = result.get("chunk_b", {}).get("content", "-") if result["type"] != "削除" else "-"
        if len(doc_b_content) > 100:
            doc_b_content = doc_b_content[:97] + "..."
        
        # 詳細情報
        details = result.get("details", "")
        
        # 行の追加
        markdown += f"| {i} | {type_icon} | {doc_a_content} | {doc_b_content} | {details} |\n"
    
    # 結果を状態に保存
    state["markdown_output"] = markdown
    state["current_step"] = "complete"
    
    return state

# Langgraphの構築
def build_document_comparison_graph():
    """
    ドキュメント比較用のLanggraphを構築する
    
    Returns:
        構築されたStateGraph
    """
    # グラフの定義
    workflow = StateGraph(ComparisonState)
    
    # ノードの追加
    workflow.add_node("initialize", initialize_comparison)
    workflow.add_node("select_next_chunk", select_next_chunk_from_doc_a)
    workflow.add_node("react_agent", react_agent)
    workflow.add_node("process_comparison_results", process_comparison_results)
    workflow.add_node("format_output", format_output_as_markdown_table)
    
    # エッジの追加
    workflow.add_edge("initialize", "select_next_chunk")
    workflow.add_conditional_edges(
        "select_next_chunk",
        lambda state, result: result,
        {
            "has_more_chunks": "react_agent",
            "no_more_chunks": "format_output"
        }
    )
    workflow.add_edge("react_agent", "process_comparison_results")
    workflow.add_edge("process_comparison_results", "select_next_chunk")
    workflow.add_edge("format_output", END)
    
    # グラフのコンパイル
    return workflow.compile()

# メイン関数
def compare_documents(document_a: List[Dict], document_b: List[Dict]) -> str:
    """
    2つのドキュメントを比較する
    
    Args:
        document_a: 比較元ドキュメント（チャンクリスト）
        document_b: 比較先ドキュメント（チャンクリスト）
        
    Returns:
        比較結果のマークダウン文字列
    """
    # グラフの構築
    graph = build_document_comparison_graph()
    
    # 初期状態の設定
    initial_state = ComparisonState(
        document_a=document_a,
        document_b=document_b,
        current_chunk_a=None,
        comparison_results=[],
        processed_chunks_a=[],
        agent_thoughts=[],
        current_step="initialize"
    )
    
    # グラフの実行
    result = graph.invoke(initial_state)
    
    # 結果の取得
    return result["markdown_output"]

# 使用例
if __name__ == "__main__":
    # サンプルドキュメント
    document_a = [
        {"id": "A1", "content": "このドキュメントは、プロジェクト計画書のドラフトです。"},
        {"id": "A2", "content": "プロジェクト名：システム刷新プロジェクト"},
        {"id": "A3", "content": "プロジェクトの目標は、顧客満足度を20%向上させることです。これを達成するために、以下の3つの施策を実施します。"},
        {"id": "A4", "content": "予算は500万円を上限とします。"}
    ]
    
    document_b = [
        {"id": "B1", "content": "このドキュメントは、プロジェクト計画書のドラフトです。"},
        {"id": "B2", "content": "プロジェクト名：システム刷新プロジェクト2023"},
        {"id": "B3", "content": "プロジェクトの目標は、顧客満足度を25%向上させることです。これを達成するために、以下の4つの施策を実施します。"},
        {"id": "B4", "content": "本プロジェクトは、経営陣の承認を得て2023年4月1日に開始されました。"}
    ]
    
    # 比較の実行
    result = compare_documents(document_a, document_b)
    
    # 結果の出力
    print(result)
```

## 使用方法

1. 必要なライブラリをインストールします：

```bash
pip install langchain langchain-openai langgraph numpy
```

2. OpenAI APIキーを設定します：

```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

3. ドキュメントをチャンク化して比較を実行します：

```python
# 既にチャンク化されたドキュメントを準備
document_a = [
    {"id": "A1", "content": "チャンク1の内容"},
    {"id": "A2", "content": "チャンク2の内容"},
    # ...
]

document_b = [
    {"id": "B1", "content": "チャンク1の内容"},
    {"id": "B2", "content": "チャンク2の内容"},
    # ...
]

# 比較の実行
result = compare_documents(document_a, document_b)

# 結果の保存
with open("comparison_result.md", "w", encoding="utf-8") as f:
    f.write(result)
```

4. 比較結果をマークダウンビューアで確認します。
