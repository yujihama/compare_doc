# Langgraphを使用したReactエージェントの修正設計

## 1. 修正の背景

当初の設計では、Reactエージェントの実行フローがあらかじめ指定されており、ツールの呼び出し順序や引数設定が固定的でした。Langgraphのベストプラクティスに基づき、より自律的なエージェント設計に修正します。

## 2. Langgraphにおけるreactエージェントのベストプラクティス

Langgraphの公式ドキュメントとリファレンス実装から抽出したベストプラクティスは以下の通りです：

1. **メッセージパッシング方式の採用**：
   - エージェントの状態はメッセージのシーケンスとして管理
   - ツール呼び出しと結果はメッセージとして状態に追加

2. **LLM主導のツール選択**：
   - ツールの選択と引数の決定はLLMが自律的に行う
   - 事前に定義された実行フローではなく、LLMの推論に基づいて動的に決定

3. **シンプルなグラフ構造**：
   - 基本的には「モデル呼び出し」と「ツール実行」の2つのノード間を循環
   - 条件付きエッジでフロー制御（ツール呼び出しがある場合は継続、ない場合は終了）

4. **状態管理の簡素化**：
   - 複雑な状態管理よりもメッセージの履歴に基づく判断
   - 必要に応じて追加の状態情報を保持

## 3. 修正後のReactエージェント設計

### 3.1 状態設計

```python
class ComparisonState(TypedDict):
    """ドキュメント比較の状態"""
    # 入力ドキュメント
    document_a: List[Dict]
    document_b: List[Dict]
    
    # 現在処理中のチャンク
    current_chunk_a: Optional[Dict]
    
    # メッセージ履歴（LLMとの対話履歴）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 比較結果の蓄積
    comparison_results: List[Dict]
    
    # 処理済みチャンクの追跡
    processed_chunks_a: List[str]
```

### 3.2 ノードとエッジの設計

```
┌─────────────────┐
│    初期化       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  チャンク選択   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  モデル呼び出し │─────▶│   ツール実行    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │◀───────────────────────┘
         │
         ▼
┌─────────────────┐
│    結果整形     │
└─────────────────┘
```

#### モデル呼び出しノード

```python
def call_model(state: ComparisonState) -> ComparisonState:
    """
    LLMを呼び出してツール選択と引数決定を行う
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された比較状態
    """
    # システムプロンプトの設定
    system_prompt = SystemMessage(content="""
    あなたは2つのドキュメントを比較するアシスタントです。
    現在、ドキュメントAのチャンクを処理しています。
    ドキュメントBから対応するチャンクを見つけ、追加・削除・変更を特定してください。
    
    以下のツールが利用可能です：
    1. string_search_tool: 指定された語句で文字列検索した結果を取得する
    2. vector_similarity_tool: 指定された語句・文章とベクトル類似度が高いチャンクを取得する
    3. get_context_tool_a: 比較元チャンクの文脈情報を取得する
    4. get_context_tool_b: 比較先チャンクの文脈情報を取得する
    5. get_adjacent_chunks_tool: 特定のチャンクの前、または後に続くチャンクを指定された数だけ取得
    
    現在のチャンク情報と、これまでの比較結果を考慮して、最適なツールを選択し実行してください。
    ツールを使用する必要がなければ、最終的な比較結果を提供してください。
    """)
    
    # 現在のチャンク情報をユーザーメッセージとして追加
    if state["current_chunk_a"] and state["current_chunk_a"] not in state["processed_chunks_a"]:
        user_message = HumanMessage(content=f"""
        現在処理中のチャンク:
        ID: {state["current_chunk_a"]["id"]}
        内容: {state["current_chunk_a"]["content"]}
        
        このチャンクとドキュメントBを比較し、追加・削除・変更を特定してください。
        """)
        
        # メッセージ履歴に追加
        messages = [system_prompt] + list(state["messages"]) + [user_message]
        
        # LLMを呼び出し
        response = model.invoke(messages)
        
        # 応答をメッセージ履歴に追加
        state["messages"] = list(state["messages"]) + [user_message, response]
    
    return state
```

#### ツール実行ノード

```python
def execute_tools(state: ComparisonState) -> ComparisonState:
    """
    LLMが選択したツールを実行する
    
    Args:
        state: 現在の比較状態
        
    Returns:
        更新された比較状態
    """
    # 最新のAIメッセージからツール呼び出しを抽出
    last_message = state["messages"][-1]
    
    # ツール呼び出しがない場合は何もしない
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state
    
    # 各ツール呼び出しを処理
    tool_outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # ツール名に基づいて適切なツールを呼び出す
        if tool_name == "string_search_tool":
            result = string_search_tool(**tool_args)
        elif tool_name == "vector_similarity_tool":
            result = vector_similarity_tool(**tool_args)
        elif tool_name == "get_context_tool_a":
            result = get_context_tool_a(**tool_args)
        elif tool_name == "get_context_tool_b":
            result = get_context_tool_b(**tool_args)
        elif tool_name == "get_adjacent_chunks_tool":
            result = get_adjacent_chunks_tool(**tool_args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        # ツール実行結果をメッセージとして追加
        tool_outputs.append(
            ToolMessage(
                content=json.dumps(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
        )
    
    # ツール実行結果をメッセージ履歴に追加
    state["messages"] = list(state["messages"]) + tool_outputs
    
    return state
```

#### 条件付きエッジ関数

```python
def should_continue(state: ComparisonState) -> str:
    """
    次のステップを決定する条件関数
    
    Args:
        state: 現在の比較状態
        
    Returns:
        次のノード名を示す文字列
    """
    # 最新のメッセージを取得
    last_message = state["messages"][-1]
    
    # AIメッセージでツール呼び出しがある場合は継続
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue_tools"
    
    # AIメッセージで比較結果が含まれている場合は結果を保存
    if isinstance(last_message, AIMessage) and "comparison_result" in last_message.content:
        # 比較結果を抽出して保存
        try:
            result = extract_comparison_result(last_message.content)
            state["comparison_results"].append(result)
            state["processed_chunks_a"].append(state["current_chunk_a"]["id"])
        except:
            pass
        
        return "next_chunk"
    
    # デフォルトは次のチャンクへ
    return "next_chunk"
```

### 3.3 グラフ定義

```python
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
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("format_output", format_output_as_markdown_table)
    
    # エッジの追加
    workflow.add_edge("initialize", "select_next_chunk")
    workflow.add_conditional_edges(
        "select_next_chunk",
        lambda state, result: result,
        {
            "has_more_chunks": "call_model",
            "no_more_chunks": "format_output"
        }
    )
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue_tools": "execute_tools",
            "next_chunk": "select_next_chunk"
        }
    )
    workflow.add_edge("execute_tools", "call_model")
    workflow.add_edge("format_output", END)
    
    # グラフのコンパイル
    return workflow.compile()
```

## 4. 修正前と修正後の比較

### 4.1 主な変更点

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| ツール選択 | 事前に定義された順序で実行 | LLMが自律的に選択 |
| 引数設定 | コード内で固定的に設定 | LLMが状況に応じて決定 |
| 実行フロー | 線形的な処理フロー | 動的な判断に基づくフロー |
| 状態管理 | 複雑な状態構造 | メッセージベースの簡素な構造 |
| LLMの役割 | 限定的（差分分析のみ） | 中心的（ツール選択から結果整理まで） |

### 4.2 メリット

1. **柔軟性の向上**：
   - 事前に定義されたフローに縛られず、状況に応じた最適なツール選択が可能
   - 新しいツールの追加が容易（システムプロンプトに追加するだけ）

2. **自律性の向上**：
   - LLMがツール選択から結果整理まで一貫して担当
   - 人間の介入なしに複雑な比較タスクを完了可能

3. **保守性の向上**：
   - コードの複雑さが減少
   - ツールの追加・変更が容易

4. **Langgraphとの整合性**：
   - Langgraphの推奨パターンに準拠
   - 将来のアップデートとの互換性が向上

## 5. 実装上の注意点

1. **プロンプト設計の重要性**：
   - システムプロンプトがエージェントの性能を大きく左右
   - 各ツールの機能と使用タイミングを明確に説明する必要あり

2. **エラーハンドリング**：
   - LLMが不適切なツールを選択した場合の対処
   - ツール実行エラーの適切なフィードバック

3. **状態管理の最適化**：
   - メッセージ履歴が長くなりすぎないよう適宜要約
   - 重要な情報の保持と不要な情報の削除のバランス
