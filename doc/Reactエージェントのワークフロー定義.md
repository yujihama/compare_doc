# Reactエージェントのワークフロー定義

## 1. エージェントの概要

Reactエージェントは、2つのドキュメント間の比較を自律的に行うための中核コンポーネントです。このエージェントは、与えられたツールを使用して探索的に比較を行い、追加・削除・変更された内容を特定します。

## 2. 思考プロセス設計

Reactエージェントは以下の思考プロセスに従って動作します：

```
┌───────────────────┐
│  観察 (Observe)   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  思考 (Think)     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  ツール選択 (Tool) │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  行動 (Act)       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  結果評価 (Reflect)│
└───────────────────┘
```

### 2.1 各ステップの詳細

#### 観察 (Observe)
- 現在の状態を観察する
- 処理中のチャンク情報を確認する
- これまでの比較結果を確認する

#### 思考 (Think)
- 現在のチャンクの内容を理解する
- 比較戦略を立てる（キーワード抽出、意味的類似性の検討など）
- 必要なツールと使用順序を決定する

#### ツール選択 (Tool)
- 最適なツールを選択する
- ツールに必要なパラメータを決定する

#### 行動 (Act)
- 選択したツールを実行する
- ツールの結果を取得する

#### 結果評価 (Reflect)
- ツールの結果を評価する
- 次のステップを決定する（追加のツール使用、比較結果の記録など）
- 必要に応じて戦略を修正する

## 3. ツール選択ロジック

エージェントは以下のロジックに基づいてツールを選択します：

### 3.1 初期探索フェーズ

1. **キーワード抽出と検索**
   - 現在のチャンクから重要なキーワードを抽出
   - **ツール1（文字列検索）** を使用して、抽出したキーワードで比較先ドキュメントを検索

2. **意味的類似性検索**
   - キーワード検索で十分な結果が得られない場合
   - **ツール2（ベクトル類似度検索）** を使用して、意味的に類似したチャンクを検索

### 3.2 文脈理解フェーズ

3. **文脈の取得**
   - 比較元チャンクの文脈を理解するために **ツール3（比較元チャンクの文脈取得）** を使用
   - 候補となる比較先チャンクの文脈を理解するために **ツール4（比較先チャンクの文脈取得）** を使用

4. **前後関係の確認**
   - より広い文脈を理解するために **ツール5（前後チャンク取得）** を使用
   - 比較元または比較先チャンクの前後のチャンクを取得して関連性を確認

### 3.3 決定フェーズ

5. **比較と差分特定**
   - 収集した情報を基に、チャンク間の差分を特定
   - 追加、削除、変更のいずれかに分類

## 4. 比較アルゴリズム

エージェントは以下のアルゴリズムに従って比較を行います：

```python
def compare_chunks(chunk_a, candidates_b):
    """
    チャンクAと候補チャンクBを比較し、差分を特定する
    
    Args:
        chunk_a: 比較元チャンク
        candidates_b: 比較先の候補チャンクリスト
        
    Returns:
        比較結果（追加/削除/変更）と詳細情報
    """
    # 最も類似度の高い候補を特定
    best_match = find_best_match(chunk_a, candidates_b)
    
    if best_match is None:
        # 類似チャンクが見つからない場合は削除と判断
        return {
            "type": "削除",
            "chunk_a": chunk_a,
            "chunk_b": None,
            "details": "ドキュメントBに対応するチャンクが見つかりません"
        }
    
    # 類似度スコアが閾値以上の場合
    if best_match["similarity"] >= SIMILARITY_THRESHOLD:
        # 内容が完全に一致する場合
        if is_exact_match(chunk_a["content"], best_match["content"]):
            return {
                "type": "一致",
                "chunk_a": chunk_a,
                "chunk_b": best_match,
                "details": "内容が完全に一致しています"
            }
        else:
            # 内容が類似しているが完全一致ではない場合は変更と判断
            return {
                "type": "変更",
                "chunk_a": chunk_a,
                "chunk_b": best_match,
                "details": identify_changes(chunk_a["content"], best_match["content"])
            }
    else:
        # 類似度が低い場合は削除と判断し、最も近いものを参考情報として提供
        return {
            "type": "削除",
            "chunk_a": chunk_a,
            "chunk_b": best_match,
            "details": f"類似度が低いため削除と判断 (類似度: {best_match['similarity']})"
        }
```

### 4.1 追加・削除・変更の判定基準

- **追加**: ドキュメントBにあるがドキュメントAにない内容
  - ドキュメントBの処理時に、ドキュメントAに類似チャンクがない場合

- **削除**: ドキュメントAにあるがドキュメントBにない内容
  - ドキュメントAのチャンクに対して、ドキュメントBに類似チャンクがない場合

- **変更**: 両方のドキュメントに存在するが内容が異なる
  - 類似度が閾値以上だが完全一致ではない場合
  - 変更の詳細（追加された文、削除された文、修正された文）を特定

## 5. GPT-4.1-miniの活用方法

エージェントはGPT-4.1-miniを以下の目的で活用します：

1. **キーワード抽出**
   - チャンクから重要なキーワードを抽出する

2. **意味理解**
   - チャンクの意味内容を理解し、要約する

3. **差分分析**
   - 2つのチャンク間の意味的な差分を分析する

4. **結果整理**
   - 比較結果を構造化し、人間が理解しやすい形式に整理する

## 6. エージェントの実装例

```python
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
    
    # キーワード抽出（GPT-4.1-miniを使用）
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
```

## 7. エージェントの出力例

```
現在処理中のチャンク: A5
チャンク内容: プロジェクトの目標は、顧客満足度を20%向上させることです。これを達成するために、以下の3つの施策を実施します...

重要なキーワードを抽出して検索を行います
抽出したキーワード: プロジェクト, 目標, 顧客満足度, 20%, 向上, 施策

ツール1（文字列検索）を使用してキーワード検索を実行します
候補チャンク数: 2

候補チャンクの文脈を取得します
最良候補チャンク B7 の前後チャンクを取得します

比較結果: 変更
変更内容: 
- 顧客満足度の目標が「20%向上」から「25%向上」に変更
- 施策が3つから4つに増加
- 新たに「オンラインサポートの強化」が追加
```
