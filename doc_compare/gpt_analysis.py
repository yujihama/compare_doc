import openai
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
あなたは文書比較の専門家です。旧文書と新文書のチャンク間の関係を詳細に分析してください。

以下のパターンを理解して分析してください:
- 1:1 (内容対応): 旧チャンクと新チャンクが直接対応
- 1:N (内容分散): 1つの旧チャンクが複数の新チャンクに分散
- N:1 (内容統合): 複数の旧チャンクが1つの新チャンクに統合
- N:N (複合再編成): 複数の旧チャンクと複数の新チャンクの複雑な再編成

各パターンで以下を明確に識別してください:
1. 内容の対応関係（どこからどこへ移動したか）
2. 追加・削除・変更された具体的内容
3. 文書構造変更の意図や理由
4. 読者への影響（わかりやすさ、アクセス性の変化）

複雑な関係の場合は、段階的に分析し、全体像を把握してから詳細を説明してください。

**重要：出力は必ず以下の厳密なマークダウンフォーマットに従ってください：**

## 変更の種類
**統合** / **分散** / **再編成** / **部分変更** / **追加** / **削除** / **変更なし** のいずれかを**太字**で記載

## 変更前
（旧チャンクの内容や特徴を記載。該当しない場合は「該当なし」と記載）

## 変更後
（新チャンクの内容や特徴を記載。該当しない場合は「該当なし」と記載）

## 変更内容の概要
詳細な分析結果をここに記載。以下の観点を含める：
- **主な変更点**: 具体的な変更内容
- **変更の目的**: なぜこの変更が行われたと考えられるか
- **影響**: 読者や文書利用者への影響
- **対応関係**: チャンク間の詳細な対応関係（複数チャンクの場合）

**注意**: 各見出しは必ず「## 」で始め、種類は太字で記載してください。
"""

def format_chunks_with_ids(chunks: List[str], chunk_ids: List[str], prefix: str) -> str:
    """チャンクIDと内容を整理して表示"""
    formatted = []
    for i, (chunk_id, content) in enumerate(zip(chunk_ids, chunks), 1):
        formatted.append(f"## {prefix}{i} ({chunk_id}):\n{content}")
    return "\n\n".join(formatted)

def create_similarity_matrix(old_ids: List[str], new_ids: List[str], similarities: Dict) -> str:
    """類似度マトリックスを表形式で作成"""
    lines = ["# 関係性マトリックス:"]
    lines.append("| 旧チャンク | 新チャンク | 類似度 |")
    lines.append("|------------|------------|--------|")
    
    for old_id in old_ids:
        for new_id in new_ids:
            sim = similarities.get((old_id, new_id), 0.0)
            if sim > 0:
                lines.append(f"| {old_id} | {new_id} | {sim:.3f} |")
    
    return "\n".join(lines)

def analyze_1_to_1(old_text: str, new_text: str, old_id: str, new_id: str) -> str:
    """1:1の関係分析"""
    user_prompt = f"""
以下の旧チャンクと新チャンクを比較してください。

# 旧チャンク ({old_id}):
{old_text}

# 新チャンク ({new_id}):
{new_text}

この1:1の対応関係において、どのような変更が行われているかを分析してください。

必ずシステムメッセージで指定された厳密なマークダウンフォーマットで出力してください。
"""
    return call_gpt(user_prompt)

def analyze_1_to_n(old_text: str, new_texts: List[str], old_id: str, new_ids: List[str], similarities: Dict) -> str:
    """1:Nの関係分析（分散パターン）"""
    user_prompt = f"""
以下の1つの旧チャンクが複数の新チャンクに分散されています。

{create_similarity_matrix([old_id], new_ids, similarities)}

# 旧チャンク ({old_id}):
{old_text}

# 新チャンク群:
{format_chunks_with_ids(new_texts, new_ids, "新チャンク")}

分析観点:
1. 旧チャンクの内容がどのように分散されているか
2. 各新チャンクに対応する旧チャンクの部分
3. 新規追加された内容（あれば）
4. 分散の目的や効果

必ずシステムメッセージで指定された厳密なマークダウンフォーマットで出力してください。
"""
    return call_gpt(user_prompt)

def analyze_n_to_1(old_texts: List[str], new_text: str, old_ids: List[str], new_id: str, similarities: Dict) -> str:
    """N:1の関係分析（統合パターン）"""
    user_prompt = f"""
以下の複数の旧チャンクが1つの新チャンクに統合されています。

{create_similarity_matrix(old_ids, [new_id], similarities)}

# 旧チャンク群:
{format_chunks_with_ids(old_texts, old_ids, "旧チャンク")}

# 新チャンク ({new_id}):
{new_text}

分析観点:
1. どの旧チャンクの内容が統合されているか
2. 統合時に追加・削除・変更された内容
3. 統合の目的や効果
4. 情報の整理方法の変化

必ずシステムメッセージで指定された厳密なマークダウンフォーマットで出力してください。
"""
    return call_gpt(user_prompt)

def analyze_n_to_n(old_texts: List[str], new_texts: List[str], old_ids: List[str], new_ids: List[str], similarities: Dict) -> str:
    """N:Nの関係分析（複合再編成パターン）"""
    user_prompt = f"""
以下の複数の旧チャンクと複数の新チャンクの間に複雑な関係性があります。

{create_similarity_matrix(old_ids, new_ids, similarities)}

# 旧チャンク群:
{format_chunks_with_ids(old_texts, old_ids, "旧チャンク")}

# 新チャンク群:
{format_chunks_with_ids(new_texts, new_ids, "新チャンク")}

分析観点:
1. **対応関係マッピング**: どの旧チャンクがどの新チャンクに主に対応しているか
2. **内容の流れ**: 旧チャンクの内容が新チャンクにどう分散・統合されているか
3. **新規追加**: 完全に新しく追加された内容
4. **削除内容**: 旧チャンクにあったが新チャンクにない内容
5. **再編成の理由**: なぜこのような複雑な構造変更が行われたと考えられるか

この複合的な再編成の全体像を段階的に分析してください。

必ずシステムメッセージで指定された厳密なマークダウンフォーマットで出力してください。
"""
    return call_gpt(user_prompt)

def call_gpt(user_prompt: str) -> str:
    """GPT APIを呼び出し"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def analyze_chunk_group(old_chunks: List[Dict], new_chunks: List[Dict], group: Dict) -> str:
    """
    グループタイプに応じて適切な分析を実行
    """
    group_type = group.get("type", "unknown")
    old_ids = group["old"]
    new_ids = group["new"]
    similarities = group.get("similarities", {})
    
    # チャンクIDから実際のテキストを取得
    old_texts = []
    for oid in old_ids:
        for chunk in old_chunks:
            if chunk["id"] == oid:
                old_texts.append(chunk["text"])
                break
    
    new_texts = []
    for nid in new_ids:
        for chunk in new_chunks:
            if chunk["id"] == nid:
                new_texts.append(chunk["text"])
                break
    
    # タイプ別分析
    if group_type == "1:1":
        return analyze_1_to_1(old_texts[0], new_texts[0], old_ids[0], new_ids[0])
    elif group_type == "1:N":
        return analyze_1_to_n(old_texts[0], new_texts, old_ids[0], new_ids, similarities)
    elif group_type == "N:1":
        return analyze_n_to_1(old_texts, new_texts[0], old_ids, new_ids[0], similarities)
    elif group_type == "N:N":
        return analyze_n_to_n(old_texts, new_texts, old_ids, new_ids, similarities)
    else:
        # 未知のタイプの場合は汎用分析
        return analyze_generic(old_texts, new_texts, old_ids, new_ids, similarities)

def analyze_generic(old_texts: List[str], new_texts: List[str], old_ids: List[str], new_ids: List[str], similarities: Dict) -> str:
    """汎用的な分析（未知のタイプ用）"""
    user_prompt = f"""
以下のチャンク群の関係を分析してください。

{create_similarity_matrix(old_ids, new_ids, similarities)}

# 旧チャンク群:
{format_chunks_with_ids(old_texts, old_ids, "旧チャンク")}

# 新チャンク群:
{format_chunks_with_ids(new_texts, new_ids, "新チャンク")}

これらのチャンク間の関係性と変更内容を分析してください。

必ずシステムメッセージで指定された厳密なマークダウンフォーマットで出力してください。
"""
    return call_gpt(user_prompt)

# 後方互換性のための関数（既存コードで使用されている場合）
def analyze_chunk_pair(old_text: Optional[str], new_text: Optional[str]) -> str:
    """
    従来の1:1分析（後方互換性のため）
    """
    if old_text and new_text:
        return analyze_1_to_1(old_text, new_text, "old_chunk", "new_chunk")
    elif old_text:
        user_prompt = f"""
以下の旧チャンクは新ドキュメントで削除されています。内容を要約してください。

# 旧チャンク:
{old_text}
"""
        return call_gpt(user_prompt)
    elif new_text:
        user_prompt = f"""
以下の新チャンクは旧ドキュメントに存在しません。内容を要約してください。

# 新チャンク:
{new_text}
"""
        return call_gpt(user_prompt)
    else:
        return "" 