import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional
from .structured_models import AnalysisResult
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from .config import OPENAI_CONFIG, get_chat_llm
from .error_handling import setup_logging, create_error_analysis_result, safe_execute

load_dotenv()

# 統一されたログ設定
logger = setup_logging(__name__)

# LangChainを使用したLLMインスタンス (.env で openai/azure 切替)
llm = get_chat_llm(
    model=OPENAI_CONFIG["model"],
    temperature=OPENAI_CONFIG["temperature"]
)

def call_structured_gpt(user_prompt: str) -> AnalysisResult:
    """
    LangChainのwith_structured_outputを使用してGPT APIを呼び出し、構造化された分析結果を取得
    """
    system_prompt = """
あなたは文書比較の専門家です。旧文書と新文書のチャンク間の関係を詳細に分析してください。

以下のパターンを理解して分析してください:
- 1:1 (内容対応): 旧チャンクと新チャンクが直接対応
- 1:N (内容分散): 1つの旧チャンクが複数の新チャンクに分散
- N:1 (内容統合): 複数の旧チャンクが1つの新チャンクに統合
- N:N (複合形式変更): 複数の旧チャンクと複数の新チャンクの複雑な形式変更

各パターンで以下を明確に識別してください:
1. 内容の対応関係（どこからどこへ移動したか）
2. 追加・削除・変更された具体的内容

変更の種類は以下から適切に選択してください：
- 内容変更: 実質的な内容に修正、更新、改訂がある場合
- 形式変更: フォーマットの変更やメタデータの追加、項番の変更など形式的な変更の場合
- 追加: 新しい内容が追加された場合
- 削除: 既存の内容が削除された場合
- 変更なし: 実質的な変更がない場合

分析は具体的で詳細に行い、変更の理由や影響についても可能な範囲で言及してください。
"""
    
    try:
        # LangChainのstructured outputを使用
        structured_llm = llm.with_structured_output(AnalysisResult)
        
        # プロンプトテンプレートの作成
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_prompt}")
        ])
        
        # チェーンの作成と実行
        chain = prompt | structured_llm
        
        logger.debug("LangChain Structured Output API を呼び出し中...")
        result = chain.invoke({"user_prompt": user_prompt})
        
        logger.info(f"LangChain Structured Output 成功: change_type={result.change_type}")
        logger.debug(f"分析結果概要: {result.summary[:100]}...")
        
        return result
        
    except ValidationError as validation_error:
        logger.error(f"Pydanticバリデーションエラー: {validation_error}")
        return create_error_analysis_result(f"データ構造エラー: {str(validation_error)}")
        
    except Exception as e:
        logger.error(f"LangChain APIエラー: {e}", exc_info=True)
        return create_error_analysis_result(f"LangChain APIエラー: {str(e)}")

def _create_analysis_prompt(analysis_type: str, old_content: str, new_content: str, 
                          old_ids: List[str], new_ids: List[str]) -> str:
    """分析プロンプトの共通生成関数"""
    
    prompt_templates = {
        "1:1": f"""
以下の旧チャンクと新チャンクを比較して分析してください。

旧チャンク ({old_ids[0]}):
{old_content}

新チャンク ({new_ids[0]}):
{new_content}

注意点
- 内容変更、形式変更、追加、削除、変更なしのいずれかを選択してください。
- フォーマットの変更やメタデータの追加など形式的な変更のみの場合は、<形式変更>または<変更なし>としてください。

この1:1の対応関係において、どのような変更が行われているかを詳細に分析してください。
""",
        
        "1:N": f"""
以下の1つの旧チャンクが複数の新チャンクに分散されています。

旧チャンク ({old_ids[0]}):
{old_content}

新チャンク群:
{new_content}

分析観点:
1. 旧チャンクの内容がどのように分散されているか
2. 各新チャンクに対応する旧チャンクの部分
3. 新規追加された内容（あれば）

注意点
- 内容変更、形式変更、追加、削除、変更なしのいずれかを選択してください。
- フォーマットの変更やメタデータの追加など形式的な変更のみの場合は、<形式変更>または<変更なし>としてください。

この1:N（分散）パターンでの変更を詳細に分析してください。
""",
        
        "N:1": f"""
以下の複数の旧チャンクが1つの新チャンクに統合されています。

旧チャンク群:
{old_content}

新チャンク ({new_ids[0]}):
{new_content}

分析観点:
1. 旧チャンクの内容がどのように統合されているか
2. 追加・削除・変更された内容

注意点
- 内容変更、形式変更、追加、削除、変更なしのいずれかを選択してください。
- フォーマットの変更やメタデータの追加など形式的な変更のみの場合は、<形式変更>または<変更なし>としてください。

このN:1（統合）パターンでの変更を詳細に分析してください。
""",
        
        "N:N": f"""
以下の複数の旧チャンクと複数の新チャンクの間の関係を分析してください。

旧チャンク群:
{old_content}

新チャンク群:
{new_content}

分析観点:
1. 旧チャンクの内容がどのように変更されているか
2. 追加・削除・変更された内容
3. 複合的な変更がある場合は変更の全体像

注意点
- 内容変更、形式変更、追加、削除、変更なしのいずれかを選択してください。
- フォーマットの変更やメタデータの追加など形式的な変更のみの場合は、<形式変更>または<変更なし>としてください。

このN:N（複合変更）パターンでの変更を段階的に詳細分析してください。
"""
    }
    
    return prompt_templates.get(analysis_type, prompt_templates["N:N"])

def analyze_1_to_1_structured(old_text: str, new_text: str, old_id: str, new_id: str) -> AnalysisResult:
    """1:1の関係を構造化分析"""
    user_prompt = _create_analysis_prompt("1:1", old_text, new_text, [old_id], [new_id])
    return safe_execute(call_structured_gpt, user_prompt, 
                       fallback_result=create_error_analysis_result("1:1分析エラー"), 
                       logger=logger)

def analyze_1_to_n_structured(old_text: str, new_texts: List[str], old_id: str, new_ids: List[str], similarities: Dict) -> AnalysisResult:
    """1:Nの関係を構造化分析"""
    new_content_summary = "\n\n".join([f"新チャンク {nid}:\n{text[:1000]}" for nid, text in zip(new_ids, new_texts)])
    user_prompt = _create_analysis_prompt("1:N", old_text, new_content_summary, [old_id], new_ids)
    return safe_execute(call_structured_gpt, user_prompt, 
                       fallback_result=create_error_analysis_result("1:N分析エラー"), 
                       logger=logger)

def analyze_n_to_1_structured(old_texts: List[str], new_text: str, old_ids: List[str], new_id: str, similarities: Dict) -> AnalysisResult:
    """N:1の関係を構造化分析"""
    old_content_summary = "\n\n".join([f"旧チャンク {oid}:\n{text[:1000]}" for oid, text in zip(old_ids, old_texts)])
    user_prompt = _create_analysis_prompt("N:1", old_content_summary, new_text, old_ids, [new_id])
    return safe_execute(call_structured_gpt, user_prompt, 
                       fallback_result=create_error_analysis_result("N:1分析エラー"), 
                       logger=logger)

def analyze_n_to_n_structured(old_texts: List[str], new_texts: List[str], old_ids: List[str], new_ids: List[str], similarities: Dict) -> AnalysisResult:
    """N:Nの関係を構造化分析"""
    old_content_summary = "\n\n".join([f"旧チャンク {oid}:\n{text[:800]}" for oid, text in zip(old_ids, old_texts)])
    new_content_summary = "\n\n".join([f"新チャンク {nid}:\n{text[:800]}" for nid, text in zip(new_ids, new_texts)])
    user_prompt = _create_analysis_prompt("N:N", old_content_summary, new_content_summary, old_ids, new_ids)
    return safe_execute(call_structured_gpt, user_prompt, 
                       fallback_result=create_error_analysis_result("N:N分析エラー"), 
                       logger=logger)

def analyze_chunk_group_structured(old_chunks: List[Dict], new_chunks: List[Dict], group: Dict) -> AnalysisResult:
    """
    グループタイプに応じて適切な構造化分析を実行
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
                old_texts.append(chunk.get("text", chunk.get("content", "")))
                break
    
    new_texts = []
    for nid in new_ids:
        for chunk in new_chunks:
            if chunk["id"] == nid:
                new_texts.append(chunk.get("text", chunk.get("content", "")))
                break
    
    # タイプ別分析
    try:
        if group_type == "1:1":
            return analyze_1_to_1_structured(old_texts[0], new_texts[0], old_ids[0], new_ids[0])
        elif group_type == "1:N":
            return analyze_1_to_n_structured(old_texts[0], new_texts, old_ids[0], new_ids, similarities)
        elif group_type == "N:1":
            return analyze_n_to_1_structured(old_texts, new_texts[0], old_ids, new_ids[0], similarities)
        elif group_type == "N:N":
            return analyze_n_to_n_structured(old_texts, new_texts, old_ids, new_ids, similarities)
        else:
            # 未知のタイプの場合は汎用分析
            return analyze_generic_structured(old_texts, new_texts, old_ids, new_ids, similarities)
    except Exception as e:
        logger.error(f"構造化分析でエラー: {e}")
        return create_error_analysis_result(f"グループタイプ {group_type} の分析中にエラーが発生: {str(e)}")

def analyze_generic_structured(old_texts: List[str], new_texts: List[str], old_ids: List[str], new_ids: List[str], similarities: Dict) -> AnalysisResult:
    """汎用的な構造化分析（未知のタイプ用）"""
    old_content_summary = "\n\n".join([f"旧チャンク {oid}:\n{text[:800]}" for oid, text in zip(old_ids, old_texts)])
    new_content_summary = "\n\n".join([f"新チャンク {nid}:\n{text[:800]}" for nid, text in zip(new_ids, new_texts)])
    
    user_prompt = f"""
以下のチャンク群の関係を分析してください。

旧チャンク群:
{old_content_summary}

新チャンク群:
{new_content_summary}

これらのチャンク間の関係性と変更内容を詳細に分析してください。
"""
    return safe_execute(call_structured_gpt, user_prompt, 
                       fallback_result=create_error_analysis_result("汎用分析エラー"), 
                       logger=logger) 