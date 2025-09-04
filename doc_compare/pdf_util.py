from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import PyPDF2
import pdfplumber
import re
import os
from typing import BinaryIO, List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from .config import PDF_CONFIG, OPENAI_CONFIG, get_chat_llm, CACHE_CONFIG
import logging

# 環境変数を読み込み
load_dotenv()

def _read_pdf_pages(file: BinaryIO) -> List[str]:
    """
    PDF読み込み - PyPDF2とpdfplumberを比較して最適な結果を選択
    """
    pages_pypdf2 = []
    pages_pdfplumber = []
    
    # PyPDF2で抽出
    try:
        reader = PyPDF2.PdfReader(file)
        pages_pypdf2 = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        pass
    
    # pdfplumberで抽出
    try:
        file.seek(0)  # ファイルポインタをリセット
        with pdfplumber.open(file) as pdf:
            pages_pdfplumber = [page.extract_text() or "" for page in pdf.pages]
    except Exception:
        pass
    
    # 両方の結果を比較して最適な方を選択
    if pages_pypdf2 and pages_pdfplumber:
        # 最初のページで1文字行の比率を計算
        pypdf2_first = pages_pypdf2[0] if pages_pypdf2 else ""
        plumber_first = pages_pdfplumber[0] if pages_pdfplumber else ""
        
        pypdf2_lines = pypdf2_first.split('\n')
        plumber_lines = plumber_first.split('\n')
        
        pypdf2_single_ratio = sum(1 for line in pypdf2_lines if len(line.strip()) == 1) / max(1, len(pypdf2_lines))
        plumber_single_ratio = sum(1 for line in plumber_lines if len(line.strip()) == 1) / max(1, len(plumber_lines))
        
        # より良い結果を選択（1文字行が少ない方）
        if plumber_single_ratio < pypdf2_single_ratio:
            return pages_pdfplumber
        else:
            return pages_pypdf2
    
    # どちらか一方しか成功しなかった場合
    return pages_pdfplumber or pages_pypdf2 or []

def _normalize_pdf_text(text: str) -> str:
    """
    PDFから抽出されたテキストを正規化
    1文字ごとの改行や不適切な改行を修正
    """
    if not text:
        return text
    
    # 設定値を取得
    min_line_length = PDF_CONFIG["text_normalization"]["min_line_length"]
    max_combine = PDF_CONFIG["text_normalization"]["max_single_char_combine"]
    
    # 1文字ごとの改行を検出して修正
    lines = text.split('\n')
    normalized_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 空行はそのまま保持
        if not line:
            normalized_lines.append('')
            i += 1
            continue
        
        # 1文字だけの行が連続している場合は結合
        if len(line) == 1 and i + 1 < len(lines):
            combined_chars = [line]
            j = i + 1
            
            # 設定値まで、または意味のある単語になるまで結合
            while j < len(lines) and len(combined_chars) < max_combine:
                next_line = lines[j].strip()
                if len(next_line) == 1:
                    combined_chars.append(next_line)
                    j += 1
                elif not next_line:  # 空行で区切り
                    break
                else:
                    # 1文字でない行が来たら結合終了
                    break
            
            # 結合した文字列を作成
            combined_text = ''.join(combined_chars)
            
            # 意味のある単語（設定値以上）になった場合は結合
            if len(combined_text) >= min_line_length:
                normalized_lines.append(combined_text)
                i = j
            else:
                # 意味のある単語にならない場合は元のまま
                normalized_lines.append(line)
                i += 1
        else:
            # 通常の行はそのまま追加
            normalized_lines.append(line)
            i += 1
    
    # 連続する空行を1つにまとめる
    result_lines = []
    prev_empty = False
    
    for line in normalized_lines:
        if not line.strip():
            if not prev_empty:
                result_lines.append('')
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False
    
    return '\n'.join(result_lines)

def extract_text_from_pdf(file: BinaryIO) -> str:
    """PDFからテキスト全体を抽出"""
    pages = _read_pdf_pages(file)
    raw_text = "\n".join(pages)
    # テキストを正規化して1文字ごとの改行問題を解決
    return _normalize_pdf_text(raw_text)

def extract_chunks_from_pdf(file: BinaryIO, prefix: str) -> List[Dict]:
    """PDFからページ単位でチャンクを抽出"""
    pages = _read_pdf_pages(file)
    chunks = []
    for i, text in enumerate(pages, 1):
        chunk_id = f"{prefix}_chunk_{i}"
        chunks.append({"id": chunk_id, "text": text.strip()})
    return chunks

class Chunk(BaseModel):
    start_row: int = Field(description="チャンクの開始行番号")
    title: str = Field(description="チャンクのタイトル")
    level: int = Field(description="このチャンクが何階層目か ※目次など本文ではない場合は999を設定してください")

class ChunkBreaksOutput(BaseModel):
    """チャンク分割の結果を表すPydanticモデル"""
    chunk_reason: str = Field(description="チャンク分割の理由")
    chunks: List[Chunk] = Field(description="チャンクのリスト")

def extract_abstract(text: str) -> str:
    """ドキュメントの概要抽出"""
    llm = get_chat_llm(
        model=OPENAI_CONFIG["model"], 
        temperature=OPENAI_CONFIG["temperature"]
    )
    prompt = f"""
    以下のドキュメントがわかる範囲でどのようなものか概要を日本語で説明してください。
    内容だけでなく、章節条などの階層構造も分かるようであれば抽出してください。
    回答フォーマット
    概要：XXXXXX
    階層構造：<章節条などの構成の概要 例：このドキュメントの階層構造は、第1階層は章で、第2階層は節で、第3階層は条である>
    ドキュメントの先頭数ページ抜粋:
    {text}
    """
    return llm.invoke(prompt)

def extract_chunks(text: str, prefix: str) -> List[Dict]:
    # LLMを使ってチャンク化
    # 正規化されたテキストを適切に行分割
    lines = text.split("\n")
    
    # 設定値を取得
    min_line_length = PDF_CONFIG["text_normalization"]["min_line_length"]
    
    # 空行や極端に短い行を除外して意味のある行のみを対象とする
    meaningful_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and len(stripped_line) >= min_line_length:  # 設定値以上の行のみ
            meaningful_lines.append(stripped_line)
        elif stripped_line and len(stripped_line) < min_line_length:
            # 短い行は前の行と結合を試みる
            if meaningful_lines:
                meaningful_lines[-1] += stripped_line
            else:
                meaningful_lines.append(stripped_line)
    
    # 行番号を付与
    text_with_line_numbers = [f"row_{i+1}: {line}" for i, line in enumerate(meaningful_lines)]
    
    chunk_breaks = []
    all_chunk_info = []  # 全チャンクの詳細情報を保存
    processed_chunks = []  # これまでの抽出結果を保存
    last_chunk_break = 1
    abstract = extract_abstract(text).content
    
    while text_with_line_numbers:
        # last_chunk_breakから設定値の行数までをチャンクにする
        chunk = text_with_line_numbers[last_chunk_break-1:last_chunk_break+PDF_CONFIG["max_chunk_processing"]]
        
        # これまでの抽出結果をフォーマット
        previous_chunks_info = ""
        if processed_chunks:
            previous_chunks_info = "\n\n**これまでの抽出結果:**\n"
            previous_chunks_info += str(processed_chunks)
            previous_chunks_info += "\n\n**これまでの抽出結果を参考にして、一貫性のある階層構造とタイトル付けを行ってください。**"

        # LLMに意味の塊の境界を検出させる
        system_prompt = f"""
            あなたの仕事は、与えられたドキュメントを**正確に解析して構造を理解する**ことです。
            日本語で丁寧かつ厳密に指示に従ってください。
            与えられたテキストは、以下のドキュメントの一部です。
            {abstract}
            {previous_chunks_info}
        """
        
        chunk_size_min, chunk_size_max = PDF_CONFIG["chunk_size_range"]
        user_prompt = f"""
            各チャンクの開始行番号、そのチャンクのタイトル、そのチャンクが何階層目かを配列で返してください。
            
            #ドキュメント:
            {"\n".join(chunk)}

            #制約
            - 1チャンクあたりの文字数: {chunk_size_min}～{chunk_size_max}文字
            - 抽象構文木を考慮し、意味、階層構造の切れ目で分割する
            - PDFから抽出した文章のためページ跨ぎでヘッダーやフッターがある場合があるが、それに影響されずに意味の切れ目で分割してください。
            - 見出しや番号付きリストは新しいチャンクの開始点とする
            - 文章や階層構造の途中で切れないようにする
            - 目次と思われる箇所は分割せずに1つのチャンクにまとめてください。
        """

        llm = get_chat_llm(
            model=OPENAI_CONFIG["model"], 
            temperature=OPENAI_CONFIG["temperature"]
        )

        structured_llm = llm.with_structured_output(ChunkBreaksOutput)
        result = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # result.chunksから開始行番号を抽出し、全チャンク情報を保存
        chunk_start_rows = [chunk.start_row for chunk in result.chunks]
        chunk_breaks.extend(chunk_start_rows[:-1])
        
        # 全チャンク情報を保存（最後のchunkは次の処理に持ち越されるので除外）
        for chunk_info in result.chunks[:-1]:
            all_chunk_info.append({
                'start_row': chunk_info.start_row,
                'title': chunk_info.title,
                'level': chunk_info.level
            })
        
        # 処理済みチャンク情報を蓄積（最後のchunkは次の処理に持ち越されるので除外）
        for chunk_info in result.chunks[:-1]:
            processed_chunks.append({
                'start_row': chunk_info.start_row,
                'title': chunk_info.title,
                'level': chunk_info.level
            })
        
        last_chunk_break = chunk_start_rows[-1] if chunk_start_rows else last_chunk_break
        if len(chunk) < PDF_CONFIG["max_chunk_processing"]:
            break

    # all_chunk_infoをファイルに出力（キャッシュディレクトリ配下）
    cache_dir = CACHE_CONFIG.get("cache_directory", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    all_info_path = os.path.join(cache_dir, f"{prefix}_all_chunk_info.txt")
    with open(all_info_path, "w", encoding="utf-8") as f:
        f.write(str(all_chunk_info))

    # 各チャンクのテキストを上位階層と結合して作成
    res_chunks = []
    for i in range(len(chunk_breaks)-1):
        current_chunk_info = all_chunk_info[i] if i < len(all_chunk_info) else None
        
        # 下位レベルのチャンクが存在するかチェック
        has_sub_chunks = False
        if current_chunk_info:
            current_level = current_chunk_info['level']
            # 現在のチャンクより後のチャンクを確認
            for j in range(i+1, len(all_chunk_info)):
                next_chunk_info = all_chunk_info[j]
                if next_chunk_info['level'] > current_level:
                    # 下位レベル（より大きなレベル値）のチャンクが存在
                    has_sub_chunks = True
                    break
                elif next_chunk_info['level'] <= current_level:
                    # 同じレベル以上のチャンクが見つかったら、下位レベルのスコープ終了
                    break
        
        # 下位レベルのチャンクが存在する場合は除外
        if has_sub_chunks:
            continue
        
        # 現在のチャンクのテキスト
        current_text = "\n".join(text_with_line_numbers[int(chunk_breaks[i])-1:int(chunk_breaks[i+1])-1])
        # 行番号プレフィックスを削除
        current_text = re.sub(r'^row_\d+: ', '', current_text, flags=re.MULTILINE)
        
        # 上位階層のチャンクを収集
        if current_chunk_info:
            current_level = current_chunk_info['level']
            combined_text_parts = []
            
            # 現在のチャンクより前のチャンクを逆順で確認
            for j in range(i-1, -1, -1):
                prev_chunk_info = all_chunk_info[j]
                if prev_chunk_info['level'] < current_level:
                    # 上位階層のチャンクを見つけた場合
                    prev_start = chunk_breaks[j] if j < len(chunk_breaks) else 1
                    prev_end = chunk_breaks[j+1] if j+1 < len(chunk_breaks) else chunk_breaks[j] + 50
                    prev_text = "\n".join(text_with_line_numbers[int(prev_start)-1:int(prev_end)-1])
                    # 行番号プレフィックスを削除
                    prev_text = re.sub(r'^row_\d+: ', '', prev_text, flags=re.MULTILINE)
                    
                    # 上位階層のプレフィックスを追加
                    hierarchy_prefix = f"[上位階層{prev_chunk_info['level']}] {prev_chunk_info['title']}"
                    prefixed_text = f"{hierarchy_prefix}\n{prev_text}"
                    
                    combined_text_parts.insert(0, prefixed_text)
                    current_level = prev_chunk_info['level']
            
            # 上位階層のテキスト + 現在のチャンクのテキストを結合
            if combined_text_parts:
                final_text = "\n\n".join(combined_text_parts + [current_text])
            else:
                final_text = current_text
        else:
            final_text = current_text
        
        res_chunks.append({"id": f"{prefix}_chunk_{len(res_chunks)+1}", "text": final_text})
        
    # ファイル出力（キャッシュディレクトリ配下）
    chunks_path = os.path.join(cache_dir, f"{prefix}_chunks.txt")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in res_chunks:
            f.write(f"#id: {chunk['id']} ------------------------------\n")
            f.write(chunk["text"])
            f.write("\n--------------------------------\n\n")
    
    return res_chunks

def extract_chunks_by_headings(file: BinaryIO, prefix: str) -> List[Dict]:
    """
    LLMを使った意味の塊でのチャンク化
    """
    try:
        # PDFからテキスト全体を抽出
        text = extract_text_from_pdf(file)
        
        if not text.strip():
            return []
        
        # LLMプロバイダ判定: Azure/OpenAI いずれかの資格情報があるか
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_azure = bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

        if has_openai or has_azure:
            result = extract_chunks(text, prefix)

            # プレフィックスを追加してIDを調整
            chunks = []
            for chunk in result:
                chunks.append({
                    "id": chunk['id'], 
                    "text": chunk['text']
                })
            
            return chunks
        else:
            return extract_chunks_by_headings_fallback(file, prefix)
            
    except Exception as e:
        logging.error(f"AIチャンク化でエラーが発生: {e}")
        # フォールバック: 見出しベースの分割
        return extract_chunks_by_headings_fallback(file, prefix)

def extract_chunks_by_headings_fallback(file: BinaryIO, prefix: str) -> List[Dict]:
    """
    フォールバック用の見出しベースチャンク化（従来の実装）
    """
    chunks = []
    
    with pdfplumber.open(file) as pdf:
        current_chunk = {
            "id": f"{prefix}_chunk_1", 
            "text": "", 
            "heading": "導入部"
        }
        chunk_counter = 1
        
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            lines = page_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 見出し検出
                if is_heading_line(line):
                    # 前のチャンクを保存
                    if current_chunk["text"].strip():
                        chunks.append({
                            "id": current_chunk["id"],
                            "text": current_chunk["text"].strip()
                        })
                    
                    # 新しいチャンクを開始
                    chunk_counter += 1
                    current_chunk = {
                        "id": f"{prefix}_chunk_{chunk_counter}",
                        "text": line + "\n",
                        "heading": line[:50] + "..." if len(line) > 50 else line
                    }
                else:
                    current_chunk["text"] += line + "\n"
        
        # 最後のチャンクを追加
        if current_chunk["text"].strip():
            chunks.append({
                "id": current_chunk["id"],
                "text": current_chunk["text"].strip()
            })
    
    return chunks

def is_heading_line(line: str) -> bool:
    """
    設定ファイルの正規表現パターンによる見出し判定
    """
    for pattern in PDF_CONFIG["heading_patterns"]:
        if re.match(pattern, line):
            return True
    
    # 追加判定（短い行 + 記号を含む）
    if (len(line) < 50 and                    # 短い行
        not line.endswith('。') and           # 句点で終わらない  
        any(char in line for char in '■◆●○□◇★☆▲△')):  # 記号を含む
        return True
        
    return False 