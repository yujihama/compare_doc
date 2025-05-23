import PyPDF2
import pdfplumber
import re
from typing import BinaryIO, List, Dict

def extract_text_from_pdf(file: BinaryIO) -> str:
    reader = PyPDF2.PdfReader(file)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def extract_chunks_from_pdf(file: BinaryIO, prefix: str) -> List[Dict]:
    reader = PyPDF2.PdfReader(file)
    chunks = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        chunk_id = f"{prefix}_chunk_{i}"
        chunks.append({"id": chunk_id, "text": text.strip()})
    return chunks

def extract_chunks_by_headings(file: BinaryIO, prefix: str) -> List[Dict]:
    """
    見出しベースでチャンク化
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
    正規表現パターンによる見出し判定
    """
    heading_patterns = [
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
        r'^第\d+条\s+.+',         # "第1条 定義"
        r'^第\d+節\s+.+',         # "第1節 総則"
        r'^附則\s*\d*\s*.+',      # "附則 経過措置"
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, line):
            return True
    
    # 追加判定（短い行 + 記号を含む）
    if (len(line) < 50 and                    # 短い行
        not line.endswith('。') and           # 句点で終わらない  
        any(char in line for char in '■◆●○□◇★☆▲△')):  # 記号を含む
        return True
        
    return False 