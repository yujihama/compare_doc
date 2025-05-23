import re
from typing import List, Dict

def chunk_document(text: str, prefix: str) -> List[Dict]:
    """
    空行でチャンク分割し、各チャンクにIDを付与。
    """
    chunks = re.split(r'\n\s*\n', text.strip())
    result = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"{prefix}_chunk_{i}"
        result.append({
            "id": chunk_id,
            "text": chunk.strip()
        })
    return result

def split_sentences(chunk_text: str) -> List[str]:
    """
    「。」で文分割（末尾に「。」を残す）。
    """
    sentences = re.findall(r'.+?。', chunk_text, flags=re.DOTALL)
    # 残りの文（「。」で終わらない）も追加
    rest = re.sub(r'.+?。', '', chunk_text, flags=re.DOTALL)
    if rest.strip():
        sentences.append(rest.strip())
    return [s.strip() for s in sentences if s.strip()] 