from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Literal

class ChunkInfo(BaseModel):
    """チャンク情報"""
    id: str = Field(description="チャンクID")
    content: str = Field(description="チャンクの内容")
    heading: Optional[str] = Field(default=None, description="見出し情報")

class CorrespondenceInfo(BaseModel):
    """対応関係情報"""
    old_chunk_ids: List[str] = Field(description="対応する旧チャンクのIDリスト")
    new_chunk_ids: List[str] = Field(description="対応する新チャンクのIDリスト")
    correspondence_type: str = Field(description="対応関係のタイプ（1:1, 1:N, N:1, N:N）")

class AnalysisResult(BaseModel):
    """分析結果の構造化モデル - OpenAI Structured Output最適化版"""
    
    # Pydantic v2の設定
    model_config = ConfigDict(
        str_strip_whitespace=True,  # 文字列の前後の空白を自動削除
        validate_assignment=True,   # 代入時のバリデーション
        extra='forbid'              # 余分なフィールドを禁止
    )
    
    # Literalを使用してEnum参照を避け、必須フィールドとして明確に定義
    change_type: Literal["変更", "形式変更", "追加", "削除", "変更なし"] = Field(
        description="変更の種類",
        examples=["変更", "形式変更", "追加", "削除", "変更なし"]
    )
    summary: str = Field(
        description="変更内容の概要（200文字程度）",
        min_length=1,
        max_length=500
    )
    detailed_analysis: str = Field(
        description="詳細な分析結果",
        min_length=1
    )
    main_changes: List[str] = Field(
        description="主な変更点のリスト",
        min_length=1,
        examples=[["項目1の変更", "項目2の追加"]]
    )
    correspondence_details: str = Field(
        description="対応関係の詳細説明",
        min_length=1
    )
    
    @field_validator('change_type', mode='before')
    @classmethod
    def validate_change_type(cls, v):
        """change_typeの値を正規化"""
        if isinstance(v, str):
            v = v.strip()
            # 許可された値のチェック
            valid_values = ["変更", "追加", "削除", "変更なし"]
            if v in valid_values:
                return v
            # 類似値のマッピング
            mapping = {
                "修正": "変更",
                "更新": "変更", 
                "改訂": "変更",
                "新規": "追加",
                "追記": "追加",
                "削除": "削除",
                "除去": "削除",
                "なし": "変更なし",
                "無し": "変更なし",
                "変化なし": "変更なし",
                "内容変更": "変更",
                "形式変更": "変更なし"
            }
            return mapping.get(v, "変更")
        return "変更"
    
    @field_validator('summary', mode='before')
    @classmethod
    def validate_summary(cls, v):
        """summaryのバリデーション"""
        if not v or not isinstance(v, str):
            return "概要を取得できませんでした"
        v = v.strip()
        if len(v) == 0:
            return "概要を取得できませんでした"
        return v
    
    @field_validator('detailed_analysis', mode='before') 
    @classmethod
    def validate_detailed_analysis(cls, v):
        """detailed_analysisのバリデーション"""
        if not v or not isinstance(v, str):
            return "詳細分析を取得できませんでした"
        v = v.strip()
        if len(v) == 0:
            return "詳細分析を取得できませんでした"
        return v
    
    @field_validator('main_changes', mode='before')
    @classmethod
    def validate_main_changes(cls, v):
        """main_changesのバリデーション"""
        if not v:
            return ["変更内容を特定できませんでした"]
        
        if isinstance(v, str):
            # 文字列の場合、改行や区切り文字で分割を試行
            if '\n' in v:
                items = [item.strip() for item in v.split('\n') if item.strip()]
            elif ',' in v:
                items = [item.strip() for item in v.split(',') if item.strip()]
            elif ';' in v:
                items = [item.strip() for item in v.split(';') if item.strip()]
            else:
                items = [v.strip()]
            return items if items else ["変更内容を特定できませんでした"]
        
        if isinstance(v, list):
            # リストの場合、各要素を文字列に変換
            filtered_items = []
            for item in v:
                if item and str(item).strip():
                    filtered_items.append(str(item).strip())
            return filtered_items if filtered_items else ["変更内容を特定できませんでした"]
        
        return ["変更内容を特定できませんでした"]
    
    @field_validator('correspondence_details', mode='before')
    @classmethod
    def validate_correspondence_details(cls, v):
        """correspondence_detailsのバリデーション"""
        if not v or not isinstance(v, str):
            return "対応関係を特定できませんでした"
        v = v.strip()
        if len(v) == 0:
            return "対応関係を特定できませんでした"
        return v

class GroupAnalysisResult(BaseModel):
    """グループ分析結果"""
    group_number: int = Field(description="グループ番号")
    group_type: str = Field(description="グループタイプ（1:1, 1:N, N:1, N:N）")
    old_chunks: List[ChunkInfo] = Field(description="旧チャンク情報")
    new_chunks: List[ChunkInfo] = Field(description="新チャンク情報")
    strength: float = Field(description="類似度強度")
    refinement_method: str = Field(description="細分化方法")
    processing_method: str = Field(description="処理方法（bypassed / gpt_analyzed）")
    analysis: AnalysisResult = Field(description="分析結果")
    correspondence: CorrespondenceInfo = Field(description="対応関係情報")
    # 階層情報を追加
    old_hierarchy: Optional[str] = Field(default=None, description="旧文書の階層情報")
    new_hierarchies: Optional[List[str]] = Field(default=None, description="新文書の階層情報リスト")

class ComparisonReport(BaseModel):
    """比較レポート全体の構造"""
    summary: Dict[str, int] = Field(description="概要統計")
    groups: List[GroupAnalysisResult] = Field(description="グループ分析結果")
    added_chunks: List[ChunkInfo] = Field(description="追加されたチャンク")
    deleted_chunks: List[ChunkInfo] = Field(description="削除されたチャンク") 