"""統一されたエラーハンドリング"""

import logging
from typing import Optional, Any, Callable
from .structured_models import AnalysisResult
from .config import LOGGING_CONFIG

def setup_logging(name: str) -> logging.Logger:
    """統一されたログ設定"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            LOGGING_CONFIG["format"],
            datefmt=LOGGING_CONFIG["date_format"]
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    return logger

def create_error_analysis_result(error_message: str, 
                               error_type: str = "変更") -> AnalysisResult:
    """エラー時の統一されたフォールバック結果"""
    processed_error_message = str(error_message)[:100].strip() if error_message else "不明なエラー"
    summary_text = f"処理エラー: {processed_error_message}"
    if not processed_error_message or processed_error_message.isspace(): # error_messageが空または空白のみの場合
        summary_text = "処理エラー: 詳細不明"

    return AnalysisResult(
        change_type=error_type,
        summary=summary_text,
        detailed_analysis=f"処理中にエラーが発生しました: {error_message}",
        main_changes=["処理エラー"],
        correspondence_details="エラーのため対応関係を特定できませんでした"
    )

def safe_execute(func: Callable, *args, fallback_result=None, logger=None, **kwargs):
    """安全な関数実行（統一されたエラーハンドリング）"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"関数 {func.__name__} でエラー: {e}", exc_info=True)
        
        if fallback_result is not None:
            return fallback_result
        elif hasattr(func, '__annotations__') and func.__annotations__.get('return') == AnalysisResult:
            return create_error_analysis_result(str(e))
        else:
            return None

def log_performance(func: Callable):
    """パフォーマンス測定デコレータ"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logging(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} 実行時間: {end_time - start_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} エラー (実行時間: {end_time - start_time:.2f}秒): {e}")
            raise
    
    return wrapper

class ErrorContext:
    """エラーコンテキスト管理"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or setup_logging(__name__)
        self.errors = []
    
    def __enter__(self):
        self.logger.info(f"{self.operation_name} 開始")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"{self.operation_name} でエラー: {exc_val}")
            self.errors.append(str(exc_val))
        else:
            self.logger.info(f"{self.operation_name} 完了")
        return False  # 例外を再発生させる
    
    def add_error(self, error_message: str):
        """エラーメッセージを追加"""
        self.errors.append(error_message)
        self.logger.warning(f"{self.operation_name}: {error_message}")
    
    def has_errors(self) -> bool:
        """エラーがあるかチェック"""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """エラーサマリーを取得"""
        if not self.errors:
            return "エラーなし"
        return f"{len(self.errors)}個のエラー: " + "; ".join(self.errors[:3])

def validate_input(data: Any, expected_type: type, field_name: str = "data") -> bool:
    """入力データの検証"""
    if not isinstance(data, expected_type):
        raise ValueError(f"{field_name} は {expected_type.__name__} 型である必要があります。実際: {type(data).__name__}")
    return True

def validate_chunks(chunks: list, prefix: str = "chunks") -> bool:
    """チャンクデータの検証"""
    if not isinstance(chunks, list):
        raise ValueError(f"{prefix} はリスト型である必要があります")
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"{prefix}[{i}] は辞書型である必要があります")
        
        if "id" not in chunk:
            raise ValueError(f"{prefix}[{i}] に 'id' フィールドがありません")
        
        if "text" not in chunk and "content" not in chunk:
            raise ValueError(f"{prefix}[{i}] に 'text' または 'content' フィールドがありません")
    
    return True 