import logging
from colorama import Fore, Style, init
import os
from typing import List, Dict, Any, Tuple, Optional
init(autoreset=True)

# 定义自定义日志级别 DIALOGUE
DIALOGUE_LOG_LEVEL = 15
logging.addLevelName(DIALOGUE_LOG_LEVEL, "DIALOGUE")

class ColoredFormatter(logging.Formatter):
    """自定义彩色日志格式器."""
    def format(self, record):
        log_colors = {
            'DEBUG': Fore.BLUE,
            'INFO': Fore.GREEN,
            'DIALOGUE': Fore.MAGENTA,  # 新增 DIALOGUE 颜色
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
        color = log_colors.get(record.levelname, Fore.WHITE)
        
        # 保存原始级别名称，以便在格式化后恢复
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        formatted_record = super().format(record)
        
        # 恢复原始级别名称
        record.levelname = original_levelname
        
        return formatted_record

def setup_logger(name, log_file=None, level=logging.DEBUG): # 保持 level=logging.DEBUG
    """配置并返回一个 Logger 实例."""
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 检查是否已经配置过这个 logger
    if hasattr(logger, '_configured_level') and logger._configured_level == level:
        # 如果已经配置过，并且级别相同，检查文件处理器是否匹配（如果需要）
        if log_file:
            has_matching_file_handler = False
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
                    has_matching_file_handler = True
                    break
            if not has_matching_file_handler:
                # 清理旧的文件处理器，如果目标log_file不同
                for handler in list(logger.handlers):
                    if isinstance(handler, logging.FileHandler):
                        logger.removeHandler(handler)
                        if hasattr(handler, 'close'):
                            handler.close()
                # 重新添加新的文件处理器
                if log_file: # 确保log_file真的被提供了
                    log_dir = os.path.dirname(log_file)
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)
                    fmt_str_file = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s' # 文件日志使用非彩色格式
                    date_fmt_file = '%Y-%m-%d %H:%M:%S,%03d'
                    file_formatter = logging.Formatter(fmt_str_file, datefmt=date_fmt_file)
                    file_formatter.default_msec_format = '%s,%03d'
                    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                    fh.setLevel(level)
                    fh.setFormatter(file_formatter)
                    logger.addHandler(fh)
            return logger # 已配置过，直接返回
        elif not log_file and any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            # 如果之前有文件处理器，但现在log_file为None，也需要清理
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    if hasattr(handler, 'close'):
                        handler.close()

    # 清理已有的处理器，防止重复添加 (如果上面没有完全清理或首次配置)
    if not (hasattr(logger, '_configured_level') and logger._configured_level == level):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            if hasattr(handler, 'close'):
                handler.close()
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 设置 propagate 为 False，防止日志传播到父 logger (除非是root logger)
    logger.propagate = (name == 'root') 

    # 创建统一的格式字符串
    fmt_str_console = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s' # 控制台用彩色
    fmt_str_file = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'    # 文件用非彩色
    date_fmt = '%Y-%m-%d %H:%M:%S,%03d'
    
    # 创建控制台 Handler
    console_formatter = ColoredFormatter(fmt_str_console, datefmt=date_fmt) # 使用你的ColoredFormatter
   # console_formatter.default_msec_format = '%s,%03d' # ColoredFormatter应处理此问题或Formatter已处理
    sh = logging.StreamHandler()
    sh.setLevel(level) # 确保控制台处理器也遵循级别
    sh.setFormatter(console_formatter)
    logger.addHandler(sh)

    # 如果指定了日志文件，则创建文件 Handler
    if log_file:
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir: # 只有当log_dir非空时才创建，避免为相对路径文件名创建当前目录
            os.makedirs(log_dir, exist_ok=True)
        
        file_formatter = logging.Formatter(fmt_str_file, datefmt=date_fmt)
      # file_formatter.default_msec_format = '%s,%03d' # Formatter已处理
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    
    # 标记这个 logger 已经配置过及其级别
    logger._configured_level = level
    
    return logger



def reset_logging_system(log_file: Optional[str] = None, level=logging.DEBUG): # 接受 log_file 和 level 参数
    """
    重置整个日志系统，清除所有日志处理器和记录器，
    并设置基本配置确保一致的日志格式。
    """
    # 重置根日志记录器
    root = logging.getLogger()
    root.setLevel(level) # 使用传入的level
    
    # 移除所有处理器
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if hasattr(handler, 'close'):
            handler.close()
    
    # 清理所有其他日志记录器 (除了root)
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name != 'root': # 确保不意外地再次处理root
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()
            # 重置配置标记，以便可以重新配置
            if hasattr(logger, '_configured_level'):
                del logger._configured_level
    
    # 设置根日志记录器的基本配置，并传入 log_file 和 level
    setup_logger('root', log_file=log_file, level=level)
    
    # 特别处理transitions库的日志记录器
    transitions_logger = logging.getLogger('transitions')
    transitions_logger.setLevel(logging.INFO) # 可以根据需要调整 transitions 的级别
    
    # 确保transitions日志使用与root相同的处理器
    root_logger = logging.getLogger('root')
   # 清理transitions logger可能存在的旧处理器
    for handler in list(transitions_logger.handlers):
        transitions_logger.removeHandler(handler)
        if hasattr(handler, 'close'):
            handler.close()

    for handler in root_logger.handlers:
        # 创建处理器的副本，以避免共享状态问题
        handler_copy = None
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler_copy = logging.StreamHandler()
        elif isinstance(handler, logging.FileHandler):
            handler_copy = logging.FileHandler(handler.baseFilename, mode=handler.mode, encoding=handler.encoding)
      
    if handler_copy:
        handler_copy.setFormatter(handler.formatter)
        handler_copy.setLevel(handler.level) # 保持原处理器的级别
        transitions_logger.addHandler(handler_copy)
    
    transitions_logger.propagate = False # 确保transitions库的日志不会传播到根
    
    # 同样处理其他常见库的日志记录器
    for lib_logger_name in ['httpx', 'httpcore', 'openai', 'urllib3']:
        lib_logger = logging.getLogger(lib_logger_name)
        lib_logger.setLevel(logging.INFO) # 这些通常设置为INFO或WARNING以减少噪音
        lib_logger.propagate = False