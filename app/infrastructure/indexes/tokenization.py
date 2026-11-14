"""全文文档与查询共用的固定词典中文切词。变更算法后必须重新回填。"""
import re
import unicodedata

import jieba


_WORDS = re.compile(r'[a-z_][a-z0-9_]*|[0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+')
_CHINESE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
_TOKENIZER = jieba.Tokenizer()


def tokenize_search(text: str, *, for_query: bool = False) -> list[str]:
    """文档保留搜索子词，查询精确切词，避免“数据库”以 OR 退化为“数据”。"""
    tokens = []
    for match in _WORDS.finditer(unicodedata.normalize('NFKC', text).lower()):
        word = match.group()
        if _CHINESE.match(word):
            cut = _TOKENIZER.cut if for_query else _TOKENIZER.cut_for_search
            tokens.extend(cut(word, HMM=False))
        else:
            tokens.append(word)
    return tokens


def search_text(text: str) -> str:
    return ' '.join(tokenize_search(text))
