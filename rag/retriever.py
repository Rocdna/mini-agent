"""
retriever — BM25 稀疏检索（第 3 章实验 3-5）

对 chunker 切出的文档建立倒排索引，用 BM25 打分排序，回答"用户问一句，哪几个 chunk 相关"。

BM25 的核心直觉（见 book 3-5）：
  1. TF 词频      —— 一个词在 chunk 里出现越多越相关，但边际递减
  2. IDF 逆文档频率 —— 一个词在整个知识库越稀有越值钱（"蒸馏"比"模型"更有区分度）
  3. 长度归一     —— 同样命中一次，短 chunk 里的词比长 chunk 里的更中心

打分公式（k1=1.5, b=0.75，book 标准参数）：
    Score(Q,D) = Σ_{q in Q} IDF(q) · TF(q,D)·(k1+1) / (TF(q,D) + k1·(1 - b + b·|D|/avgdl))
    IDF(q) = ln((N - df(q) + 0.5) / (df(q) + 0.5))    # df=含该词的文档数, 越稀有越大

中文分词用 jieba（你的语料是中文）。RETRIEVER 与 chunker 一样零"LLM API"调用。
"""

import math
import re

import jieba

# 中文停用词：高频但在检索里几乎不携带区分度的词，去掉能显著提精度。
# 只挑对"知识库检索"真正无用的；保留"不""能"等可能承载语义的词。
STOP_WORDS = set("""的 了 是 在 我 你 他 她 它 这 那 也 都 就 和 与 及 或 而 但 被 把 让 对 到 于 从 由 之 其 且 如 若 可 并 又 很 更 最 才 则 仍 所""")

# Markdown 语法字符 + 标点 + 空白，分词后剔除
# 用双引号包字符串，避免内部撇号(单引号)提前结束字符串
_PUNC = re.compile(r"^[\s#*\-_、。，、；：！？!?()（）\[\]【】\"\'·~`|/\\0-9.%]+$")


def tokenize(text: str) -> list[str]:
    """用 jieba 把中文文本切成词，去掉停用词和纯符号 token。

    例：'退款多久到账' -> ['退款', '多久', '到账']
    """
    if not text:
        return []
    toks = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w or w in STOP_WORDS:
            continue
        if _PUNC.match(w):
            continue  # 纯标点/符号，无检索价值
        toks.append(w)
    return toks


class BM25Index:
    """倒排索引 + BM25 打分。

    倒排索引（Inverted Index）把"文档 → 词"的打字反过来，变成"词 → 文档"的查表，
    让查询时不用扫全部文档，而是立刻定位到含该词的文档。

    结构：
      self.doc_freq[word]            = 含该词的文档数 df(word)
      self.postings[word]            = {doc_index: tf}   该词在哪些文档出现、各出现几次
      self.doc_len[doc_index]        = 该文档的 token 总数
      self.docs[doc_index]           = 原始 chunk（返回用）
    """

    def __init__(self, tokens_by_doc: list[list[str]]):
        """tokens_by_doc: list of token lists，每个 chunk 切成的一串词。"""
        self.docs = [None] * len(tokens_by_doc)
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_freq: dict[str, int] = {}
        self.doc_len = [0] * len(tokens_by_doc)

        for doc_idx, toks in enumerate(tokens_by_doc):
            self.doc_len[doc_idx] = len(toks)
            # Counter 统计每篇词频
            seen = set()
            for t in toks:
                if t not in self.postings:
                    self.postings[t] = {}
                self.postings[t][doc_idx] = self.postings[t].get(doc_idx, 0) + 1
                seen.add(t)
            # df 用"含该词的文档数"，一个词在一篇里只计一次
            for t in seen:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        n = len(tokens_by_doc)
        self.avgdl = (sum(self.doc_len) / n) if n else 1.0
        self.n = n

    def idf(self, word: str) -> float:
        """BM25 IDF：越稀有越大。公式见模块 docstring。"""
        df = self.doc_freq.get(word, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5)) if df else 0.0

    def score_doc(self, query_toks: list[str], doc_idx: int,
                  k1: float = 1.5, b: float = 0.75) -> float:
        """对单个文档算 BM25 总分（累加每个命中查询词的贡献）。"""
        length = self.doc_len[doc_idx]
        score = 0.0
        for q in query_toks:
            tf = self.postings.get(q, {}).get(doc_idx, 0)
            if tf == 0:
                continue
            # 词频饱和 + 长度归一 组合项（book 公式）
            denom = tf + k1 * (1 - b + b * length / self.avgdl)
            score += self.idf(q) * (tf * (k1 + 1) / denom)
        return score

    def explain(self, query: str, doc_idx: int,
                k1: float = 1.5, b: float = 0.75) -> dict:
        """诊断模式：把 BM25 在指定文档上的打分一步步拆开，逐词展示贡献。

        返回 {总得分, 文档信息, 逐词明细[词,df,IDF,TF,词频饱和项,贡献]}，
        便于对照 book 的"词X→命中N篇,IDF=...,贡献=..."日志手算复现。
        纯解释用，不改动打分逻辑。
        """
        length = self.doc_len[doc_idx]
        query_toks = tokenize(query)
        total = 0.0
        parts = []
        for q in query_toks:
            df = self.doc_freq.get(q, 0)
            tf = self.postings.get(q, {}).get(doc_idx, 0)
            idf = self.idf(q)
            if tf == 0:
                parts.append({"词": q, "df": df, "IDF": round(idf, 4),
                              "TF": 0, "贡献": 0.0, "命中?" : False})
                continue
            # 长度归一 + 词频饱和 组合项（book 公式分母）
            denom = tf + k1 * (1 - b + b * length / self.avgdl)
            contrib = idf * (tf * (k1 + 1) / denom)
            total += contrib
            parts.append({"词": q, "df": df, "IDF": round(idf, 4), "TF": tf,
                          "词频饱和项": round(tf * (k1 + 1) / denom, 4),
                          "贡献": round(contrib, 4), "命中?": True})
        chunk = self.docs[doc_idx] if doc_idx < len(self.docs) else None
        return {
            "query": query,
            "query_tokens": query_toks,
            "doc_idx": doc_idx,
            "doc": (chunk or {}).get("meta", {}).get("doc", "?"),
            "doc_len": length,
            "avgdl": round(self.avgdl, 2),
            "n": self.n,
            "total": round(total, 4),
            "parts": parts,
        }

    def search(self, query: str, top_k: int = 5, k1: float = 1.5,
               b: float = 0.75) -> list[dict]:
        """检索：切查询 → 对全部文档打分 → 返回 top_k。

        返回 [{"index", "score", "doc", "chunk_text"}]，按分数降序。
        doc 是 chunk 的 meta.doc（来源文件名），index 是 chunk 序号。
        """
        query_toks = tokenize(query)
        if not query_toks:
            return []
        scored = []
        for doc_idx in range(self.n):
            s = self.score_doc(query_toks, doc_idx, k1, b)
            if s > 0:
                scored.append((s, doc_idx))
        scored.sort(reverse=True)
        results = []
        for s, doc_idx in scored[:top_k]:
            chunk = self.docs[doc_idx]
            results.append({
                "index": doc_idx,
                "score": round(s, 4),
                "doc": (chunk or {}).get("meta", {}).get("doc", "?"),
                "chunk_text": (chunk or {}).get("text", ""),
            })
        return results


def build_index(chunks: list[dict]) -> BM25Index:
    """把 chunker 的 chunk 列表建成 BM25Index。

    chunk 形如 {"index":.., "text":.., "meta":{"doc":..}}。
    """
    tokens_by_doc = []
    docs_snapshot = []
    for c in chunks:
        tokens_by_doc.append(tokenize(c["text"]))
        docs_snapshot.append(c)
    idx = BM25Index(tokens_by_doc)
    idx.docs = docs_snapshot   # 保存原始 chunk，供 search 返回
    return idx