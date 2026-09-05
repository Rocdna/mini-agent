"""
rrf — 混合检索的融合算法（Reciprocal Rank Fusion，倒数排名融合）

背景：BM25(稀疏) 给的分数是无上界的"0~几"，稠密(余弦) 给的是"-1~1"。
两套分数的量纲不同、单位不同，直接相加没意义（会被量纲大的一边主导）。

RRF 的解法：丢掉分数，只保留"名次"。把每个 chunk 在各自检索器里的排名
套进 1/(k + rank) 求和——谁在两个检索器里都排得靠前，谁的融合分就高。

    融合分(c) = Σ_i  1 / (k + rank_i(c))
              └ 第 i 个检索器       └ 它在第 i 个检索器里的名次(1 起)

为什么 k 取 60？这是论文/社区的经验常数，作用是把"第1名"和"第2名"的差距
拉得小一点、让"出现在不同位置"的差异不会被名次差过度放大。k 越大，名次
差对结果的惩罚越平缓。

本模块是纯算法：不碰任何检索器、不申请网 / API 调用，只实现"把多份名次并成
一份"这一个函数。这样它可单独单测、可复用到任何检索器组合上。
"""


def rrf(rankings: list[dict[int, int]], k: int | float = 60) -> list[tuple[int, float]]:
    """把多份 {chunk_index: rank} 融合成一份按融合分降序的排名。

    Args:
        rankings: 每个检索器产出的 {chunk_index: 该 chunk 的名次}。
                  名次从 1 起（第 1 高）。某 chunk 没被某检索器召回 → 不贡献，
                  即不放进该检索器的 dict 即可。
        k:        RRF 常数，默认 60。

    Returns:
        [(chunk_index, fused_score)]，按融合分从高到低排序。
        融合分 = Σ 1/(k + rank)。两个检索器都排第 1 的 chunk 分数最高。
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for idx, rank in ranking.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda t: t[1], reverse=True)