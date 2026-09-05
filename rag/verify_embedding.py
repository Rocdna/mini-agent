"""
verify_embedding — 小范围内验证豆包 embedding API 通路 + 语义捕捉能力

只调 1 次 API：对 4 句语义两两相近/相远的句子嵌入，打印余弦相似度矩阵。
预期：
  - 语义相近的两句相似度高（如「退款多久到账」vs「购买后好几天才收到钱」）
  - 语义无关的两句相似度低（如「退款到账」vs「会员积分规则」）

因为要调用付费的方舟 embedding API，默认不自动跑，等你确认后执行：
  python -m partical.rag.verify_embedding
"""

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 语义两两相近/相远的句子（构造余弦相似度矩阵用）
# 三组对照：①真正近义的两句 ②与①同主题(退款) ③明显无关(积分/包邮)
SAMPLES = [
    "订单签收后7天内可申请全额退款",           # 退款
    "买的东西不喜欢，七天之内能退钱吗",         # 退款（近义,字面无"退款"）
    "会员每消费1元累计1积分，可抵扣现金",       # 积分
    "火箭通过燃料推进飞向太空",               # 火箭（完全无关）
]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 = cos(向量夹角)。值越接近 1 越语义相近。"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


async def main():
    from partical.rag.embedded import embed_texts

    # 关键：同一次请求，批量嵌入 4 句（比逐句快、省请求）
    vecs = await embed_texts(SAMPLES)
    print(f"\nAPI 通! 返回 {len(vecs)} 个向量, 维度 = {len(vecs[0])}\n")

    # 打印两两余弦相似度矩阵
    print("       " + "".join(f"  S{i}" for i in range(len(vecs))))
    for i in range(len(vecs)):
        row = f" S{i}   "
        for j in range(len(vecs)):
            row += f"  {cosine(vecs[i], vecs[j]):.2f} "
        print(row)
    print("\n对角=1(自身), 越接近1越语义相似。看 A/A' 是否显著高于 A/其他。")


if __name__ == "__main__":
    asyncio.run(main())