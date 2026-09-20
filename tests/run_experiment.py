"""
run_experiment — 方案B：实验脚本。跑 book 评估集，用 LLM 提炼对话 → 工具落地 → 检索验证。

工作流（对齐 book 的「LLM 分析 → 调工具 → memory_manager 落地」）：
  1. loader 读一个 test case 的对话历史
  2. 把【单段对话】压成文本，撞进格式专属 prompt（system 消息）
  3. run_agent_loop 带着 memory 工具跑：LLM 读对话 → 调 add_memory/update/delete → 存储
  4. 展示生成的记忆，并与 gold_facts 关键词比对命中率

粒度设计：一次跑【一个 case 的一段 conversation】。
  - book 每个 case 含多段 conversation，每段几十轮消息；
  - 整 case 全塞会超 deepseek-v4-flash 的安全输入长度（>2000 字符偶发空返回），
    按单段对话喂最稳，也符合"一段对话提炼一批记忆"的自然粒度。

用法：
  python -m partical.tests.run_experiment --layer layer1 --case layer1_01 --conv 0 --format simple_notes
  # --dry 只打印将送入 LLM 的消息，不调 API（先审查，再放行）
  python -m partical.tests.run_experiment --layer layer1 --case layer1_01 --format simple_notes --dry
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 utf-8 输出避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from partical.agent_loop import run_agent_loop
from partical.memory import loader
from partical.memory.formats.simple_notes import SimpleNotes  # 复用其 _flatten 压对话
from partical.memory.memory_manager import MemoryManager
from partical.memory.prompts import PROMPTS
from partical.memory.tools import build_memory_tools


def _history_text(conv: dict) -> str:
    """把单段对话压成文本历史（复用 SimpleNotes 的 _flatten，按轮次+角色排版）。"""
    return SimpleNotes._flatten([conv])


def _collect(results: list[dict], mgr: MemoryManager):
    """从 run_agent_loop 的事件流里收集工具落地结果 + 最终输出。非 LLM，纯收。"""
    for ev in results:
        t = ev.get("type")
        if t == "tool_start":
            print(f"  🔧 {ev['name']}: {ev['arguments']}")
        elif t == "tool_result":
            print(f"     ↳ {ev['result']}")
        elif t == "error":
            print(f"  ⚠ {ev['message']}")


def _normalize(text: str) -> str:
    """对齐 book metrics._normalize：小写 + 压缩空白，用于宽容的子串匹配。"""
    import re
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _fact_matched(fact, haystack: str) -> bool:
    """fact 可能是 str 或 list(any-of 可接受写法任务，命中其一即算)。"""
    variants = fact if isinstance(fact, list) else [fact]
    return any(_normalize(v) in haystack for v in variants)


def evaluate_against_gold(test_id: str, mgr: MemoryManager):
    """用 book 的 KeywordRecall 口径评估：gold 事实命中率 + 0.8 及格线。

    纯子串匹配，不调 LLM。haystack 是当前 manager 里全部记忆拼成的文本。
    """
    import re
    gold = loader.load_gold_facts()
    facts = gold.get(test_id, {}).get("required_facts", []) if isinstance(
        gold.get(test_id), dict) else gold.get(test_id, [])

    # haystack 用 context_string() 拼(同时含 note 的 content 和 card 的 key/value)，
    # 而不是只拼 content——json_cards 的数据在 value 里，只看 content 会漏。
    haystack = _normalize(mgr.context_string())
    print("\n── 评估（book KeywordRecall 口径）──")

    if not facts:
        print("  该 case 无 gold facts 定义，跳过评估。")
        return

    matched = 0
    missing = []
    for fact in facts:
        label = fact[0] if isinstance(fact, list) else str(fact)
        label = label + " (任一)" if isinstance(fact, list) else label
        hit = _fact_matched(fact, haystack)
        matched += int(hit)
        print(f"  {'✅' if hit else '❌'} {label}")
        if not hit:
            missing.append(label)

    recall = matched / len(facts)
    passed = recall >= 0.8
    verdict = "passed ✅" if passed else "FAILED ❌ (线=0.8)"
    print(f"\n  recall = {matched}/{len(facts)} = {recall:.2f}   → {verdict}")
    if missing:
        print(f"  missing: {', '.join(missing)}")


def summarize(mgr: MemoryManager):
    """把 memory_manager 里的记忆打印成可读文本。

    复用 context_string()：它在 memory_manager 里已正确区分
    note(content) 与 card(category.subcategory.key = value)，
    避免这里再为每种格式写一套拼串逻辑。
    """
    print("\n──────────────── 生成记忆 ────────────────")
    if not mgr.items:
        print("(空)")
        return
    html = mgr.context_string()
    print("\n".join("  " + ln for ln in html.splitlines()))


async def run(case: dict, conv_idx: int, format_name: str, dry: bool):
    convs = case.get("conversation_histories", [])
    if not convs:
        print(f"⚠ case {case['test_id']} 无对话历史")
        return
    if conv_idx >= len(convs):
        print(f"⚠ case 只有 {len(convs)} 段 conversation（指定 --conv {conv_idx} 超界）")
        return

    conv = convs[conv_idx]
    history = _history_text(conv)

    # ── 组装消息：system = 格式专属指令 + 对话历史 ──
    prompt = PROMPTS[format_name].format(history=history)
    messages = [{"role": "system", "content": prompt}]

    # ── 落地层：memory_manager + 工具(按格式分文件，避免不同实验串台)──
    mgr = MemoryManager(user_id=case["test_id"], filename_key=format_name)
    tool_executors = build_memory_tools(mgr)

    print(f"════ 实验: {case['test_id']}  格式: {format_name}  第 {conv_idx} 段对话 ════")
    print(f"对话轮次: {len(conv.get('messages', []))}  字符数: {len(history)}")

    if dry:
        print("\n┄┄┄ dry 模式：下面是即将送入 LLM 的 system 消息全文（未调 API）┄┄┄\n")
        print(prompt)
        print("\n┄┄┄ 工具可用：add_memory / update_memory / delete_memory ┄┄┄")
        print("[dry] 未调用 API。确认无误后去掉 --dry 运行。")
        return

    # ── 真正跑：run_agent_loop（关掉 status_bar/compressor，避免污染工具结果）──
    async for ev in run_agent_loop(
        messages,
        tool_executors,
        max_turns=15,
        use_status_bar=False,
        use_compressor=False,
    ):
        _collect([ev], mgr)

    summarize(mgr)
    evaluate_against_gold(case["test_id"], mgr)


# ═══════════════════════════════════════════════════════════
# ▶ 想改实验目标,只动下面这几行,然后直接 python run_experiment.py
# ═══════════════════════════════════════════════════════════
RUN_LAYER  = "layer1"          # layer1 / layer2 / layer3
RUN_CASE   = "layer1_01"       # 测哪个 case?见下面 CASE_ID 列表
RUN_CONV   = 0                 # 用这个 case 的第几段对话(0 起)
RUN_FORMAT = "json_cards"    # simple_notes / enhanced_notes / json_cards / advanced_json_cards
# ═══════════════════════════════════════════════════════════

# 常用 case id 速查(每层 20 个,这里列 layer1 全部)
CASE_ID = """layer1_01_bank_account   layer1_02_insurance_claim   layer1_03_medical_appointment
layer1_04_airline_booking      layer1_05_internet_service    layer1_06_credit_card_app
layer1_07_car_rental           layer1_08_hotel_reservation   layer1_09_home_security
layer1_10_pharmacy_transfer    layer1_11_mortgage_application layer1_12_gym_membership
layer1_13_tax_preparation     layer1_14_cellphone_upgrade   layer1_15_college_enrollment
layer1_16_home_renovation     layer1_17_veterinary_care     layer1_18_retirement_planning
layer1_19_wedding_venue       layer1_20_daycare_enrollment"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="跑 book 记忆评估集的一个 case")
    ap.add_argument("--layer", default=RUN_LAYER, help="layer1/layer2/layer3")
    ap.add_argument("--case", default=RUN_CASE, help="test_id，如 layer1_01（或只写编号 01）")
    ap.add_argument("--conv", type=int, default=RUN_CONV, help="第几段 conversation（0 起）")
    ap.add_argument("--format", dest="format_name", default=RUN_FORMAT,
                    help=PROMPTS_PRETTY)
    ap.add_argument("--dry", action="store_true", help="只打印将送入的消息，不调 API")
    ap.add_argument("--list", action="store_true", help="列出指定 layer 的所有 case id")
    args = ap.parse_args(argv)

    if args.list:
        for c in loader.load_layer(args.layer):
            print(c["test_id"])
        return

    if args.format_name not in PROMPTS:
        print(f"格式必须是: {list(PROMPTS.keys())}")
        return

    if not args.case:
        print("缺少 --case（或拼写错误）。用 --list 看可用 case id。")
        return

    case_id = args.case if args.layer in args.case else f"{args.layer}_{args.case}"
    case = next(
        (c for c in loader.load_layer(args.layer)
         if c["test_id"] == case_id or c["test_id"].startswith(case_id)),
        None,
    )
    if case is None:
        print(f"找不到 case: {case_id}（在 {args.layer} 里）")
        return

    asyncio.run(run(case, args.conv, args.format_name, args.dry))


PROMPTS_PRETTY = "simple_notes | enhanced_notes | json_cards | advanced_json_cards"


if __name__ == "__main__":
    main()