"""
loader — 读取 book 配套评估集的对话历史（test_cases/*.yaml）

book 的每个测试用例（test_id，如 layer1_01_bank_account）是一个 yaml，
里面有:
  category   : layer1 / layer2 / layer3
  conversation_histories : list[dict]
      每个 dict: conversation_id + messages
        messages: list[{"role","content"}]
  以及 evaluation_criteria / expected（部分用例）

本模块负责把 yaml 读成统一结构，供记忆系统 generate 消费。
"""

import os
from pathlib import Path

import yaml

# book 评估集所在目录（相对本文件）
TEST_CASES_DIR = Path(__file__).resolve().parent.parent.parent / "ai-agent-book" \
    / "chapter3" / "user-memory-evaluation" / "test_cases"

# gold_facts 位置
GOLD_FACTS_PATH = TEST_CASES_DIR.parent / "fixtures" / "gold_facts.json"


def load_layer(layer: str = "layer1") -> list[dict]:
    """读取某一层所有测试用例。返回 list[dict]，每个 dict 包含:
        test_id / category / title / conversation_histories / (gold facts via fixture)
        conversation_histories: list[dict]（含 messages）
    """
    layer_dir = TEST_CASES_DIR / layer
    cases = []
    for yaml_path in sorted(layer_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cases.append(data)
    return cases


def load_all_layers() -> dict:
    """加载三层所有用例，返回 {layer: [cases...]}"""
    return {layer: load_layer(layer) for layer in ("layer1", "layer2", "layer3")}


def load_gold_facts() -> dict:
    """读取 gold_facts.json，返回 {test_id: {"required_facts": [...]}}"""
    import json
    with open(GOLD_FACTS_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # 自检：打印 layer1 第一个用例的对话长度，确认 loader 工作
    cases = load_layer("layer1")
    c = cases[0]
    n_msgs = sum(len(conv.get("messages", [])) for conv in c["conversation_histories"])
    print(f"layer1 共 {len(cases)} 个用例")
    print(f"第一个用例 {c['test_id']}：{len(c['conversation_histories'])} 段会话，共 {n_msgs} 条消息")
    gf = load_gold_facts()
    print(f"gold_facts 含 {len(gf)} 个用例的关键事实")