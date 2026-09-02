"""
memory_manager — 记忆存储 + 记忆动作(执行层,纯代码,不调 LLM)

对齐 book 配套项目的分工:
  LLM 负责【分析对话】→ 决定记什么、用什么格式,
  本模块负责【落地存储】:add_memory / update_memory / delete_memory / search_memory。

同一个 manager 可服务四种格式,区别在"存进去的数据结构"(content 形状由格式 prompt 决定):
  Simple/Enhanced Notes : notes 列表,每条是一个字符串
  JSON Cards            : cards 列表,每条是 {category, subcategory, key, value}
  Advanced JSON Cards   : cards 列表,每条带 {category, ..., backstory, person, relationship, timestamp}

本模块不关心"怎么从对话提炼",只关心"怎么把 LLM 给的事实存起来,检索时取回"。
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryItem:
    """一条记忆。Notes 用 content;Cards 用 category/subcategory/key/value 及扩展字段。"""
    memory_id: str = ""
    content: str = ""
    memory_type: str = "note"          # note | card | advanced_card
    category: str = ""
    subcategory: str = ""
    key: str = ""
    value: str = ""
    # Advanced JSON Cards 独有
    backstory: str = ""
    person: str = ""
    relationship: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryManager:
    """记忆的存储与增删改查。持久化到本地 JSON 文件(对齐 book)。

    设计(抄 book 配套 NotesMemoryManager):
      - 每个 user_id 一个 JSON 文件,存进 MEMORY_DIR
      - 构造时 load_memory() 读回磁盘已有记忆(跨会话保留)
      - 每次 add/update/delete 后 save_memory() 落盘(内存与磁盘实时同步)
      - save 用「先写 .tmp 再 os.replace」原子替换,崩溃不截断唯一副本
      - memory_id 用 uuid,避免按长度索引在存盘后再次 append 时撞 ID
    """

    def __init__(self, user_id: str = "test", storage_dir: str = "", filename_key: str = ""):
        self.user_id = user_id
        self.items: list[MemoryItem] = []
        # 目标目录缺省用本文件同级的 data/；可通过 storage_dir 覆盖
        self.storage_dir = storage_dir or str(
            Path(__file__).resolve().parent / "data"
        )
        # filename_key 允许把格式/子空间掺进文件名，避免不同实验串台
        # 缺省 = 仅 user_id（真实 agent"一个用户一份记忆"）；实验时传格式名。
        fname = f"{user_id}_{filename_key}_memory.json" if filename_key \
            else f"{user_id}_memory.json"
        self.memory_file = str(Path(self.storage_dir) / fname)
        self.load_memory()

    # ── 动作实现(被工具调用)──
    def add_memory(self, content: str = "", memory_type: str = "note", **kw) -> str:
        """新增一条记忆。返回 memory_id。写入后自动落盘。"""
        item = MemoryItem(content=content, memory_type=memory_type, **kw)
        item.memory_id = f"{memory_type}_{uuid.uuid4().hex[:8]}"
        self.items.append(item)
        self.save_memory()
        return item.memory_id

    def update_memory(self, memory_id: str, content: str = "", **kw) -> str:
        """按 memory_id 更新已有记忆。找不到返回错误说明。"""
        for it in self.items:
            if it.memory_id == memory_id:
                if content:
                    it.content = content
                for k, v in kw.items():
                    if hasattr(it, k):
                        setattr(it, k, v)
                it.timestamp = datetime.now().isoformat(timespec="seconds")
                self.save_memory()
                return f"updated {memory_id}"
        return f"error: 找不到 memory_id={memory_id}"

    def delete_memory(self, memory_id: str) -> str:
        """删除一条记忆。删除后自动落盘。"""
        before = len(self.items)
        self.items = [it for it in self.items if it.memory_id != memory_id]
        if len(self.items) < before:
            self.save_memory()
            return f"deleted {memory_id}"
        return f"error: 找不到 memory_id={memory_id}"

    def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        """按关键词检索记忆(简单子串匹配,纯代码)。返回 dict 列表。"""
        q = query.lower()
        hits = [it for it in self.items if q in (it.content + it.category + it.key + it.value).lower()]
        return [it.to_dict() for it in hits[:limit]]

    # ── 检索给 Agent 用(转成可读字符串)──
    def context_string(self, limit: int = 20) -> str:
        """把记忆转成一段文本,供注入 agent 上下文。"""
        if not self.items:
            return "(无记忆)"
        lines = []
        for it in self.items[:limit]:
            if it.memory_type in ("note",):
                lines.append(f"- {it.content}")
            else:
                lines.append(f"- [{it.category}.{it.subcategory}.{it.key}] {it.value}"
                             + (f" (person={it.person}, backstory={it.backstory})" if it.backstory else ""))
        return "\n".join(lines)

    def to_json(self) -> str:
        """序列化成 JSON,便于存盘/复查。"""
        import json
        return json.dumps([it.to_dict() for it in self.items], ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MemoryManager":
        mgr = cls()
        for d in json.loads(text):
            it = MemoryItem(**d)
            mgr.items.append(it)
        return mgr

    # ── 持久化(对齐 book：原子写 + 启动读回)──
    def save_memory(self) -> None:
        """把当前记忆原子地写进 JSON 文件。

        注：MemoryItem 多字段由 to_dict() 统一序列化，storage 结构是
        {"user_id", "updated_at", "items": [...]}。先写 .tmp 再 os.replace，
        崩溃/断电不会截断唯一的旧副本。
        """
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            payload = {
                "user_id": self.user_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "items": [it.to_dict() for it in self.items],
            }
            tmp = self.memory_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.memory_file)
        except Exception as e:
            # 落盘失败不能炸掉上层流程，记一条错误让调用方知道即可
            print(f"⚠ 记忆落盘失败: {e}", file=sys.stderr)

    def load_memory(self) -> None:
        """启动时从 JSON 文件读回已有记忆。文件不存在则从空开始。"""
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, encoding="utf-8") as f:
                data = json.load(f)
            self.items = [MemoryItem(**d) for d in data.get("items", [])]
        except Exception as e:
            # 文件损坏也别让整个记忆系统起不来——宁可空,不崩
            self.items = []
            print(f"⚠ 记忆加载失败(已重置为空): {e}", file=sys.stderr)