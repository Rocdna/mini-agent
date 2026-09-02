"""
tools — 记忆动作工具(add/update/delete),供 LLM 调用,纯代码执行

对齐 book:LLM 分析对话 → 调用这几个工具 → memory_manager 落地。
工具是普通函数 + __tool_schema__(同 partical.tools 风格)。

关键设计:工具的 schema 跟着【记忆格式】走(build_memory_tools 的 format_name 决定)。
  因为不同格式给 LLM 暴露的字段不同——
    simple_notes / enhanced_notes : 只要 content(一段事实 / 一段上下文)
    json_cards                   : category/subcategory/key/value
    advanced_json_cards          : 上面 + backstory/person/relationship
  这样 LLM 看到的参数形状总是和该格式的存储结构一致,不会把字段用错。

每个工具接收一个 memory_manager 实例,并通过闭包绑定给它。
"""

# 各格式 → add_memory 该暴露哪些"结构化字段"
# 只暴露当前格式需要的字段,避免 LLM 在 notes 格式乱塞 category 之类。
_FORMAT_FIELDS = {
    "simple_notes": {"content"},                      # 最小事实
    "enhanced_notes": {"content"},                    # 一段完整上下文
    "json_cards": {"category", "subcategory", "key", "value"},
    "advanced_json_cards": {"category", "subcategory", "key", "value",
                            "backstory", "person", "relationship"},
}

# 兼容缺省:未知格式也当 notes(content only)
_ALL_FIELDS = {
    "content": ("string", "记忆事实内容(= value)。Notes 格式只填这个"),
    "category": ("string", "顶级类别,如 personal"),
    "subcategory": ("string", "子类别,如 contact"),
    "key": ("string", "键,如 email"),
    "value": ("string", "值"),
    "backstory": ("string", "来源背景('为什么存')"),
    "person": ("string", "该信息涉及的主体身份('为谁')"),
    "relationship": ("string", "与用户的关系"),
}


def build_memory_tools(mgr, format_name: str = "enhanced_notes"):
    """给一个 memory_manager 实例造出记忆工具字典,返回 {name: fn}。

    工具 schema 用 partical 标准格式({"function": {...}} + parameters),
    与 run_agent_loop 兼容。format_name 决定 add_memory 暴露哪些字段。
    """
    # 依据格式决定 add_memory 对外暴露的字段集
    fields = _FORMAT_FIELDS.get(format_name, _FORMAT_FIELDS["enhanced_notes"])

    def _props(*names):
        return {n: {"type": _ALL_FIELDS[n][0], "description": _ALL_FIELDS[n][1]} for n in names}

    def __tool_schema__(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    # ── add_memory ──
    async def add_memory(args: dict) -> str:
        # 无论哪个格式,都只把 LLM 实际传了、且属于当前格式的字段交给 manager
        kwargs = {"content": args.get("content", "")}
        mapped = {
            "category": "category", "subcategory": "subcategory",
            "key": "key", "value": "value",
            "backstory": "backstory", "person": "person",
            "relationship": "relationship",
        }
        for argkey, mgrkey in mapped.items():
            if argkey in fields and argkey in args:
                kwargs[mgrkey] = args[argkey]
        return mgr.add_memory(memory_type="note", **kwargs)

    _note_desc = {
        "simple_notes": "保存一条【最小、不可再分】的用户事实,如 '用户邮箱: john@example.com'。",
        "enhanced_notes": "保存一段【保留完整上下文】的用户记忆段落,如 '用户在 TechCorp 任高级软件工程师已 3 年,领导推荐系统项目'。",
    }
    add_memory.__tool_schema__ = __tool_schema__(
        "add_memory",
        _note_desc.get(format_name, "新增一条用户记忆。") + " 返回 memory_id。",
        _props(*(["content"] + [f for f in fields if f != "content"])),
        ["content"],
    )

    # ── update_memory ──
    async def update_memory(args: dict) -> str:
        return mgr.update_memory(
            args.get("memory_id", ""),
            content=args.get("content", ""),
        )

    update_memory.__tool_schema__ = __tool_schema__(
        "update_memory",
        "按 memory_id 修改已有记忆(例如信息更正)。",
        {
            "memory_id": {"type": "string", "description": "要更新的记忆ID"},
            "content": {"type": "string", "description": "新的内容"},
        },
        ["memory_id"],
    )

    # ── delete_memory ──
    async def delete_memory(args: dict) -> str:
        return mgr.delete_memory(args.get("memory_id", ""))

    delete_memory.__tool_schema__ = __tool_schema__(
        "delete_memory",
        "删除一条记忆(例如认定该信息已过时/错误)。",
        {
            "memory_id": {"type": "string", "description": "要删除的记忆ID"},
        },
        ["memory_id"],
    )

    return {
        "add_memory": add_memory,
        "update_memory": update_memory,
        "delete_memory": delete_memory,
    }