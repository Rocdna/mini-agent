"""
DeepSeek Tool Calling Agent — ReAct 循环核心

对应书里 chapter2/local_llm_serving 的 agent.py (VLLMToolAgent) +
ollama_native.py (OllamaNativeAgent) 两个文件，合并并简化为一个类。

与原书代码的关键区别：
1. DeepSeek 提供 OpenAI 兼容接口，所以只需把 base_url 指向
   https://api.deepseek.com/v1，其余照常用 openai SDK。
2. DeepSeek 原生支持 function calling（tools + tool_calls 字段），
   不需要 Qwen3 那套 <tool_call> XML 标签和 <tool_response> 包装：
   工具结果直接以 {"role": "tool", "tool_call_id": ...} 消息回灌。
3. 系统提示词不再需要嵌入工具 JSON —— 工具定义通过 API 的 tools 参数传。

对外接口与原书保持一致：
    chat(message, use_tools=True, stream=False)  # 非流式返回 str
    chat(message, stream=True)                   # 流式返回 chunk 生成器
    reset_conversation() / get_conversation_history() / add_custom_tool()

流式 chunk 协议（agent.py 的渲染代码依赖它）：
    {"type": "thinking" | "tool_call" | "tool_result" | "tool_error"
           | "content" | "error", "content": ...}
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from tools import ToolRegistry

# 读取仓库根目录的 .env（DEEPSEEK_API_KEY / DEEPSEEK_MODEL 在那里）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ReAct 循环最大轮次，防止模型无限调工具
MAX_ITERATIONS = 10

SYSTEM_PROMPT_WITH_TOOLS = (
    "You are a helpful assistant that can use tools to answer questions and perform tasks.\n"
    "When you need real-time or factual information (weather, time, currency rates) "
    "or need to compute something, call the appropriate tool.\n"
    "After receiving tool results, use them to provide a comprehensive answer to the user.\n"
    "If no tool is needed, answer directly."
)

SYSTEM_PROMPT_PLAIN = "You are a helpful assistant."


class DeepSeekToolAgent:
    """Agent that uses the DeepSeek API for tool calling"""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
    ):
        """
        Args:
            api_key:  DeepSeek API Key，默认从环境变量 DEEPSEEK_API_KEY 读取
            model:    模型名，默认从环境变量 DEEPSEEK_MODEL 
            base_url: API 地址，默认 https://api.deepseek.com/v1
        """
        self.api_key = api_key or DEEPSEEK_API_KEY
        # 如果 key 不存在 提示
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未设置。请在仓库根目录 .env 中配置，"
                "例如：DEEPSEEK_API_KEY=sk-xxxx"
            )

        self.model = model or DEEPSEEK_MODEL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.tool_registry = ToolRegistry()
        self.conversation_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized DeepSeekToolAgent (model={self.model})")

    # ══════════════════════════════════════════════════════════
    # 工具执行
    # ══════════════════════════════════════════════════════════
     

    # 工具调用  键值对数组
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行工具调用并返回结果。同一轮的多个调用彼此独立，并行执行。
        """
        def run_one(tool_call: Dict[str, Any]) -> Dict[str, Any]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            tool_id = tool_call["id"]

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            result = self.tool_registry.execute_tool(tool_name, tool_args)

            # 检查工具返回是否是错误结构（{"success": false, "error": ...}）
            try:
                result_dict = json.loads(result) if isinstance(result, str) else result
                if isinstance(result_dict, dict) and not result_dict.get("success", True):
                    error_msg = f"❌ Tool '{tool_name}' execution failed:\n"
                    if "error" in result_dict:
                        error_msg += f"Error: {result_dict['error']}\n"
                    if "error_type" in result_dict:
                        error_msg += f"Type: {result_dict['error_type']}\n"
                    if "traceback" in result_dict:
                        error_msg += f"Traceback:\n{result_dict['traceback']}\n"
                    logger.error(f"Tool {tool_name} failed: {result_dict.get('error', 'Unknown error')}")
                    result = error_msg
            except (json.JSONDecodeError, TypeError):
                pass

            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result if isinstance(result, str) else str(result),
            }

        if len(tool_calls) <= 1:
            return [run_one(tc) for tc in tool_calls]

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            return list(executor.map(run_one, tool_calls))

    def _execute_single_tool(self, tool_data: Dict[str, Any]) -> Tuple[str, bool]:
        """
        执行单个已解析的工具调用（{"id", "name", "arguments"}）。
        返回 (结果文本, 是否出错)。
        """
        tool_name = tool_data["name"]
        try:
            result = self.tool_registry.execute_tool(tool_name, tool_data["arguments"])
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"❌ Tool execution exception: {str(e)}", True

        try:
            result_dict = json.loads(result) if isinstance(result, str) else result
            if isinstance(result_dict, dict) and not result_dict.get("success", True):
                error_msg = f"❌ Tool '{tool_name}' execution failed:\n"
                if "error" in result_dict:
                    error_msg += f"Error: {result_dict['error']}\n"
                if "error_type" in result_dict:
                    error_msg += f"Type: {result_dict['error_type']}\n"
                if "traceback" in result_dict:
                    error_msg += f"Traceback:\n{result_dict['traceback']}\n"
                logger.error(f"Tool {tool_name} failed: {result_dict.get('error', 'Unknown error')}")
                return error_msg, True
        except (json.JSONDecodeError, TypeError):
            pass
        return result, False

    # ══════════════════════════════════════════════════════════
    # 消息构建
    # ══════════════════════════════════════════════════════════

    def _build_messages(self, use_tools: bool) -> List[Dict[str, Any]]:
        """system 提示词 + 对话历史。每轮循环都重新构建。"""
        system = SYSTEM_PROMPT_WITH_TOOLS if use_tools else SYSTEM_PROMPT_PLAIN
        return [{"role": "system", "content": system}, *self.conversation_history]

    @staticmethod
    def _assistant_tool_calls_for_history(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """API 规范要求历史消息里 tool_calls 的 arguments 必须是 JSON 字符串。"""
        return [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                    if isinstance(tc["function"]["arguments"], str)
                    else json.dumps(tc["function"]["arguments"]),
                },
            }
            for tc in tool_calls
        ]

    # ══════════════════════════════════════════════════════════
    # 非流式对话（ReAct 循环）
    # ══════════════════════════════════════════════════════════

    def chat(self, message: str, use_tools: bool = True,
             temperature: float = 0.3, max_tokens: int = 2048,
             stream: bool = False):
        """
        发送消息并处理工具调用（ReAct 循环）。

        Returns:
            最终回答字符串；stream=True 时返回 chunk 生成器（见 chat_stream）
        """
        if stream:
            return self.chat_stream(message, use_tools, temperature, max_tokens)

        self.conversation_history.append({"role": "user", "content": message})

        tools = self.tool_registry.get_tool_schemas() if use_tools else None

        iteration = 0
        final_response = ""

        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")

            messages = self._build_messages(use_tools)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if use_tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            assistant_message = response.choices[0].message
            content = assistant_message.content or ""

            # DeepSeek 原生返回结构化 tool_calls，无需解析文本标签
            tool_calls = []
            if use_tools and assistant_message.tool_calls:
                for tc in assistant_message.tool_calls:
                    raw_args = tc.function.arguments
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool arguments for {tc.function.name}: {e}")
                        parsed_args = {}
                    logger.info(f"Model requested tool call: {tc.function.name}({raw_args})")
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": parsed_args,
                        },
                    })

            if tool_calls:
                logger.info(f"Model requested {len(tool_calls)} tool call(s)")

                # assistant 消息（带 tool_calls）写入历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": self._assistant_tool_calls_for_history(tool_calls),
                })

                # 执行工具，结果以 role="tool" 消息回灌
                for result in self._execute_tool_calls(tool_calls):
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"],
                    })

                continue  # 继续 ReAct 循环
            else:
                # 没有工具调用 → 最终回答
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content,
                })
                final_response = content
                break

        if iteration >= MAX_ITERATIONS:
            logger.warning("Maximum iterations reached in ReAct loop")
            final_response = "I've reached the maximum number of reasoning steps. " + final_response

        return final_response

    # ══════════════════════════════════════════════════════════
    # 流式对话（ReAct 循环）
    # ══════════════════════════════════════════════════════════

    def chat_stream(self, message: str, use_tools: bool = True,
                    temperature: float = 0.3, max_tokens: int = 2048):
        """
        流式版本。yield 的 chunk 类型：
        thinking / tool_call / tool_result / tool_error / content / error
        """
        self.conversation_history.append({"role": "user", "content": message})

        tools = self.tool_registry.get_tool_schemas() if use_tools else None

        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"ReAct stream iteration {iteration}")

            messages = self._build_messages(use_tools)

            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if use_tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            collected_content = []
            thinking_buffer = ""
            tool_call_parts: Dict[int, Dict[str, Any]] = {}

            for chunk in stream_response:
                if not (chunk.choices and chunk.choices[0].delta):
                    continue
                delta = chunk.choices[0].delta

                # DeepSeek 推理模型（deepseek-reasoner）的思考内容在独立字段里
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "thinking", "content": reasoning}

                if delta.content:
                    content_chunk = delta.content
                    collected_content.append(content_chunk)

                    # 兼容 <think>...</think> 标签形式的思考内容
                    if '<think>' in content_chunk or thinking_buffer:
                        thinking_buffer += content_chunk
                        if '</think>' in thinking_buffer:
                            thinking_match = re.search(r'<think>(.*?)</think>', thinking_buffer, re.DOTALL)
                            if thinking_match:
                                for char in thinking_match.group(1).strip():
                                    yield {"type": "thinking", "content": char}
                            remaining = re.sub(
                                r'<think>.*?</think>', '', thinking_buffer, flags=re.DOTALL
                            )
                            thinking_buffer = ""
                            if remaining:
                                yield {"type": "content", "content": remaining}
                    else:
                        yield {"type": "content", "content": content_chunk}

                # 流式 tool_calls 是分片到达的：id / name / arguments 可能分散在
                # 多个 chunk 里，按 index 累积（与 OpenAI 行为一致）
                for fragment in getattr(delta, "tool_calls", None) or []:
                    index = getattr(fragment, "index", None)
                    if index is None:
                        index = 0
                    buffered = tool_call_parts.setdefault(index, {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if getattr(fragment, "id", None):
                        buffered["id"] = fragment.id
                    if getattr(fragment, "type", None):
                        buffered["type"] = fragment.type
                    function = getattr(fragment, "function", None)
                    if function:
                        if getattr(function, "name", None):
                            buffered["function"]["name"] += function.name
                        if getattr(function, "arguments", None):
                            buffered["function"]["arguments"] += function.arguments

            complete_response = ''.join(collected_content)

            if tool_call_parts:
                pending_tool_calls = []
                parse_errors = []
                assistant_tool_calls = []

                for index in sorted(tool_call_parts):
                    buffered = tool_call_parts[index]
                    call_id = buffered["id"] or str(uuid.uuid4())[:8]
                    tool_name = buffered["function"]["name"] or "unknown"
                    raw_args = buffered["function"]["arguments"] or "{}"
                    assistant_tool_calls.append({
                        "id": call_id,
                        "type": buffered["type"],
                        "function": {
                            "name": tool_name,
                            "arguments": raw_args,
                        },
                    })

                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        error_msg = f"❌ Tool call parse exception: {str(e)}"
                        logger.error(f"Tool call parse error: {e}")
                        parse_errors.append((tool_name, error_msg))
                        yield {"type": "tool_error", "content": error_msg}
                        continue

                    pending_tool_calls.append({
                        "id": call_id,
                        "name": tool_name,
                        "arguments": parsed_args,
                    })
                    yield {
                        "type": "tool_call",
                        "content": {"name": tool_name, "arguments": parsed_args},
                    }

                # assistant 消息（带 tool_calls）写入历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": complete_response,
                    "tool_calls": assistant_tool_calls,
                })

                for tool_name, error_msg in parse_errors:
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": next(
                            tc["id"] for tc in assistant_tool_calls
                            if tc["function"]["name"] == tool_name
                        ),
                        "content": error_msg,
                    })

                # 并行执行本轮所有有效工具调用
                if not pending_tool_calls:
                    outcomes = []
                elif len(pending_tool_calls) == 1:
                    outcomes = [self._execute_single_tool(pending_tool_calls[0])]
                else:
                    with ThreadPoolExecutor(max_workers=len(pending_tool_calls)) as executor:
                        outcomes = list(executor.map(self._execute_single_tool, pending_tool_calls))

                for tool_data, (result, is_error) in zip(pending_tool_calls, outcomes):
                    if is_error:
                        yield {"type": "tool_error", "content": result}
                    else:
                        yield {"type": "tool_result", "content": result}

                    # 工具结果以 role="tool" 消息回灌历史
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_data["id"],
                        "content": result,
                    })

                continue  # 继续 ReAct 循环，让模型决定下一步
            else:
                # 没有工具调用 → 最终回答
                self.conversation_history.append({
                    "role": "assistant",
                    "content": complete_response,
                })
                break

        if iteration >= MAX_ITERATIONS:
            yield {"type": "error", "content": "Maximum iterations reached in ReAct loop"}

    # ══════════════════════════════════════════════════════════
    # 其他接口（与原书一致）
    # ══════════════════════════════════════════════════════════

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history"""
        return self.conversation_history

    def add_custom_tool(self, name: str, function: callable, description: str, parameters: Dict):
        """Add a custom tool to the registry"""
        self.tool_registry.register_tool(name, function, description, parameters)
        logger.info(f"Added custom tool: {name}")
