2-6 手动代码 实验 复现 []
2-2 跑               []
2-4 跑               []

Agent 状态栏
Agent skill
上下文压缩
    第一，解决长度约束和成本约束
    第二，提升思考质量——总结后的知识比原始形式更利于模型使用
    第三，缓解模型的上下文焦虑（Context Anxiety） 提前进行上下文压缩

deepseek 输出格式
```
{
  "id": "a01a0b9d-0de6-4028-8306-c47940ffab89",
  "object": "chat.completion",
  "created": 1787207797,
  "model": "deepseek-v4-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是代码审查助手，可以帮你：\n\n- 🔍 **审查代码**：分析逻辑、发现 bug、提出优化建议\n- 🌐 **联网查证**：查询版本号、CVE 漏洞、API 变更、最新技术资讯等\n- 📚 **解释概念**：解答编程相关问题\n\n有什么需要帮忙的吗？比如发一段代码让我看看，或者问某个技术问题？",
        "reasoning_content": "The user just said \"你好\" (Hello). This is a simple greeting, no need to search or use tools. I should just respond in Chinese with a friendly greeting and ask what they need help with."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1086,
    "completion_tokens": 128,
    "total_tokens": 1214,
    "prompt_tokens_details": {
      "cached_tokens": 1024
    },
    "completion_tokens_details": {
      "reasoning_tokens": 42
    },
    "prompt_cache_hit_tokens": 1024,
    "prompt_cache_miss_tokens": 62
  },
  "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e"
}
```