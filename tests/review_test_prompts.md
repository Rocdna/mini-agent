# 代码审查助手测试提示词集

> 用途:在 `python -m main` 的 `你 >` 提示符下,一条条复制进去,验证 Agent 是否
> 主动调 `search_knowledge` 检索规范,并引用编号(Rx.y)给出审查结论。
> 提示:`/agent` 进入 Agent 模式;`/debug` 打开可看每次发给模型的 messages 快照。

## 建议的单条验证流程
1. 复制下方任一条提示词 → 回车。
2. 观察:Agent 是否调用了 `search_knowledge`,query 是否是规范问题。
3. 看最终回答是否:命中正确规范编号 + 给出严重级别 + 给出可落地的修复建议。

---

## 1. 安全类(第 6 章,最高优先级,最该审)

```
请审查：user_input = request.form['q']; html = "<div>" + user_input + "</div>"; return html
```
期望命中:`R6.1`(XSS:用户内容未转义渲染进 HTML)

```
请审查：cursor.execute("SELECT * FROM orders WHERE id = " + str(order_id))
```
期望命中:`R6.1`(SQL 注入:用户输入字符串拼接进查询)

```
请审查：conn = open('/var/www/' + filename, 'r'): 读用户给定的 filename
```
期望命中:`R6.6`(路径穿越:未校验文件名是否落在允许目录内)

```
请审查：Client(api_key='sk-proj-xxxx明文' 硬编码在代码里)
```
期望命中:`R6.3`(密钥硬编码:应走环境变量/密钥管理)

```
请审查：return pickle.loads(client_data)    # client_data 来自网络请求
```
期望命中:`R6.7`(反序列化:不可信输入泛型反序列化可能 RCE)

## 2. 复杂度/反模式类(第 4、5 章)

```
请审查下面这个函数，它是不是太长、嵌套太深？
def process(self, data):
    if data:
        for x in data:
            if x.get('ok'):
                for y in x['items']:
                    if y > 0:
                        self.stuff.append(y)
```
期望命中:`R4.2`(深层嵌套)/`R4.3`(函数过长)

```
请审查：total = 0; for i in range(n): for j in range(n): total += a[i]*b[j]
```
期望命中:`R8.1`(O(n²) 复杂度)/`R4.8`(可用内置特性简化)

```
请审查：if action == 1: do_a() elif action == 2: do_b() elif action == 3: do_c()
```
期望命中:`R4.5`(魔法数字:1/2/3 应命名为常量)

## 3. 可读性/错误处理类(第 3、7 章)

```
请审查 def f(a, b):
    r = tmp
    pass
```
期望命中:`R3.1`(命名不自解释)

```
请审查：
try:
    do_something()
except:
    pass
```
期望命中:`R7.1`/`R7.2`(裸 except 吞一切 + 静默吞异常)

```
请审查 f = open('log.txt','w'); f.write(msg)  # 没 close
```
期望命中:`R7.6`(资源未用 with 上下文管理器清理)

## 4. 性能/并发类(第 8、10 章)

```
请审查：for user in users: info = db.query(user.id); print(info)
```
期望命中:`R8.2`(循环内查询 = N+1 反模式)

```
请审查：resp = http.get(url)
```
期望命中:`R10.3`(外部调用缺 timeout,可能挂死)

---

## 进阶验证(不喂整段代码,只测检索命中)

快速确认某类问题该归哪条规范,省 token:
```
这类问题该归哪条规范？SQL 拼接进查询
```
期望:agent 应给出 `R6.1`,而不该去 web_search 或编造编号。

如果能直接命中预期编号,说明 RAG 的规范检索质量 OK;若偏号/检索乱,就去检查
`rag/data_review/review_standards.md` 的分块和 query 嵌入。