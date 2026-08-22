# nca ← deepseek-harness 借鉴实施计划

> 来源研究：2026-04 对 `deepseek-harness`（dsh，DeepSeek 官方 agent harness，TS/Cordis）四路深挖。
> 本文把研究结论展开为可执行的工程计划。每项标注：价值、规模、依赖、验收标准。

## 现状校准（已完成的部分，不重复造轮子）

研究阶段误判了两个"缺口"，实际 nca 已有雏形：

| dsh 概念 | nca 现状 |
|---|---|
| 分级压缩 | `core/src/context_view.rs`（580 行）已有 provider-request 智能压缩：旧工具结果截断、去重、文件提及精简，DryRun/On 双模式 |
| 工具失败循环检测 | `agent.rs` 已有 `MAX_CONSECUTIVE_TOOL_FAILURES = 3` + 诊断详情上报 |

本计划聚焦真实缺口：**调度模型（inbox）、事件事实源、压缩事件化、水车中间件、内核沙箱、会话恢复**。

---

## P1 — Turn/Step 分层 + 单一 Inbox（价值最高，架构级）

### 问题

`run_turn` 是阻塞调用：用户输入、插话（steering）、ask_question 回答都被迫走侧通道
（`question_answer_tx`，见 repl.rs:1614 注释）。turn 进行中无法排队后续消息。
TUI 侧 `TuiCmd::Submit` 在 turn 期间无法送达主循环。

### dsh 的解法

一个 driver + 单一 inbox。turn = 0..N steps；step = 一次模型请求 + 其工具执行。
用户消息在 step 边界被 claim。pre-step 钩子可 reject 或重写。

### 目标设计（Rust 化，不引入运行时容��）

**1. 新增 `TurnDriver`（core，新文件 `agent_driver.rs`）**

```
AgentLoop::run_turn(user_input)          // 保留，向后兼容，内部走 driver
  └─ TurnDriver::drive(inbox)            // 新入口
       inbox: mpsc::Receiver<InboxItem>
       InboxItem { kind: UserPrompt{..} | Steering{..} | QuestionAnswer{..}, .. }

turn loop: while !owed.is_empty()
  step loop:
    claim batch ← inbox (step 边界非阻塞 drain)
    1. provider.chat(messages + claimed)
    2. tool_pipeline 执行（现有）
    3. step_end：若流中有 tool_calls → 还有 owed work → 回到 1
turn_end
```

- inbox 是 `tokio::sync::mpsc::Receiver<InboxItem>`（有界，容量 16）。
- steering 只影响下一次 step 的请求组装（追加为 user 消息，标记 `steering=true` 的事件），
  不打断进行中的流（dsh 同款语义：injected context 等待 claim）。
- `ask_question` 的回答也作为 `InboxItem::QuestionAnswer` 入 inbox，侧通道保留为
  兼容路径，标记 deprecated。

**2. 事件分层（common，向后兼容）**

```
AgentEvent::TurnStarted { turn_id }
AgentEvent::StepStarted { turn_id, step_index }
AgentEvent::StepCompleted { turn_id, step_index, duration_ms, had_tool_calls: bool }
AgentEvent::TurnCompleted { duration_ms }        // 已存在，加 turn_id
```

serde 全部 `#[serde(default)]`，旧事件日志可继续重放。

**3. CLI/TUI 对接**

- REPL：turn 进行中输入 → `InboxItem::UserPrompt`（排队下一 step）或 Steering（默认：
  排队；`/interrupt` 命令走现有 cancel_flag）。
- TUI（Elm）：composer Submit 在 busy 时写 inbox 而非报错；状态栏显示
  "queued: N"。`QuestionSubmit` 路径不变（现有 side channel 保留）。

### 验收标准

1. `cargo test -p nca-core turn_driver`：scripted provider 模拟 3-step turn，
   step 2 进行中注入 steering，step 3 的请求包含 steering 消息。
2. 现有全部 agent.rs 测试不回归（run_turn 兼容路径）。
3. REPL 集成测试：turn 进行中输入排队成功，turn 结束后按序消费。
4. `question_answer_tx` 侧通道仍工作（老路径回归）。

### 规模与顺序

- core：agent_driver.rs 新建 ~400 行 + agent.rs 改造（拆 run_turn_inner 为 step 函数）
- common：event.rs +4 变体
- cli：repl.rs + tui/elm ~150 行
- 依赖：无。**建议第一个做**（P2 的压缩事件化、P4 的 middleware 都挂在 step 边界上）。

---

## P2 — 会话事实源统一（事件日志为唯一真相）

### 问题

`<id>.json`（全量快照）与 `<id>.events.jsonl`（事件流）双轨。快照是 save 时点状态，
事件流是过程。resume 时读快照、忽略事件流；fork/重放只能靠 events.jsonl。两者可能漂移。

### dsh 的解法

纯事件溯源：SessionEvent 带 seq，模型可见历史用 `deriveMessages()` 从日志投影，
"Model-visible means logged" 是运行时断言强制的不变量。Surface 事件（user/assistant/tool_result）
与 log-only 事件（turn/*、chunk 类）类型级区分。

### 目标设计

**Phase A（读路径统一，低风险）— 先做**

- `EventEnvelope` 增加 `surface: bool` 字段（default false）：
  `MessageReceived{role:"user"|"assistant"}`、`ToolCallCompleted`（含 tool_use 配对信息）、
  `QuestionResolved` → surface=true。
- session_store.rs 新增 `replay_surface_events(reader) -> Vec<Message>`：
  顺序扫描 events.jsonl，折叠 surface 事件为消息列表。
- supervisor 的 `resume_session`：优先用 replay 结果重建 `agent.messages`；
  json 快照降级为缓存/加速路径（json mtime > events.jsonl mtime 且校验通过时可用）。

**Phase B（写入原子性）**

- 现在事件经 mpsc fanout → 磁盘写是异步 best-effort（`try_send`）。
  surface 事件改为：`run_turn` 返回前 flush 事件日志（`EventLogWriter.flush()`），
  确保崩溃后模型可见状态与日志一致。
- 新增 `EventLogWriter`（runtime）：内部 buf + 行级写 + fsync 策略（每 turn 结束 fsync，
  而非每事件）。

**Phase C（快照降级为投影缓存，可选/最后）**

- `<id>.json` 只在会话正常关闭时写；resume 走事件重放。json 损坏/缺失不再致命
  （现在 resume 失败会硬报错）。

### 验收标准

1. 单测：构造事件序列（user→assistant→tool_use→tool_result→…）重放，消息列表与
   agent.messages 等价。
2. 集成测试：turn 中途 kill -9 进程，resume 后 agent.messages 与崩溃前 provider 视角一致
   （无孤儿 tool_result，sanitize 逻辑兜底）。
3. 旧日志（无 surface 字段）可重放：按 type 白名单回退推导 surface。

### 规模

- common：event.rs +1 字段；runtime：session_store.rs / session_utils.rs +200 行；
  supervisor.rs resume 路径改造 ~80 行。依赖 P1（surface 事件在 step 边界写入）。
- 依赖：P1 弱依赖（可并行，Phase A 无依赖）。

---

## P3 — 压缩事件化 + 溢出恢复（dsh compaction recovery）

### 问题

nca 压缩是 provider-request 前的一次性 plan（`plan_context_view`），不落事件日志，
压缩后状态不可审计、不可重放。上下文溢出（provider 返回 context overflow 错误）时
没有恢复路径——直接把错误抛给用户。

### dsh 的解法

compaction 是事件流上的三事件 bracket：`compaction/start → summary → compaction/end`，
配 surfaceOp（replace range）原子替换模型���见历史。上下文溢出时在失败 step 与 turn
关闭之间恢复：先尝试工具结果剪枝，不行再摘要，重开新 turn。

### 目标设计

**1. 压缩事件化（context_view.rs → 事件）**

- 新事件：`AgentEvent::ContextCompactionStart { tokens_before, reason }` 与
  `ContextCompactionEnd { tokens_after }`（现有 `ContextCompaction` 保留兼容，
  最终弃用——新代码用 Start/End 对）。
- `plan_context_view` 拆两阶段：`plan()`（纯计算）+ `apply()`（发事件 + 构造请求视图）。
  driver（P1）在 step 边界调用。

**2. 溢出恢复（agent.rs）**

- 识别 provider 错误中的 context overflow 模式（DeepSeek: "context length exceeded"，
  OpenAI: "maximum context length"，需在 ProviderError 上加 `is_context_overflow()`
  分类方法，从各 provider 错误文本归一化）。
- 恢复循环（有界，最多 2 次）：
  ```
  on ProviderError::ContextOverflow:
    1. 加压压缩：retention 降一档（如 On → aggressive）
    1'. 若已最低档 → 失败上报
    2. 重新 plan_context_view → 若 tokens_after 下降 → 重试请求
    3. 若不再下降 → 强制摘要（perform_auto_summarize 兜底，现有 supervisor 逻辑下放）
  ```
- 强制摘要的 KV-cache 考量：dsh 用采样锚定。nca 简化为：摘要后 +`ContextCompactionEnd`
  事件记录 `kv_prefix_broken: true`（信息性，成本面板可见）。

**3. 工具结果预剪枝（补齐 context_view 缺口）**

- 现有 plan 只截断旧工具结果。补一个 `prune_tool_results()`：溢出恢复第 1 步先
  对**最老的 N 个**工具结果做整体删除（不是截断）而非摘要调用（dsh：pruning before
  summary），保 KV 前缀尽量长。注意与 `adjust_cutoff_for_tool_groups` 一样不能切
  工具对——删除整组（assistant+tool_calls + 对应 results）。

### 验收标准

1. 单测：mock provider 第 1 次请求返回 overflow，第 2 次成功 → turn 最终成功，
   `ContextCompactionStart/End` 事件对出现在日志中。
2. 单测：prune 删除的是完整工具组（含 assistant 载体消息），无孤儿产生
   （复用现有 `assert_no_orphaned_tool_results` helper）。
3. DryRun 模式报告 tokens_after 下降路径（现有测试改造）。

### 觔回注意

`context_view.rs` 的去重/截断/文件提及精简是有意设计（研究阶段确认），溢出恢复
**复用**它而不是替换。规模：core ~300 行改动。依赖 P1（step 边界）+ P2（事件对落盘）。

---

## P4 — Waterfall 中间件层（tower 风格，core）

### 问题

压缩、成本、遥测、重试逻辑全部内联在 `run_turn_inner` 的 300 行循环里。
新增横切能力（如 P3 溢出恢复、未来的 retry 插件、脱敏中间件）都要改循环本身。

### dsh 的解法

llm/stream、tools/execute 等是 waterfall 事件，插件包装 next() 实现拦截/重写/短路。

### 目标设计

**1. `core/src/middleware.rs`**

```rust
pub trait AgentMiddleware: Send + Sync {
    fn name(&self) -> &str;
    async fn call(&self, req: StepRequest, next: Next) -> StepOutcome;
}
pub struct StepRequest  { messages: Vec<Message>, tools: Vec<ToolDefinition>, model: String }
pub enum StepOutcome { Proceed, Rewrite(Vec<Message>), ShortCircuit(String /*final_text*/) }
```

- `MiddlewareChain`：`Vec<Arc<dyn AgentMiddleware>>`，按序 wrap，内部 `Next` 仿 tower。
- 中间件顺序（初始）：observability → cost-guard → compaction → retry（P3 的溢出恢复
  作为 retry middleware 的一个 arm）。

**2. 接入点**

- 只 wrap provider.chat 一层（不动 tool_pipeline——它已有清晰的 approval→hooks→execute
  分相，强拆反而破坏现有审批语义）。dsh 同款边界：waterfall 在请求组装侧。
- agent.rs 的 run_turn_inner 改造为：请求组装 → `chain.call(...)` → 流处理不变。

**3. P3 挂载**

溢出恢复作为 `CompactionMiddleware`（处理 ContextOverflow 错误重试）实现，从
run_turn_inner 内联代码移出。

### 验收标准

1. 单测：三个哑中间件按注册序执行，可 Rewrite/ShortCircuit。
2. P3 验收测试不变，但溢出恢复代码位于 middleware.rs 而非 agent.rs。
3. 现有测试不回归（chain 默认为空 = 现状直通）。

### 规模与顺序

- core ~350 行。依赖：P1（StepRequest 定义需要 step 边界存在）；P3 可作为其首个消费者
  一起做或随后。**建议 P1 → P4 → P3**（P3 的恢复循环在 middleware 里更干净）。

---

## P5 — 内核级沙箱（Landlock，Rust 原生）

### 问题

nca 的 bash 执行只有进程组隔离 + SIGKILL，无文件系统约束。模型发起的
`rm -rf /` 或写 ~/.ssh 的命令在 accept-edits 权限模式下可能直接执行。

### dsh 的解法

bwrap → Landlock → Seatbelt → ACL 链，fail-closed（内核不支持则拒绝执行），
Landlock 启动器是 C 写的 self-restrict-then-exec（ABI 探测 + NO_NEW_PRIVS +
restrict + execvp）。dsh 在 TS 里只做 runner 选择与 argv 包装。

### 目标设计

**1. 依赖**：`landlock = "0.4"` crate（Rust 嘴，内核 LSM 嘴）。加入 nca-runtime。

**2. `runtime/src/sandbox.rs`**

```rust
pub enum SandboxBackend {
    Landlock { ruleset: landlock::Ruleset },
    Unconfined,   // 探测失败且用户未要求强制时
}
pub struct SandboxPolicy {
    ro: Vec<PathBuf>,   // 只读根：/usr /bin /lib /nix /etc/alternatives …
    rw: Vec<PathBuf>,   // 读写根：workspace、tmp、cargo home、XDG cache
    net: bool,          // true=允许网络（默认；关网是后续独立项）
    // net=false 时叠加 seccomp-like 限制暂不做——landlock crate 不覆盖 network
}
```

- `exec_confined(cmd, policy) -> Result<PtyOutput>`：fork 后 pre-exec 阩塞
  apply_ruleset（landlock crate 的 `RulesetCreated::restrict_self()`，仅当前线程）
  → execvp。与现有 `PtyManager::exec_streaming` 的 process_group(0) 组合
  （Landlock 约束随 exec 保留给子进程，父进程 unconfined）。
- **fail-closed 语义**：`sandbox = "required"` 时探测失败 → 返回
  `PtyError::SandboxUnavailable`，绝不静默 unconfined 执行；`sandbox = "auto"`
  （默认）→ 探测失败记 warn 日志、降级 unconfined（与 dsh 一致但更保守：dsh 的
  auto 也 fail-closed，nca 先给逃逸阀以免存量用户 broken）。

**3. 配置面（common/config.rs）**

```toml
[permissions.sandbox]
mode = "auto"            # auto | required | off
ro_paths = []            # 追加只读根（默认集 + 用户追加）
rw_paths = []            # 追加读写根（workspace + tmp 默认包含）
net = true
```

**4. 验证**

- sandbox 冒烟命令：`sandbox-run true`（CLI 子命令）做 probe + 执行，输出
  full/partial/unavailable（dsh `--probe` 同款）。
- 现有 pty.rs 测试不动；新增 sandbox 测试需要 rootless 环境支持 Landlock（CI 的
  ubuntu-latest 内核 6.8+ 支持）。

### 验收标准

1. sandbox=required 模式：写 workspace 外路径（如 /etc/foo）的命令失败，
   stderr 带 EACCES；写 workspace 内成功。
2. sandbox=auto 且内核不支持：命令仍执行，`.nca/nca.log` 有 warn 一次。
3. `nca sandbox-run true` 输出探测结果；`--probe` 只探测不执行。
 3'. `cargo test -p nca-runtime sandbox`（标记 ignore 当内核不支持，本地跑）。
4. 文档：AGENTS.md 系统依赖小节 + `docs/tech-stack.md` 同 commit 更新（doc-sync 约定）。

### 规模

- runtime ~400 行 + config ~40 行 + cli ~80 行。依赖：无。可与 P1 并行启动。

---

## P6 — 工具护栏加固（低成本快赢）

### 问题

dsh 有两个便宜但有效的行为 guard，nca 没有：
1. **重复调用检测**：同工具+同参数连续 N 次 → 注入提醒。
2. **协作式工具超时**：`ToolDefinition.timeout_ms` 声明式超时。

### 目标设计

**1. RepeatCallGuard（core，挂 tool_pipeline）**

- Phase 1 权限检查前：对 batch 内每个 call 计算 `hash(tool_name + canonical input)`，
  查 `recent_calls: HashMap<u64, (count, first_seen)>`（容量 32，FIFO 淘汰）。
- 阈值 3/5/8（dsh 同款递进）：第 3 次输出附加提示，第 5 次加强语气，第 8 次硬停
  （返回失败 result + 建议换策略错误信息）。
- 提示注入方式：dsh 用 `additionalContexts`；nca 简化为 append 到 tool result 的
  output 末尾（"[guard] 你已连续第 N 次调用 …"）。会话级 state，随 supervisor 重建。

**2. ToolTimeout（core，tool_pipeline Phase 2）**

- `ToolDefinition` 加 `#[serde(default)] timeout_ms: Option<u64>`。
- Phase 2 并发执行包 `tokio::time::timeout`：超时返回失败 result（错误信息含工具名
  与超时值），不杀外部进程（bash 的进程组 kill 由 pty 超时负责）。
- 各工具标注默认值：web_search/fetch_url 30s、run_validation 300s、其余 None。

**3. dsh 的 "report orthogonal results" 模式**

- `ToolResult` 已有 success/output/error，补 `timed_out: bool`（serde default），
  钉死 "超时" 与 "失败" 不混在一个 bool 里（重放/遥测都受益）。

### 验收标准

1. 单测：同 hash 3 次内无提示、第 3/5/8 次行为分档、不同参数不误伤。
2. 单测：timeout_ms=50 的 stub 工具 50ms 后返回 timed_out=true。
3. 现有 tool_pipeline 测试不回归。

### 规模

- core ~250 行。依赖：无。**可与 P5 并行，建议第一批做**（快赢）。
