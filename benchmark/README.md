# Napaxi On-Device Benchmark

端到端 benchmark：每个 case 独立地卸载重装 App（彻底清除工作区/记忆/配置等持久化状态）、
无头运行、采集 6 项指标、按 suite 汇总成报告。

## Suite 结构

case 按 suite 组织，存放在 `benchmark/suites/<suite>.jsonl`（每行一个 JSON case）。
新增 suite 只需添加新的 `.jsonl` 文件，harness 与报告自动识别。

### basic_tool_call（当前唯一 suite）

考察「智能体能否按用户指令正确发起工具调用」。case 的 prompt 明确指定要调用的工具和
关键参数，评分三档：

| 分数 | 条件 |
|---|---|
| **1.0** | 本轮**第一次**工具调用就是指定的工具且参数匹配 |
| **0.5** | 第一次不是，但整轮中**某次**调用以指定参数调用了该工具 |
| **0.0** | 全轮从未以指定参数调用该工具 |

参数匹配规则：`required_arguments` 中的每个键值对都须出现在实际调用的参数 JSON 中
（字符串比较忽略首尾空白）。当前 18 个 case 覆盖 16 种工具：shell、read_file、
apply_patch、memory_write/read/search/tree、session_recall、skill_list/search/info、
mcp_server_list、mcp_tool_list、http、web_fetch、web_search。

case 可选 `setup_prompt` 字段：正式评测前在同一会话先跑一轮布置消息（如创建待读取的
文件），setup 轮的工具调用与耗时**不计入**指标。

### 复杂任务 suite（预留）

后续设计的多步任务 suite 直接新增 `suites/<name>.jsonl` 即可；非工具调用型 case 使用
`expect` 规则评分（`must_contain` / `must_not_contain` / `min_length` / `min_tool_calls`），
得分 = 命中规则数 / 总规则数。

## 六项指标

| 指标 | 采集方式 |
|---|---|
| 任务完成度 (0~1) | 按 suite 的评分规则（见上） |
| 消耗总时间 | 发送 → `RunCompleted` 事件到达的毫秒差 |
| 首 token 延迟 (TTFT) | 发送 → 第一个 `ResponseDelta`（或 `Response`/首个推理增量）到达 |
| 消耗 token 总数 | Rust context engine 记录的 `LlmUsage`（prompt + output，含 cache 细分） |
| 工具调用次数 | 事件流 `ToolCall` 计数，与 `RunCompleted.tool_call_count` 交叉验证 |
| 工具调用成功率 | `ToolResult.is_error == false` 的比例 |

## 使用

```sh
# 1. 配置模型（第一次）
cp benchmark/config.example.json benchmark/config.local.json
#    编辑 base_url / api_key / model（config.local.json 已加入 .gitignore）

# 2. 一键运行（自动编译Rust+APK → 逐case卸载/安装/运行/拉取 → 按suite汇总）
./benchmark/run_benchmark.sh

# 常用变体
./benchmark/run_benchmark.sh --skip-build                        # 复用现有APK
./benchmark/run_benchmark.sh --suites basic_tool_call            # 只跑指定suite
./benchmark/run_benchmark.sh --cases btc-shell-uname,btc-http    # 只跑指定case
./benchmark/run_benchmark.sh --reset-mode clear                  # 用 pm clear 代替卸载重装（快，但隔离弱）
./benchmark/run_benchmark.sh --out /tmp/my-results               # 指定输出目录
./benchmark/run_benchmark.sh --record                            # 每个case录屏（mp4与结果JSON同目录）
```

结果在 `benchmark/results/<run_id>/`：每 case 一个 `result-<suite>-<id>.json`
（含完整对话 trace、工具调用明细、原始 context status）+ `report.md` + `report.json`
（按 suite 分组，多 suite 时附总体汇总）。`--record` 时另有每 case 一个
`<suite>-<id>.mp4` 录屏（480×1008 H264，由伴生的 bench-recorder.apk 捕获；
系统投影授权对话框由 harness 自动点击；recorder 是独立包名，不受每 case
卸载重装影响）。

## basic_tool_call case 格式

```json
{"id": "btc-shell-uname",
 "prompt": "请用 shell 工具执行命令 uname -r，并把输出原样告诉我。",
 "timeout_seconds": 300,
 "required_tool": "shell",
 "required_arguments": {"command": "uname -r"},
 "setup_prompt": null}
```

## 工作原理

```
run_benchmark.sh (host)
 ├─ ./tools/scripts/build.sh fast android + flutter build apk
 └─ for suite in suites/*.jsonl: for case in suite:
     ├─ adb uninstall + adb install          ← 每case全新环境（无记忆残留）
     ├─ am start --es benchmark_b64 <base64> ← 配置+case经intent传入，key不落盘
     ├─ App: main()检测payload → 无头模式
     │    └─ (可选setup轮) → NapaxiEngine.create → sendToSession
     │       → 监听事件流计时/计数 → contextStatus()读token → 三档评分 → JSON落盘
     ├─ 轮询 *.done → adb pull result.json   ← 卸载前必须拉走
     └─ aggregate.py → 按suite分组 report.md/json
```

App 端改动见 `examples/flutter/lib/benchmark/benchmark_runner.dart` 与
`MainActivity.kt`（intent extra 经 MethodChannel 传入 Dart）。
