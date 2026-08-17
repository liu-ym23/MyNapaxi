# candle-qwen

`candle-qwen` 是基于 Candle 的 Qwen2 CPU 量化推理库，使用 GGUF
Q4_0 模型。它只提供 Rust API，不包含命令行程序。

当前提供的主要功能：

- 从 Hugging Face 下载模型并复用本地缓存
- 从本地 GGUF 和 `tokenizer.json` 离线加载
- 单次文本生成
- temperature、top-k、top-p、随机种子和重复惩罚配置

## 编译

在本仓库根目录执行：

```bash
cargo build --release -p candle-qwen
```

构建产物为：

```text
target/release/libcandle_qwen.rlib
```

## 添加依赖

在同一 workspace 中，可以直接使用 workspace 成员。

从其他 Rust 项目引用时，在该项目的 `Cargo.toml` 中添加：

```toml
[dependencies]
anyhow = "1"
candle-qwen = { path = "../candle/candle-qwen" }
```

Cargo 包名是 `candle-qwen`，Rust 代码中的 crate 名是 `candle_qwen`。

## 自动下载并生成

```rust
use candle_qwen::{GenerationConfig, ModelSize, Qwen};

fn main() -> anyhow::Result<()> {
    // 首次运行时下载文件，后续运行复用这个目录中的缓存。
    let mut qwen =
        Qwen::from_hugging_face(ModelSize::B0_5, "../candle/cache/huggingface")?;

    let config = GenerationConfig {
        temperature: 0.2,
        max_new_tokens: 128,
        ..Default::default()
    };

    let answer = qwen.generate("请简单介绍 Rust", &config)?;
    println!("{answer}");
    Ok(())
}
```

缓存路径按调用程序的当前工作目录解析。为了避免在不同目录产生多份模型缓存，
实际应用中建议传入绝对路径或由应用统一管理的缓存目录。

## 离线加载本地文件

```rust
use candle_qwen::{GenerationConfig, Qwen};

fn main() -> anyhow::Result<()> {
    let mut qwen = Qwen::from_files(
        "/path/to/qwen2-0_5b-instruct-q4_0.gguf",
        "/path/to/tokenizer.json",
    )?;

    let answer = qwen.generate("你好", &GenerationConfig::default())?;
    println!("{answer}");
    Ok(())
}
```

GGUF 和 tokenizer 必须属于匹配的 Qwen2 Instruct 模型版本。

## 公共接口

### `Qwen::from_hugging_face`

```rust
pub fn from_hugging_face(
    size: ModelSize,
    cache_dir: impl AsRef<Path>,
) -> anyhow::Result<Qwen>
```

下载或复用缓存中的 GGUF 模型和 tokenizer，然后在 CPU 上加载模型。

- `size`：选择模型大小。
- `cache_dir`：Hugging Face 缓存根目录；库会在其中创建 `hub/`。

该接口需要网络访问，除非所需文件已完整缓存。

### `Qwen::from_files`

```rust
pub fn from_files(
    model_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
) -> anyhow::Result<Qwen>
```

从本地文件加载模型，不执行下载。

- `model_path`：Qwen2 Q4_0 GGUF 文件路径。
- `tokenizer_path`：匹配模型的 `tokenizer.json` 路径。

### `Qwen::generate`

```rust
pub fn generate(
    &mut self,
    prompt: &str,
    config: &GenerationConfig,
) -> anyhow::Result<String>
```

根据输入生成一次回答并返回 `String`。

- `prompt`：用户输入。库会自动套用 Qwen2 Instruct 对话模板。
- `config`：本次生成参数。

每次调用都会清空 KV cache，因此多次调用彼此独立，不会自动保留对话历史。
同一个 `Qwen` 实例可以重复使用，模型不需要反复加载。

## 模型选项

`ModelSize` 提供以下选项：

- `ModelSize::B0_5`：Qwen2 0.5B Instruct Q4_0，默认选项。
- `ModelSize::B1_5`：Qwen2 1.5B Instruct Q4_0。
- `ModelSize::B7`：Qwen2 7B Instruct Q4_0。
- `ModelSize::B72`：Qwen2 72B Instruct Q4_0。

更大的模型需要更多内存，CPU 推理速度也会明显降低。移动端建议优先使用
0.5B，确认内存和速度满足要求后再尝试更大的模型。

## 生成参数

`GenerationConfig::default()` 的默认配置为：

```rust
GenerationConfig {
    temperature: 0.8,
    top_p: None,
    top_k: None,
    max_new_tokens: 256,
    seed: 42,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
}
```

各参数含义：

- `temperature: f64`
  - 控制随机性。值越高，输出越随机。
  - 小于等于 `0.0` 时使用 ArgMax 贪心生成，结果更稳定。
- `top_p: Option<f64>`
  - nucleus sampling 阈值，通常设置为 `0.0` 到 `1.0`。
  - `None` 表示不启用 top-p。
- `top_k: Option<usize>`
  - 每一步只在概率最高的 k 个 token 中采样。
  - `None` 表示不启用 top-k。
- `max_new_tokens: usize`
  - 最多生成多少个新 token。
  - 设为 `0` 时直接返回空字符串。
- `seed: u64`
  - 随机采样种子。相同模型、输入和参数通常会产生相同结果。
- `repeat_penalty: f32`
  - 重复惩罚系数。
  - `1.0` 表示关闭重复惩罚，大于 `1.0` 会降低重复内容出现的概率。
- `repeat_last_n: usize`
  - 对最近多少个已生成 token 应用重复惩罚。

`top_k` 和 `top_p` 可以单独使用，也可以同时使用。`temperature <= 0.0`
时会直接使用贪心生成，top-k 和 top-p 不生效。

推荐的稳定生成配置：

```rust
let config = GenerationConfig {
    temperature: 0.0,
    max_new_tokens: 128,
    ..Default::default()
};
```

推荐的常规采样配置：

```rust
let config = GenerationConfig {
    temperature: 0.7,
    top_p: Some(0.9),
    top_k: Some(40),
    max_new_tokens: 256,
    ..Default::default()
};
```

## 错误处理

所有加载和生成接口都返回 `anyhow::Result`。调用方可以使用 `?` 传播错误，
也可以自行记录并转换错误：

```rust
match qwen.generate("你好", &config) {
    Ok(answer) => println!("{answer}"),
    Err(error) => eprintln!("生成失败: {error:#}"),
}
```

常见错误包括模型或 tokenizer 路径错误、缓存不完整、下载失败、GGUF
格式不兼容以及内存不足。

## 当前限制

- 仅支持 CPU 推理。
- 仅支持 Qwen2 Instruct GGUF Q4_0 模型。
- `generate` 返回完整结果，暂未提供公共流式回调接口。
- 不自动管理多轮对话历史。
- 当前 crate 类型是 `rlib`，不能直接从 Kotlin/Java 调用。集成 Android
  还需要增加 JNI 接口并构建 `cdylib`。
