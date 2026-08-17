# Candle Qwen2 本地推理库

基于 [Candle](https://github.com/huggingface/candle) 精简的 Qwen2 CPU 量化推理库。

## 环境

需要 Rust（`rustc` / `cargo`），以及：

```bash
sudo apt install -y build-essential pkg-config libssl-dev
```

## 编译

```bash
cargo build --release -p candle-qwen
```

构建产物是 Rust 库，不再生成 `candle-qwen` 命令行程序。

## 使用

```rust
use candle_qwen::{GenerationConfig, ModelSize, Qwen};

fn main() -> anyhow::Result<()> {
    // 自动下载或复用 Hugging Face 缓存。
    let mut qwen =
        Qwen::from_hugging_face(ModelSize::B0_5, "cache/huggingface")?;

    let config = GenerationConfig {
        temperature: 0.2,
        max_new_tokens: 128,
        ..Default::default()
    };
    let response = qwen.generate("你好", &config)?;
    println!("{response}");
    Ok(())
}
```

也可以完全离线加载本地文件：

```rust
let mut qwen = Qwen::from_files("model.gguf", "tokenizer.json")?;
```

公共 API：

- `Qwen::from_files`：加载本地 GGUF 和 tokenizer。
- `Qwen::from_hugging_face`：下载或复用 Hugging Face 缓存。
- `Qwen::generate`：执行一次生成并返回 `String`。
- `GenerationConfig`：温度、top-p、top-k、最大 token、随机种子和重复惩罚。
- `ModelSize`：`B0_5`、`B1_5`、`B7`、`B72`。

## 项目结构

```
candle-core/           张量运算、GGUF 读取
candle-nn/             神经网络层
candle-transformers/   Qwen2 量化模型
candle-qwen/           candle-qwen 库 crate
```
