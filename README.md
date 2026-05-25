# OpenCanton — 广东·大湾区问答智能体

基于 RAG（检索增强生成）的广东文化问答系统，涵盖粤语方言、广东美食、广东地理、广东历史、粤港澳大湾区政策等六大领域。

## 功能

- **💬 问阿广** — 以广东人视角回答各类广东问题
- **🔤 粤语速查** — 普通话↔粤语常用对照
- **🧭 主题导航** — 按六大分类浏览知识库
- **📚 引用溯源** — 回答附带参考来源

## 知识库

| 分类 | 内容 |
|------|------|
| 粤语方言 | 日常用语、俚语俗语、语法特点 |
| 广东美食 | 早茶文化、经典名菜、地方特色 |
| 广东地理 | 21个地级市概览 |
| 广东历史 | 南越国、海上丝绸之路、改革开放 |
| 大湾区政策 | 规划纲要、互联互通、产业协同 |
| 潮汕客家雷州 | 三大民系文化 |

## 快速开始

```bash
git clone https://github.com/aaroncxxx/OpenCanton.git
cd OpenCanton
pip install -r requirements.txt
python setup.py          # 一键配置
python app.py            # 启动 Web 界面
```

## 手动安装

```bash
pip install -r requirements.txt

# Web 界面（免费，无需 API Key）
python app.py
# 浏览器打开 http://localhost:7860

# 命令行
python scripts/canton_agent.py
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CANTON_EMBEDDING_MODEL` | 向量模型 | `BAAI/bge-base-zh-v1.5` |
| `CANTON_LLM_MODEL` | 生成模型 | `Qwen/Qwen2.5-72B-Instruct` |
| `CANTON_TEMPERATURE` | 温度 | `0.3` |
| `CANTON_RETRIEVAL_K` | 检索文档数 | `5` |
| `CANTON_SERVER_PORT` | Web 端口 | `7860` |
| `HF_TOKEN` | HuggingFace Token（可选） | 空 |

## 架构

```
OpenCanton/
├── app.py                          # Gradio Web 界面
├── setup.py                        # 一键配置脚本
├── requirements.txt                # 依赖清单
├── canton_agent/                   # 核心模块
│   └── __init__.py
├── scripts/
│   └── canton_agent.py             # CLI 命令行入口
├── canton_source_docs/             # 知识库源文档
│   ├── 粤语方言/
│   ├── 广东美食/
│   ├── 广东地理/
│   ├── 广东历史/
│   ├── 大湾区政策/
│   └── 潮汕客家雷州/
├── canton_knowledge_base/          # 向量库（自动生成）
└── docs/                           # 文档
```

## 免费方案

- **Embedding**: BAAI/bge-base-zh-v1.5（本地加载，免费）
- **LLM**: Qwen/Qwen2.5-72B-Instruct（HuggingFace 免费推理 API）
- 无需 API Key，开箱即用

## License

MIT
