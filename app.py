#!/usr/bin/env python3
"""
OpenCanton — 广东·大湾区问答智能体
基于 RAG 的广东文化、方言、美食、历史问答系统
Embedding: BAAI/bge-base-zh-v1.5 (本地加载，免费)
LLM: Qwen/Qwen2.5-72B-Instruct (HuggingFace 免费推理 API)
"""

import os
import re
import time
import gradio as gr
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ── 配置 ──
EMBEDDING_MODEL = os.getenv("CANTON_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
LLM_MODEL = os.getenv("CANTON_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = float(os.getenv("CANTON_TEMPERATURE", "0.3"))
RETRIEVAL_K = int(os.getenv("CANTON_RETRIEVAL_K", "5"))
FETCH_K = int(os.getenv("CANTON_FETCH_K", "20"))
PERSIST_DIR = os.getenv("CANTON_PERSIST_DIR", "./canton_knowledge_base")
COLLECTION_NAME = os.getenv("CANTON_COLLECTION", "canton_collection")
SOURCE_DIR = os.getenv("CANTON_SOURCE_DIR", "./canton_source_docs")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SERVER_PORT = int(os.getenv("CANTON_SERVER_PORT", "7860"))


def load_dotenv():
    env_file = Path(".env")
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value

load_dotenv()


def build_knowledge_base():
    """从源文档构建向量库。"""
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("✅ 向量库已存在，跳过构建")
        return

    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"⚠️ 源文档目录不存在: {SOURCE_DIR}")
        return

    print("🔨 正在构建向量库（首次启动，需要几分钟）...")

    loader = DirectoryLoader(
        SOURCE_DIR, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    print(f"  📄 加载 {len(raw_docs)} 个文件")

    # 注入元数据
    for doc in raw_docs:
        source_file = doc.metadata.get("source", "")
        filename = os.path.basename(source_file)
        parent = os.path.basename(os.path.dirname(source_file))
        doc.metadata["category"] = parent
        doc.metadata["filename"] = filename.replace(".txt", "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  📦 生成 {len(chunks)} 个文本块")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=PERSIST_DIR, collection_name=COLLECTION_NAME,
    )
    print("  ✅ 向量库构建完成！")


# ── 启动 ──
print("=" * 55)
print("  🏮 OpenCanton — 广东·大湾区问答智能体")
print("=" * 55)

build_knowledge_base()

print("📦 加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("🗄️ 连接向量数据库...")
vector_db = Chroma(
    persist_directory=PERSIST_DIR, embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vector_db.as_retriever(
    search_type="mmr", search_kwargs={"k": RETRIEVAL_K, "fetch_k": FETCH_K},
)

print(f"🤖 加载 LLM: {LLM_MODEL}...")
llm_kwargs = {"temperature": TEMPERATURE, "max_new_tokens": 1024}
if HF_TOKEN:
    llm_kwargs["huggingfacehub_api_token"] = HF_TOKEN
llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs=llm_kwargs)

# ── Prompt ──
CANTON_PROMPT_TEMPLATE = """你是"阿广"，一个土生土长的广东人，精通粤语方言、广东美食、广东地理、广东历史和粤港澳大湾区政策。

### 你的身份与原则
1. 你以广东本地人的视角回答问题，带有亲切的广东味
2. 涉及粤语时，同时给出粤语写法和普通话解释
3. 涉及美食时，推荐具体的店家和地区
4. 涉及地理时，结合当地特色和实用信息
5. 所有回答基于提供的上下文，超出范围坦诚说"呢个我唔太清楚"
6. 适当穿插粤语词汇，让回答更有广东味

### 回答流程
1. 从上下文中找到最相关的信息
2. 用通俗易懂的语言解释
3. 结合实际经验给出建议
4. 如有不同说法，简要说明

### 对话历史
{history}

### 参考上下文
{context}

### 用户问题
{question}

请以广东人"阿广"的身份回答："""

canton_prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template=CANTON_PROMPT_TEMPLATE,
)

memory = ConversationBufferMemory(
    memory_key="history", input_key="question", return_messages=True,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever,
    chain_type_kwargs={"prompt": canton_prompt, "memory": memory},
    return_source_documents=True,
)

print("✅ OpenCanton 就绪！")
print("=" * 55)


# ── 核心函数 ──
def canton_chat(message, history):
    """广东问答。"""
    start_time = time.time()
    try:
        result = qa_chain({"query": message})
        answer = result["result"]

        for prefix in ["Answer:", "答：", "回答："]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        sources = []
        for doc in result["source_documents"]:
            cat = doc.metadata.get("category", "")
            fn = doc.metadata.get("filename", "")
            if cat:
                sources.append(f"{cat}/{fn}")

        if sources:
            unique = list(dict.fromkeys(sources))[:3]
            answer += "\n\n📚 参考：" + " | ".join(unique)

        return answer
    except Exception as e:
        return f"出错了：{str(e)}\n\n请检查网络连接和模型配置"


def search_topic(topic, category=None):
    """按主题搜索。"""
    filter_dict = {"category": category} if category else None
    docs = vector_db.similarity_search(topic, filter=filter_dict, k=3)
    if docs:
        return "\n\n".join([f"**{d.metadata.get('category', '')}**\n{d.page_content}" for d in docs])
    return f"未找到与「{topic}」相关的内容"


def get_cantonese_translation(text):
    """普通话→粤语翻译提示。"""
    common = {
        "什么": "乜嘢", "怎么": "点解", "哪里": "边度", "没有": "冇",
        "不是": "唔系", "不行": "唔得", "不知道": "唔知", "好看": "靓",
        "好吃": "好食", "谢谢": "唔该/多谢", "对不起": "对唔住",
        "朋友": "老友", "厉害": "犀利", "便宜": "平", "贵": "贵",
        "聊天": "倾偈", "睡觉": "瞓觉", "吃饭": "食饭", "喝酒": "饮酒",
        "上班": "返工", "下班": "收工", "回家": "返屋企",
    }
    result = "## 🔤 常用粤语对照\n\n"
    result += "| 普通话 | 粤语 |\n|--------|------|\n"
    for k, v in common.items():
        result += f"| {k} | {v} |\n"
    return result


# ── Gradio 界面 ──
THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="red", neutral_hue="stone")

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.header {
    text-align: center; padding: 24px;
    background: linear-gradient(15deg, #fff3e0, #ffe0b2);
    border-radius: 12px; margin-bottom: 20px;
}
.header h1 { font-size: 2em; color: #e65100; margin: 0; }
.header p { color: #bf360c; font-size: 1.1em; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="OpenCanton 广东·大湾区智能体") as demo:
    gr.HTML("""
    <div class="header">
        <h1>🏮 OpenCanton</h1>
        <p>广东·大湾区问答智能体 — 食在广州，玩在广东</p>
        <p style="font-size:0.9em; color:#e65100;">
            免费方案 · Embedding: BGE-base-zh · LLM: Qwen2.5-72B · 无需 API Key
        </p>
    </div>
    """)

    with gr.Tabs():
        # Tab 1: 问答
        with gr.Tab("💬 问阿广"):
            gr.ChatInterface(
                fn=canton_chat,
                examples=[
                    "粤语'食脑'是什么意思？",
                    "广州哪里吃早茶最好？",
                    "深圳和广州有什么区别？",
                    "潮汕工夫茶怎么泡？",
                    "粤港澳大湾区有什么政策？",
                    "广东21个地级市各有什么特色？",
                    "客家菜和潮州菜有什么不同？",
                    "港珠澳大桥有多长？",
                ],
                retry_btn="🔄 重试",
                undo_btn="↩️ 撤销",
                clear_btn="🗑️ 清空",
            )

        # Tab 2: 粤语翻译
        with gr.Tab("🔤 粤语速查"):
            gr.Markdown("### 普通话→粤语常用对照")
            trans_output = gr.Markdown(value=get_cantonese_translation(""))
            gr.Button("🔄 刷新").click(fn=lambda: get_cantonese_translation(""), outputs=trans_output)

        # Tab 3: 主题导航
        with gr.Tab("🧭 主题导航"):
            gr.Markdown("按主题浏览广东相关知识")
            with gr.Row():
                topic_input = gr.Textbox(label="搜索主题", placeholder="如：肠粉、丹霞山、南越国...")
                category_select = gr.Dropdown(
                    choices=["全部", "粤语方言", "广东美食", "广东地理", "广东历史", "大湾区政策", "潮汕客家雷州"],
                    value="全部", label="分类筛选",
                )
            topic_btn = gr.Button("🔍 搜索", variant="primary")
            topic_output = gr.Markdown()
            topic_btn.click(
                fn=lambda t, c: search_topic(t, None if c == "全部" else c),
                inputs=[topic_input, category_select], outputs=topic_output,
            )

        # Tab 4: 系统状态
        with gr.Tab("⚙️ 系统状态"):
            gr.Markdown(f"""
### 系统信息

| 项目 | 值 |
|---|---|
| LLM 模型 | `{LLM_MODEL}` |
| Embedding 模型 | `{EMBEDDING_MODEL}` |
| 温度 | {TEMPERATURE} |
| 检索数量 | {RETRIEVAL_K} |
| 知识库目录 | `{PERSIST_DIR}` |
| 端口 | {SERVER_PORT} |
""")

    gr.Markdown("""
    ---
    <center>

    **OpenCanton · 广东·大湾区智能体** | [GitHub](https://github.com/aaroncxxx/OpenCanton)

    知识库：粤语方言 · 广东美食 · 21地级市 · 广东历史 · 大湾区政策 · 潮汕客家雷州

    </center>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT)
