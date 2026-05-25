#!/usr/bin/env python3
"""
OpenCanton CLI — 命令行版广东问答
用法: python scripts/canton_agent.py [--brief] [--md] [--topic 主题]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


# 配置
EMBEDDING_MODEL = os.getenv("CANTON_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
LLM_MODEL = os.getenv("CANTON_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = float(os.getenv("CANTON_TEMPERATURE", "0.3"))
RETRIEVAL_K = int(os.getenv("CANTON_RETRIEVAL_K", "5"))
FETCH_K = int(os.getenv("CANTON_FETCH_K", "20"))
PERSIST_DIR = os.getenv("CANTON_PERSIST_DIR", "./canton_knowledge_base")
COLLECTION_NAME = os.getenv("CANTON_COLLECTION", "canton_collection")
SOURCE_DIR = os.getenv("CANTON_SOURCE_DIR", "./canton_source_docs")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def build_knowledge_base():
    """从源文档构建向量库。"""
    from pathlib import Path as P
    source_path = P(SOURCE_DIR)
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

    for doc in raw_docs:
        source_file = doc.metadata.get("source", "")
        filename = os.path.basename(source_file)
        parent = os.path.basename(os.path.dirname(source_file))

        content = doc.page_content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1].strip()
                body = parts[2].strip()
                for line in fm_text.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        doc.metadata[k.strip()] = v.strip()
                doc.page_content = body

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


def load_dotenv():
    env_file = Path(__file__).resolve().parent.parent / ".env"
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


def init_system():
    """初始化系统（含知识库构建）。"""
    load_dotenv()

    # 确保知识库存在
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print("🔨 首次运行，构建知识库...")
        build_knowledge_base()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"k": RETRIEVAL_K, "fetch_k": FETCH_K},
    )

    llm_kwargs = {"temperature": TEMPERATURE, "max_new_tokens": 1024}
    if HF_TOKEN:
        llm_kwargs["huggingfacehub_api_token"] = HF_TOKEN
    llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs=llm_kwargs)

    prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template="""你是"阿广"，一个土生土长的广东人，精通粤语方言、广东美食、广东地理、广东历史和粤港澳大湾区政策。

你的原则：
1. 以广东本地人的视角回答，带有亲切的广东味
2. 涉及粤语时，同时给出粤语写法和普通话解释
3. 所有回答基于提供的上下文，超出范围坦诚说"呢个我唔太清楚"
4. 适当穿插粤语词汇

对话历史：{history}
参考上下文：{context}
用户问题：{question}

请以广东人"阿广"的身份回答：""",
    )

    memory = ConversationBufferMemory(
        memory_key="history", input_key="question", return_messages=True,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
        return_source_documents=True,
    )
    return chain, vector_db


def main():
    parser = argparse.ArgumentParser(description="OpenCanton CLI — 广东·大湾区问答")
    parser.add_argument("--brief", action="store_true", help="简洁模式")
    parser.add_argument("--md", action="store_true", help="Markdown 输出")
    parser.add_argument("--topic", type=str, help="直接查询主题")
    args = parser.parse_args()

    print("🏮 OpenCanton CLI 启动中...")
    chain, vector_db = init_system()
    print("✅ 就绪！输入问题开始对话，输入 q 退出\n")

    if args.topic:
        result = chain({"query": args.topic})
        print(result["result"])
        return

    while True:
        try:
            question = input("🙋 你: ").strip()
            if not question or question.lower() in ("q", "quit", "exit"):
                print("👋 拜拜！得闲饮茶！")
                break

            result = chain({"query": question})
            answer = result["result"]

            if args.brief:
                answer = answer[:200] + "..." if len(answer) > 200 else answer

            print(f"\n🏮 阿广: {answer}\n")

            sources = []
            for doc in result["source_documents"]:
                cat = doc.metadata.get("category", "")
                if cat:
                    sources.append(cat)
            if sources:
                unique = list(dict.fromkeys(sources))[:3]
                print(f"📚 参考: {' | '.join(unique)}\n")

        except KeyboardInterrupt:
            print("\n👋 拜拜！得闲饮茶！")
            break


if __name__ == "__main__":
    main()
