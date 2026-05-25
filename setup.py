#!/usr/bin/env python3
"""OpenCanton 一键配置脚本"""

import os
import sys
import subprocess
from pathlib import Path


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(f"⚠️ Python 3.9+ 当前: {v.major}.{v.minor}")
        return False
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    return True


def install_deps():
    print("\n📦 安装依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def create_env():
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env 已存在")
        return

    print("\n⚙️ 创建 .env 配置文件...")
    hf_token = input("HuggingFace Token（可选，回车跳过）: ").strip()

    lines = [
        "# OpenCanton 配置",
        f"CANTON_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5",
        f"CANTON_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct",
        f"CANTON_TEMPERATURE=0.3",
        f"CANTON_RETRIEVAL_K=5",
        f"CANTON_SERVER_PORT=7860",
    ]
    if hf_token:
        lines.append(f"HF_TOKEN={hf_token}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("✅ .env 已创建")


def create_scripts():
    """创建启动脚本。"""
    Path("start.sh").write_text(
        '#!/bin/bash\ncd "$(dirname "$0")"\npython app.py\n',
        encoding="utf-8",
    )
    os.chmod("start.sh", 0o755)
    print("✅ 启动脚本 start.sh 已创建")


def main():
    print("=" * 50)
    print("  🏮 OpenCanton 一键配置")
    print("=" * 50)

    if not check_python():
        sys.exit(1)

    install_deps()
    create_env()
    create_scripts()

    print("\n" + "=" * 50)
    print("  ✅ 配置完成！")
    print("=" * 50)
    print("\n启动方式：")
    print("  Web 界面:  python app.py")
    print("  命令行:    python scripts/canton_agent.py")
    print("  启动脚本:  bash start.sh")
    print(f"\n浏览器打开: http://localhost:7860")


if __name__ == "__main__":
    main()
