# run_api.ps1 —— 启动 GraphRAG 服务（含正确的环境变量）
#
# 为什么需要这个脚本：
#   这条链路依赖 5 个环境变量，其中 EMBEDDING_MODEL 必须**灌库和查库一致**，
#   漏一个就会退化成乱召回（表现为"系统好像不会答"，极难排查）。
#   手敲容易漏，所以固化下来。
#
# 用法：
#   .\run_api.ps1              # 默认 8000 端口
#   .\run_api.ps1 -Port 8001   # 换端口（避免和已有进程抢）

param(
    [int]$Port = 8000,
    [string]$Model = "BAAI/bge-small-zh-v1.5"
)

$ErrorActionPreference = "Stop"

# HuggingFace 镜像（国内必需，否则模型下载会超时）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 离线模式：模型已在本地缓存（~\.cache\huggingface\hub）时启用。
# 否则 huggingface_hub 每次加载都会联网检查更新，镜像抽风时报
#   [SSL: UNEXPECTED_EOF_WHILE_READING]
# 首次下载新模型时请把这行注释掉。
$env:HF_HUB_OFFLINE = "1"

# 向量模型：必须与灌库时用的一致。
# 换模型 = 换向量空间，旧向量作废，必须重新跑 ingest_simco.py --reset
$env:EMBEDDING_MODEL = $Model

# 打开 LLM 生成。默认 false 时 /ask 只返回检索到的原文，不是答案。
$env:USE_LLM_ANSWER = "true"

# 打开 LLM 兜底路由（关键词判不准时让模型判断走 local 还是 global）
$env:USE_LLM_ROUTER = "true"

# 评测阶段关掉语义缓存：缓存会跨问题污染（实测 Q3 返回了 Q2 的答案）
$env:USE_SEMANTIC_CACHE = "false"

Write-Host "启动 GraphRAG 服务" -ForegroundColor Cyan
Write-Host "  端口      : $Port"
Write-Host "  模型      : $env:EMBEDDING_MODEL"
Write-Host "  生成      : $env:USE_LLM_ANSWER"
Write-Host "  语义缓存  : $env:USE_SEMANTIC_CACHE"
Write-Host ""
Write-Host "另开窗口测试： python tests/ask_simco.py" -ForegroundColor Yellow
Write-Host ""

uvicorn api:app --port $Port
