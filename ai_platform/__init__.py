# ai/ — Invest.ai AI Platform
#
# Modules:
#   llm_router.py      — unified multi-provider LLM client (OpenAI / Anthropic / NVIDIA)
#   signal_narrator.py — convert ML predictions to investor-readable prose
#   llm_judge.py       — LLM-as-judge evaluation hook for the supervised pipeline
#   nvidia_llm.py      — NVIDIA NIM company-overview helper (legacy, now wraps llm_router)
