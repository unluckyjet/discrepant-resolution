"""Short display names for HELM model ids."""
SHORT = {
    "01-ai_yi-34b": "Yi 34B", "01-ai_yi-6b": "Yi 6B", "allenai_olmo-7b": "OLMo 7B",
    "anthropic_claude-2.1": "Claude 2.1", "anthropic_claude-3-haiku-20240307": "Claude 3 Haiku",
    "anthropic_claude-3-opus-20240229": "Claude 3 Opus", "anthropic_claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "anthropic_claude-instant-1.2": "Claude Instant 1.2", "databricks_dbrx-instruct": "DBRX Instruct",
    "deepseek-ai_deepseek-llm-67b-chat": "DeepSeek 67B", "google_gemini-1.0-pro-001": "Gemini 1.0 Pro",
    "google_gemini-1.5-flash-preview-0514": "Gemini 1.5 Flash", "google_gemini-1.5-pro-preview-0409": "Gemini 1.5 Pro",
    "google_gemma-7b": "Gemma 7B", "google_text-bison@001": "PaLM 2 Bison", "google_text-unicorn@001": "PaLM 2 Unicorn",
    "meta_llama-2-13b": "Llama 2 13B", "meta_llama-2-70b": "Llama 2 70B", "meta_llama-2-7b": "Llama 2 7B",
    "meta_llama-3-70b": "Llama 3 70B", "meta_llama-3-8b": "Llama 3 8B", "microsoft_phi-2": "Phi-2",
    "mistralai_mistral-7b-v0.1": "Mistral 7B", "mistralai_mistral-large-2402": "Mistral Large",
    "mistralai_mistral-small-2402": "Mistral Small", "mistralai_mixtral-8x22b": "Mixtral 8x22B",
    "mistralai_mixtral-8x7b-32kseqlen": "Mixtral 8x7B", "openai_gpt-3.5-turbo-0613": "GPT-3.5 Turbo",
    "openai_gpt-4-0613": "GPT-4 (0613)", "openai_gpt-4-1106-preview": "GPT-4 Turbo",
    "openai_gpt-4o-2024-05-13": "GPT-4o", "qwen_qwen1.5-14b": "Qwen1.5 14B", "qwen_qwen1.5-32b": "Qwen1.5 32B",
    "qwen_qwen1.5-72b": "Qwen1.5 72B", "qwen_qwen1.5-7b": "Qwen1.5 7B",
    "snowflake_snowflake-arctic-instruct": "Arctic Instruct", "writer_palmyra-x-v3": "Palmyra X v3",
}


def short(m):
    return SHORT.get(m, m)
