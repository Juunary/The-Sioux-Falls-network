"""
backend/llm — LLM Teacher / Planner modules (Stage 1+)

obs_to_text, text_to_action, cache, policy_context, config can be imported
without any LLM SDK installed.

teacher.py requires at least one provider SDK for actual API calls:
  - Anthropic (Claude):  pip install anthropic
  - Gemini (Google):     pip install google-generativeai
  - All at once:         pip install -r backend/requirements-llm.txt

Supported providers: "anthropic", "gemini"
"""
