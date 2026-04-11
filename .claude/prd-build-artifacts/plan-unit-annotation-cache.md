# Plan: unit-annotation-cache

## Step 1: Create annotation_cache.py

- `cache_key(prompt_text, model_id) -> str`: sha256 hex of `prompt_text + model_id`
- `get_cached(cache_dir, key) -> list[dict] | None`: read `{cache_dir}/{key}.json`, return parsed list or None
- `put_cached(cache_dir, key, categories)`: write JSON to `{cache_dir}/{key}.json`, create dirs

## Step 2: Modify llm_annotator.py — integrate cache

- Import annotation_cache functions
- In `annotate_trial_claude_code()`: after building prompt, compute cache key with model_alias, check cache, return if hit, else call LLM and cache result
- In `annotate_trial_api()`: same pattern with model_id

## Step 3: Modify llm_annotator.py — structured output via tool-use (API backend)

- Define annotate tool with `_ANNOTATION_SCHEMA` category_assignment as input_schema
- In `annotate_trial_api()`: pass tools + tool_choice to messages.create
- Parse tool_use content block instead of raw text
- Remove markdown fence stripping code

## Step 4: Write tests

- `test_annotation_cache.py`: test_cache_key_determinism, test_second_run_zero_calls, test_cache_directory_created
- `test_backend_parity.py`: same prompt through FakeLLMBackend both code paths
- `test_backend_model_parity.py`: verify \_API_MODEL_MAP and \_MODEL_ALIASES have matching keys
