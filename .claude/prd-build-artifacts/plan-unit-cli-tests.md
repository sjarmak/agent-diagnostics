# Plan: unit-cli-tests

1. Test each cmd\_\* by creating argparse.Namespace with required attrs, calling directly
2. Use tmp_path for all file I/O
3. Mock heavy dependencies (extract_all, annotate_all, generate_report, annotate_trial_llm, load_taxonomy, train, save_model, load_model, predict_all, ensemble_all, evaluate, format_eval_markdown, jsonschema.validate, valid_category_names)
4. Error path tests use pytest.raises(SystemExit)
5. For cmd_llm_annotate: create real trajectory files on disk, mock annotate_trial_llm and load_taxonomy
6. For cmd_validate: test missing file, invalid JSON, schema error, and valid file paths
