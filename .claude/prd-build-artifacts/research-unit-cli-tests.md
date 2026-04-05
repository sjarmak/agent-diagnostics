# Research: unit-cli-tests

## CLI Functions (cli.py)

10 cmd\_\* functions total (task says 8, but there are actually 10):

1. `cmd_extract(args)` - needs args.runs_dir, args.output; calls extract_all
2. `cmd_annotate(args)` - needs args.signals, args.output; calls annotate_all
3. `cmd_report(args)` - needs args.annotations, args.output; calls generate_report
4. `cmd_llm_annotate(args)` - needs args.signals, args.output, args.sample_size, args.model, args.backend; calls annotate_trial_llm, load_taxonomy
5. `cmd_train(args)` - needs args.labels, args.signals, args.output, args.min_positive, args.lr, args.epochs, args.eval; calls train, save_model, evaluate, format_eval_markdown
6. `cmd_predict(args)` - needs args.model, args.signals, args.output, args.threshold; calls load_model, predict_all
7. `cmd_ensemble(args)` - needs args.signals, args.model, args.output, args.threshold, args.min_f1; calls load_model, ensemble_all
8. `cmd_validate(args)` - needs args.annotations; calls jsonschema.validate, valid_category_names

## Error Paths

- cmd_extract: runs_dir not a directory -> SystemExit(1)
- cmd_annotate: signals file not found -> SystemExit(1)
- cmd_report: annotations file not found -> SystemExit(1)
- cmd_llm_annotate: signals file not found -> SystemExit(1); no trials with trajectories -> SystemExit(1)
- cmd_validate: annotations file not found -> SystemExit(1); invalid JSON -> SystemExit(1); schema validation error -> SystemExit(1)

## Existing Tests

- test_main_importable, test_help_output, test_no_judge_flag, test_cli_imports, test_dunder_main, test_pyproject_scripts
- No direct cmd\_\* function tests exist yet
