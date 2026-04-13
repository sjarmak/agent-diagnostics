# Test Results: unit-jsonl-format

## Run

```
python3 -m pytest tests/test_cli.py tests/test_signals.py -x -q
129 passed in 0.29s
```

## New Tests Added

### test_signals.py

- TestWriteJsonl::test_writes_one_object_per_line
- TestWriteJsonl::test_creates_meta_sidecar
- TestWriteJsonl::test_roundtrip
- TestLoadSignals::test_load_json
- TestLoadSignals::test_load_jsonl
- TestLoadSignals::test_json_non_array_raises
- TestLoadAnnotations::test_load_json_envelope
- TestLoadAnnotations::test_load_jsonl_reconstructs_envelope
- TestWriteOutput::test_json_extension_writes_json
- TestWriteOutput::test_jsonl_extension_writes_jsonl
- TestWriteOutput::test_jsonl_with_annotation_envelope
- TestIsJsonlPath::test_jsonl
- TestIsJsonlPath::test_json

### test_cli.py

- TestCmdExtractJsonl::test_jsonl_output
- TestCmdExtractJsonl::test_json_output_still_works
- TestCmdAnnotateJsonl::test_jsonl_input_and_output
- TestCmdAnnotateJsonl::test_json_input_jsonl_output

## Acceptance Criteria Coverage

- extract CLI writes .jsonl when output path ends in .jsonl: PASS
- Each line of .jsonl is a valid JSON object: PASS
- .meta.json sidecar written with schema_version, taxonomy_version, generated_at: PASS
- CLI extract works with both .json and .jsonl: PASS
- CLI annotate works with both .json and .jsonl input/output: PASS
- load_signals() transparently reads both formats: PASS
- Tests cover roundtrip, meta sidecar, format detection: PASS
