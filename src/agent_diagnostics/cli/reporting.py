"""Reporting and evaluation subcommands: report, calibrate, agreement, validate."""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def cmd_report(args):
    """Generate reliability report from annotations.

    Accepts either ``.json`` (annotation document with ``{"annotations":
    [...]}`` or bare list) or ``.jsonl`` (one record per line) so the
    output of ``observatory annotate`` is consumable directly regardless
    of the extension the user chose for that stage.

    Output destination is controlled by ``--output-dir`` (canonical) or
    ``--output`` (deprecated alias, retained for 0.8.x compatibility —
    slated for removal in 1.0).
    """
    from agent_diagnostics.report import generate_report

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        logger.error("annotations file not found: %s", annotations_path)
        sys.exit(1)

    if annotations_path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(annotations_path) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.error(
                        "malformed JSON on line %d of %s: %s",
                        lineno,
                        annotations_path,
                        exc,
                    )
                    sys.exit(1)
        annotations: dict[str, Any] = {"annotations": records}
    else:
        with open(annotations_path) as f:
            annotations = json.load(f)
        # Tolerate a bare-list .json file (older tooling wrote this shape).
        if isinstance(annotations, list):
            annotations = {"annotations": annotations}

    # Resolve output directory from --output-dir (canonical) or --output
    # (deprecated). argparse's mutually_exclusive_group guarantees at most
    # one is set when invoked through main(); direct Namespace callers may
    # supply either attribute.
    output_dir_value = getattr(args, "output_dir", None)
    legacy_output = getattr(args, "output", None)
    if legacy_output is not None and output_dir_value is None:
        _deprecation_msg = (
            "`observatory report --output` is deprecated; use --output-dir "
            "instead. The --output alias will be removed in 1.0."
        )
        warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
        # Python silences DeprecationWarning by default outside __main__, so
        # also log it via the CLI's configured logger — that path is visible
        # regardless of the user's warning filter.
        logger.warning(_deprecation_msg)
        output_dir_value = legacy_output
    if output_dir_value is None:
        logger.error("--output-dir is required")
        sys.exit(1)

    output_dir = Path(output_dir_value)
    md_path, json_path = generate_report(annotations, output_dir)

    logger.info("Report written to %s and %s", md_path, json_path)


def cmd_calibrate(args):
    """Compare two annotation files and emit calibration metrics.

    Reference ("ground truth") comes from ``--reference``; predictor (with
    emitted confidences) comes from ``--predictor``.  If ``--golden-dir`` is
    supplied, the golden corpus is collected into an in-memory annotation
    document and used as the reference.

    Writes ``calibration.md`` and ``calibration.json`` under ``--output-dir``.

    Permission contract: ``--output-dir`` is user-provided and its contents
    are created with the caller's umask — the caller is responsible for
    choosing an appropriately-permissioned directory. Note that
    ``calibration.json`` and ``calibration.md`` may contain corpus-derived
    content (category names, confidence distributions, trial paths), so on
    multi-user shared hosts callers should point ``--output-dir`` at a
    private directory. The internal temp directory used when composing
    ``--golden-dir`` into a reference document is always owner-only
    (``0o700``) and the composed ``reference.json`` is ``0o600``, regardless
    of caller umask.
    """
    import tempfile

    from agent_diagnostics.calibrate import compare_annotations, format_markdown

    predictor_path = Path(args.predictor)
    if not predictor_path.is_file():
        logger.error("predictor annotations not found: %s", predictor_path)
        sys.exit(1)

    # Reference: either a plain annotations file or the golden corpus dir.
    # When the golden corpus is used we materialise the composed document in
    # an internal temp directory that is always owner-only regardless of
    # interpreter. CPython's ``mkdtemp`` already creates the directory with
    # mode 0o700 on POSIX, and the file is written inside it — the explicit
    # chmod calls below are a portability hedge (not a race fix) and pin the
    # file's own mode to 0o600 so that if the file ever escapes the temp dir
    # (backup sweep, future refactor) it is still owner-readable only.  The
    # window between open() and the post-write chmod is not exploitable here
    # because the enclosing directory is 0o700, so no other user can traverse
    # to the file during that window.
    tmp_dir: tempfile.TemporaryDirectory | None = None
    reference_path: Path
    if args.golden_dir:
        golden_dir_path = Path(args.golden_dir)
        if not golden_dir_path.is_dir():
            logger.error("golden corpus directory not found: %s", golden_dir_path)
            sys.exit(1)
        golden_doc = _collect_golden_corpus(golden_dir_path)
        tmp_dir = tempfile.TemporaryDirectory(prefix="observatory-calibrate-")
        os.chmod(tmp_dir.name, 0o700)
        reference_path = Path(tmp_dir.name) / "reference.json"
        with open(reference_path, "w", encoding="utf-8") as f:
            json.dump(golden_doc, f)
        os.chmod(reference_path, 0o600)
    elif args.reference:
        reference_path = Path(args.reference)
        if not reference_path.is_file():
            logger.error("reference annotations not found: %s", reference_path)
            sys.exit(1)
    else:
        logger.error("provide --reference or --golden-dir for the label source")
        sys.exit(1)

    try:
        summary = compare_annotations(predictor_path, reference_path)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "calibration.md"
    md_path.write_text(format_markdown(summary))

    json_path = output_dir / "calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    shared_trials = summary.get("shared_trials", 0)
    logger.info(
        "Calibration report: %s (markdown), %s (json). Shared trials: %d",
        md_path,
        json_path,
        shared_trials,
    )
    if shared_trials == 0:
        # Most common cause: the predictor was annotated against a runs/
        # directory (trial_path = filesystem path) while the golden corpus
        # uses trial_id_short dir names. The join silently produces an
        # empty report — surface the likely cause so the user doesn't have
        # to guess.
        logger.warning(
            "shared_trials=0: predictor and reference have no overlapping "
            "trial_path values. If --golden-dir was used, the predictor's "
            "trial_path fields must match the golden-corpus directory names "
            "(typically trial_id_short hashes), not filesystem paths from "
            "`ingest --runs-dir` / `annotate --signals` against a runs tree."
        )


def _collect_golden_corpus(dir_path: Path) -> dict[str, Any]:
    """Compose per-trial ``expected_annotations.json`` files into one document.

    Each subdirectory of *dir_path* is treated as a trial; its
    ``expected_annotations.json`` contributes one annotation record using the
    directory name as ``trial_path``.
    """
    if not dir_path.is_dir():
        raise FileNotFoundError(f"golden corpus directory not found: {dir_path}")

    annotations: list[dict[str, Any]] = []
    for trial_dir in sorted(dir_path.iterdir()):
        # Reject symlinks defensively: a symlinked trial dir could point
        # outside `dir_path`, letting a malicious --golden-dir read
        # arbitrary `expected_annotations.json` files from the filesystem.
        # Path.is_dir(follow_symlinks=False) is Python 3.12+; use an
        # explicit is_symlink() check for 3.10/3.11 compatibility.
        if trial_dir.is_symlink() or not trial_dir.is_dir():
            continue
        ann_file = trial_dir / "expected_annotations.json"
        if ann_file.is_symlink() or not ann_file.is_file():
            continue
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)
        categories = [
            {
                "name": c.get("name", ""),
                "confidence": c.get("confidence", 1.0),
                "evidence": c.get("evidence", ""),
            }
            for c in data.get("categories", [])
            if c.get("name")
        ]
        annotations.append(
            {
                "trial_path": trial_dir.name,
                "categories": categories,
            }
        )
    return {"annotations": annotations}


def cmd_agreement(args):
    """Report pairwise inter-annotator agreement (Cohen's kappa) per category.

    Reads the narrow-tall annotation store (the ``--annotations-out`` JSONL
    that the annotate / llm-annotate / predict / ensemble commands write)
    and writes ``agreement.md`` + ``agreement.json`` under ``--output-dir``.
    """
    from agent_diagnostics.agreement import compute_agreement, format_markdown
    from agent_diagnostics.annotation_store import AnnotationStore

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        logger.error("annotations file not found: %s", annotations_path)
        sys.exit(1)

    rows = AnnotationStore(annotations_path).read_annotations()
    summary = compute_agreement(rows)

    n_pairs = len(summary["pairs"])
    if n_pairs == 0:
        logger.warning(
            "fewer than two annotator identities in %s — nothing to compare. "
            "Run at least two annotators (e.g. `annotate` and `llm-annotate`) "
            "with --annotations-out pointing at the same store.",
            annotations_path,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "agreement.md"
    md_path.write_text(format_markdown(summary))

    json_path = output_dir / "agreement.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Agreement report: %s (markdown), %s (json). Annotators: %d, pairs: %d",
        md_path,
        json_path,
        len(summary["annotators"]),
        n_pairs,
    )


def cmd_validate(args):
    """Validate annotation files against schema and taxonomy."""
    import jsonschema

    from agent_diagnostics.taxonomy import valid_category_names

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        logger.error("annotations file not found: %s", annotations_path)
        sys.exit(1)

    # Load annotations
    try:
        with open(annotations_path) as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("invalid JSON: %s", e)
        sys.exit(1)

    errors = []

    # Validate against JSON Schema
    schema_path = Path(__file__).parent.parent / "annotation_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=annotations, schema=schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")

    # Validate category names against taxonomy
    valid_names = valid_category_names()
    for i, ann in enumerate(annotations.get("annotations", [])):
        for cat in ann.get("categories", []):
            name = cat.get("name", "")
            if name not in valid_names:
                errors.append(
                    f"Annotation [{i}] ({ann.get('task_id', '?')}): unknown category '{name}'"
                )

    if errors:
        logger.error("Validation FAILED with %d error(s):", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        sys.exit(1)

    n_annotations = len(annotations.get("annotations", []))
    logger.info("Validation passed: %d annotations, all valid.", n_annotations)
    sys.exit(0)
