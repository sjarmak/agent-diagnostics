"""CLI entrypoint for the Agent Reliability Observatory.

Logging discipline
------------------
The root logger is configured only in :func:`main` (the process entry point);
library modules obtain loggers via ``logging.getLogger(__name__)`` and never
install handlers of their own.  Log records are emitted on stderr with the
format ``"%(levelname)s %(name)s: %(message)s"`` so structured pipelines can
parse them without interleaving user-facing stdout.

Genuine user-facing stdout output (tables, schema reports, pipeline summaries)
is still emitted with :func:`print`; each such call site carries an inline
``# STDOUT: <why>`` comment.  Every other caller in this module uses the module
logger below.
"""

from agent_diagnostics.cli._helpers import (
    _MODEL_ALIAS_TO_IDENTITY,
    _annotations_to_narrow_rows,
    _resolve_llm_annotator_identity,
    _write_to_annotation_store,
)
from agent_diagnostics.cli.annotate import cmd_annotate, cmd_llm_annotate
from agent_diagnostics.cli.classify import cmd_blend, cmd_ensemble, cmd_predict, cmd_train
from agent_diagnostics.cli.data import (
    cmd_db_schema,
    cmd_export,
    cmd_manifest_refresh,
    cmd_pipeline_run,
    cmd_query,
)
from agent_diagnostics.cli.ingest import cmd_extract, cmd_ingest
from agent_diagnostics.cli.main import (
    _LOG_FORMAT,
    _configure_logging,
    _lookup_level_by_name,
    _resolve_log_level,
    main,
)
from agent_diagnostics.cli.reporting import (
    _collect_golden_corpus,
    cmd_agreement,
    cmd_calibrate,
    cmd_report,
    cmd_validate,
)

__all__ = [
    "_LOG_FORMAT",
    "_MODEL_ALIAS_TO_IDENTITY",
    "_annotations_to_narrow_rows",
    "_collect_golden_corpus",
    "_configure_logging",
    "_lookup_level_by_name",
    "_resolve_llm_annotator_identity",
    "_resolve_log_level",
    "_write_to_annotation_store",
    "cmd_agreement",
    "cmd_annotate",
    "cmd_blend",
    "cmd_calibrate",
    "cmd_db_schema",
    "cmd_ensemble",
    "cmd_export",
    "cmd_extract",
    "cmd_ingest",
    "cmd_llm_annotate",
    "cmd_manifest_refresh",
    "cmd_pipeline_run",
    "cmd_predict",
    "cmd_query",
    "cmd_report",
    "cmd_train",
    "cmd_validate",
    "main",
]
