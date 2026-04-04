# Agent Reliability Observatory

A behavioral taxonomy and annotation framework for analyzing why coding agents succeed or fail on benchmark tasks.

## Install

```bash
pip install agent-observatory
```

## Quick Start

```python
from agent_observatory import load_taxonomy, valid_category_names

# Load the 23-category behavioral taxonomy
taxonomy = load_taxonomy()
print(f"{len(taxonomy['categories'])} categories")

# Get valid category names
names = valid_category_names()
print(names)

# Validate an annotation
from agent_observatory import validate_annotation_categories

annotation = {
    "categories": [
        {"name": "retrieval_failure", "confidence": 0.9},
    ]
}
validate_annotation_categories(annotation)  # raises ValueError if invalid
```

## Taxonomy

The taxonomy organizes agent behaviors into three polarities:

| Polarity | Count | Purpose |
|----------|-------|---------|
| failure  | 16    | Explains why the agent failed or underperformed |
| success  | 5     | Explains which strategy led to success |
| neutral  | 2-3   | Contextual factors that affect interpretation |

## Taxonomy Versions

- **v1** (flat): Categories in a flat list with `name`, `description`, `polarity`, `detection_hints`, `examples`
- **v2** (hierarchical): Categories organized by dimension (Retrieval, Execution, etc.)

```python
from agent_observatory.taxonomy import load_taxonomy, _package_data_path

# Load v2 (hierarchical dimensions)
v2 = load_taxonomy(_package_data_path("taxonomy_v2.yaml"))
```

## Annotation Schema

The package includes a JSON Schema for machine-readable annotations:

```python
from agent_observatory.taxonomy import get_schema_path

schema_path = get_schema_path()
```

## Exemplars

25 hand-annotated examples covering all 23 taxonomy categories are bundled with the package under `exemplars/`.

## License

Apache-2.0
