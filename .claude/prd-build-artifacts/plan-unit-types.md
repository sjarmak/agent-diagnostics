# Plan: unit-types

## Steps

1. Create `src/agent_observatory/types.py`:
   - TrialSignals TypedDict with 26 keys
   - TrialInput Protocol
   - CategoryAssignment frozen dataclass
   - Annotation frozen dataclass
   - AnnotationDocument frozen dataclass

2. Create `tests/test_types.py`:
   - Test TrialSignals import and key count
   - Test TrialInput is a Protocol
   - Test frozen dataclasses (CategoryAssignment, Annotation, AnnotationDocument)
   - Test field types and defaults
   - Test immutability (frozen)

3. Run tests, fix any failures
