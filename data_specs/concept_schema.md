# Concept Schema (Draft)

## Purpose
Define report-derived clinical concepts used by concept and text explanation tracks.

## Source
- RadGraph entities and relations aligned to selected pathology tasks.

## Initial concept groups
- Observation concepts (for example: edema-like, effusion-like, opacity-like).
- Anatomical concepts (for example: pleural, cardiac silhouette, lower lobe).
- Modifier concepts (for example: bilateral, mild, increasing).

## Design constraints
- Concepts must be clinically meaningful and consistently extractable.
- Concepts should have enough support to train and evaluate reliably.
- Concept definitions must avoid overlap where possible.

## Pending decisions
- Final concept list and minimum frequency threshold.
- Whether to include relation-level concepts or entity-only concepts.
- Intervention protocol for concept edits at evaluation time.
