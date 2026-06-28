# API reference

Auto-generated from docstrings. The pipeline is the primary public API; the
orchestration stages and evaluation metrics are documented for programmatic use.

## Pipeline

::: trialmatchai.pipeline
    options:
      members:
        - StageContext
        - Stage
        - STAGES
        - select_stages
        - run_pipeline

## Orchestration stages

::: trialmatchai.orchestration
    options:
      members:
        - ingest_inputs
        - expand_queries
        - build_index
        - run_matching
        - prepare_corpus
        - build_system
        - build_state

## Registry updater

::: trialmatchai.registry.updater
    options:
      members:
        - RegistryUpdater
        - RegistryUpdateConfig
        - RegistryUpdateReport

## Evaluation metrics

::: trialmatchai.trec.metrics
    options:
      members:
        - ndcg_at_k
        - condensed_ndcg
        - precision_at_k

::: trialmatchai.trec.qrels
    options:
      members:
        - parse_qrels
        - corpus_ncts
        - evaluate
