# TODO.md

TODO for `pfun-cma-model`

Goals
=====

## Short-term Goals

+ _Complete initial implementation of admin interface:_
  + ~~Define models (User, Site)~~
  + ~~Implement views (CRUD for User, Site)~~
  + Integrate password hashing in User model
  + Test admin interface functionality
  + _Setup database migrations with Alembic:_
    + Configure Alembic for SQLite
    + Create initial migration for User and Site models
    + Test migration process (upgrade/downgrade)
+ _Configure production hosting (domain, routes, etc.):_
  + **Security, separation of concerns (frontend, backend, ...), load-balancing.**
  + _Domain configuration:_
    + **Landing page frontend** at: `pfun.one`, `pfun.me`.
    + **Demo frontend** at: `pfun.app`.
    + **Backend API** at: `api.pfun.run`.
  + _Routing and load-balancing:_
    + Cloudflare Workers for routing and load-balancing between API instances.
    + CDN for frontend assets (Still shopping around for best option here, but Cloudflare also offers CDN services).

## Overall Goal: Complete Evaluation Pipeline

### Curate Dataset(s)

+ _Create a huggingface IterableDataset:_
  + Design as a parametric data factory (generate n_days of data, potentially with specified trends).
  + <https://huggingface.co/docs/datasets/v4.4.2/en/package_reference/main_classes#datasets.IterableDataset>
  + <https://huggingface.co/docs/datasets/en/create_dataset>

### Model Development

+ _LoRA training (likely better for use-case than finetuning):_
  + <https://huggingface.co/docs/trl/lora_without_regret#takeaways>
  + <https://huggingface.co/docs/trl/lora_without_regret.md>

+ _RAFT (consider starting here for the near-term):_
  + Low-effort, relatively-performant variant of RAG.

DevOps:
=======

+ _Setup orchestration, task scheduling:_
  + _Compare options:_
    + Digital Ocean droplet (serverless)
    + Cloudflare Worker: load-balancing between api instances

+ _Finish integrating telemetry (need metrics to debug properly):_
  + NOTE: `fastapi-guard` includes telemetry routes

Demos:
======

+ _Datasets for training, counterfactuals, twin studies_
  + Consider MIMO (multi-input, multi-output) embedding approach for flexibility.
  + _ScenarioDataset:_
    + {X1: ScenarioConditionedParameters},
    + {X2: QualitativeDescription},
    + {X3: _ParameterSensitivityAnalysis_}  Jacobian: Key advantage of using world model (#DynamicalSystems)
+ Model definitions for specialized health recommendation features
  + Transformer model
  + Vector-search (enhanced) RAG (RAFT, `LlamaIndex`)
+ _Evaluation demos:_ Compare between a few systems that don't need fine-tuning.
+ _What specifically is the performance advantage?_
  + _Evaluate, compare overall performance (order-of-magnitude)_:
    + `(vector-search + RAG) <--> (multi-stage RAG)`
    + `(Non-FT modeling approach) <--> (Fine-tuned LLM)`

Infra:
======

+ Gcp Fabric (terraform)
+ Use OpenRLHF (or similar for GCP native) for LLM safeguard experiments.
+ Continue implementing gemini demo:
  + <https://codelabs.developers.google.com/devsite/codelabs/gemini-multimodal-chat-assistant-python>

<img src="https://openrlhf.readthedocs.io/en/latest/_images/openrlhf-arch.png" alt="openrlhf-arch.png" />  
