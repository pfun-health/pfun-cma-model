# TODO.md

__TODO for `pfun-cma-model`__


__Goals:__

+ **Create a huggingface IterableDataset:**
  + Design as a parametric data factory (generate n_days of data, potentially with specified trends).
  + <https://huggingface.co/docs/datasets/v4.4.2/en/package_reference/main_classes#datasets.IterableDataset>
  + <https://huggingface.co/docs/datasets/en/create_dataset>
+ **LoRA training (likely better for use-case than finetuning):**
  + <https://huggingface.co/docs/trl/lora_without_regret#takeaways>
  + <https://huggingface.co/docs/trl/lora_without_regret.md>


__Maintenance:__

**This orchestration layer can likely be replaced with cloudflare worker load-balancing between api instances**
+ ~~Setup orchestration, task scheduling with `rq`:~~
  + ~~<https://python-rq.org/patterns/>~~

+ (Continue research) **Finish integrating telemetry (need metrics to debug properly):**
  + <https://opentelemetry.io/docs/zero-code/python/logs-example/>

__Demos:__

+ ~~Complete a simple gradio-based LLM demo.~~
  + ~~Time series plotting of /model/run results.~~
+ Model definitions for specialized health recommendation features
  + Transformer model
  + Vector-search (enhanced) RAG (RAFT, `LlamaIndex`)
+ **Evaluation demos:** Compare between a few systems that don't need fine-tuning.
+ **What specifically is the performance advantage?**
  + **Evaluate, compare overall performance (order-of-magnitude)**:
	+ `(vector-search + RAG) <--> (multi-stage RAG)`
    + `(Non-FT modeling approach) <--> (Fine-tuned LLM)`

__Infra:__

+ Gcp Fabric (terraform)
+ Use OpenRLHF (or similar for GCP native) for LLM safeguard experiments.
+ Continue implementing gemini demo:
  + <https://codelabs.developers.google.com/devsite/codelabs/gemini-multimodal-chat-assistant-python>

<img src="https://openrlhf.readthedocs.io/en/latest/_images/openrlhf-arch.png" />
