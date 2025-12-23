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


__DevOps:__

+ **Setup orchestration, task scheduling with `rq`:**
  + <https://python-rq.org/patterns/>
+ **Finish integrating telemetry (need metrics to debug properly):**
  + <https://opentelemetry.io/docs/zero-code/python/logs-example/>
+ **Example illustrating how to setup Docker with uv + uvicorn:**
  + <https://uvicorn.dev/deployment/docker/#quickstart>


__Demos:__

+ ~~Complete a simple gradio-based LLM demo.~~
  + Time series plotting of /model/run results.
    + Use Gradio's plotting capabilities to visualize CMA model outputs over time.
  + Integrate with existing CMA demo UI.
  + Finish setting up as a docker-compose service.
  + Host on GCP (App Engine, utilize credits).


__Architecture:__

+ Gcp Fabric (terraform)
+ Use OpenRLHF (or similar for GCP native) for LLM safeguard experiments.
+ Continue implementing gemini demo:
  + <https://codelabs.developers.google.com/devsite/codelabs/gemini-multimodal-chat-assistant-python>

<img src="https://openrlhf.readthedocs.io/en/latest/_images/openrlhf-arch.png" />
