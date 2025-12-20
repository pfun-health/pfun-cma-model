# TODO.md

__TODO for `pfun-cma-model`__

__Maintenance:__

+ **Finish integrating telemetry (need metrics to debug properly):**
  + https://opentelemetry.io/docs/zero-code/python/logs-example/
+ **Example illustrating how to setup Docker with uv + uvicorn:**
  + https://uvicorn.dev/deployment/docker/#quickstart


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
