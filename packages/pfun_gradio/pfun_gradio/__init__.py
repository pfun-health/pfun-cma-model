__all__ = [
    "gradio_ui"
]
gradio_ui  = None
try:
    import pfun_gradio.pfun_gradio.gradio_ui  as gradio_ui  # noqa: F401
except ModuleNotFoundError as e:
    import logging
    logging.error("Failed to import gradio_ui: %s", e)
if not gradio_ui:
    try:
        import pfun_gradio.gradio_ui  as gradio_ui  # noqa: F401
    except ModuleNotFoundError as e:
        import logging
        logging.error("Failed to import gradio_ui from alternative path: %s", e)