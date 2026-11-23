"""Keep temporary checkpoint/LoRA baking state out of Neo generation."""
from functools import wraps
import logging


def protect_bake(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        from modules import sd_models

        # This adapter targets Neo; leave the upstream A1111/Forge paths intact.
        if not hasattr(sd_models, "forge_model_reload") or hasattr(sd_models, "read_state_dict"):
            return function(*args, **kwargs)

        from modules.call_queue import queue_lock
        from modules.processing import StableDiffusionProcessing
        from modules_forge.main_entry import refresh_model_loading_parameters
        from scripts.A1111 import networks as bake_networks

        # Gradio queues different callbacks independently. Use the same lock as
        # txt2img/img2img while the baker changes shared.sd_model.
        with queue_lock:
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                try:
                    # These cached modules retain references to the temporary
                    # checkpoint, even after the regular model is unloaded.
                    bake_networks.loaded_networks.clear()
                    bake_networks.networks_in_memory.clear()
                    import lora
                    if hasattr(lora, "loaded_loras"):
                        lora.loaded_loras.clear()
                    if hasattr(sd_models, "checkpoints_loaded"):
                        sd_models.checkpoints_loaded.clear()
                    StableDiffusionProcessing.cached_c = [None, None, None]
                    StableDiffusionProcessing.cached_uc = [None, None, None]
                    try:
                        sd_models.unload_model_weights()
                    finally:
                        sd_models.model_data.forge_hash = ""
                        # Respect the UI selection and additional modules. Let
                        # normal generation perform the next clean model load.
                        refresh_model_loading_parameters()
                except Exception:
                    if not failed:
                        raise
                    logging.exception("SuperMerger Neo: cleanup failed after a baking error")
    return guarded
