import gradio as gr
import traceback
from collections import deque
from queue import Empty, Queue
from threading import Event, Lock, Thread
import scripts.mergers.components as components
from scripts.mergers.mergers import smergegen, simggen, cachedealer
from scripts.mergers.xyplot import numanager
try:
    from scripts.mergers.pluslora import frompromptf
except ImportError as e:
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not installed"
    
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_version = "not installed"

    print(
        f"Version error: Failed to import module.\n"
        f"Transformers version: {transformers_version}\n"
        f"Diffusers version: {diffusers_version}\n"
        "Please ensure compatibility between these packages."
    )
    raise e

from modules import scripts, script_callbacks

_merge_lock = Lock()

# Merge-only does not read generation or dice controls, but smergegen shares
# its positional interface with Merge&Gen.
_MERGE_ONLY_GENERATION_DEFAULTS = (
    "", "", 1, "", 1, 0, 0, 0, 1,
    [], "", 0, 0.0, 1.0,
    "off", ["alpha"], "1", "0", "1", "0", -1, 1, "", 3,
)

def merge_only(*args):
    current_model = ""
    if not _merge_lock.acquire(blocking=False):
        message = "ERROR: Another merge is already running."
        yield message, current_model, message
        return

    updates = Queue()
    finished = Event()
    outcome = {}
    history = deque(maxlen=12)

    def report(message):
        print(f"SuperMerger Neo: {message}", flush=True)
        updates.put(message)

    def run_merge():
        try:
            outcome["result"] = smergegen(
                *args, *_MERGE_ONLY_GENERATION_DEFAULTS, current_model, False,
                status_callback=report,
            )
        except Exception as exc:
            traceback.print_exc()
            outcome["error"] = f"ERROR: {type(exc).__name__}: {exc}"
        finally:
            try:
                cachedealer(False)
            finally:
                _merge_lock.release()
                finished.set()

    report("Merge request received.")
    try:
        Thread(target=run_merge, name="supermerger-merge", daemon=True).start()
    except Exception:
        _merge_lock.release()
        raise

    while not finished.is_set() or not updates.empty():
        try:
            message = updates.get(timeout=0.5)
        except Empty:
            continue
        history.append(message)
        yield message, current_model, "\n".join(history)

    if "error" in outcome:
        message = outcome["error"]
        history.append(message)
        yield message, current_model, "\n".join(history)
        return

    result = outcome["result"]
    status = "Saved and loaded successfully." if result[0].startswith("Merged model saved in ") else result[0]
    history.append(status)
    yield result[0], result[1], "\n".join(history)

class GenParamGetter(scripts.Script):
    txt2img_gen_button = None
    img2img_gen_button = None

    events_assigned = False
    merge_events_assigned = False

    def title(self):
        return "Super Marger Generation Parameter Getter"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_wanted_params(params,wanted):
        output = []
        for target in wanted:
            if target is None:
                output.append(params[0])
                continue
            for param in params:
                if hasattr(param,"label"):
                    if param.label == target:
                        output.append(param)
        return output

    def after_component(self, component: gr.components.Component, **_kwargs):
        """Find generate button"""
        if component.elem_id == "txt2img_generate":
            GenParamGetter.txt2img_gen_button = component
        elif  component.elem_id == "img2img_generate":
            GenParamGetter.img2img_gen_button = component

    def get_components_by_ids(root: gr.Blocks, ids: list[int]):
        components: list[gr.Blocks] = []

        if root._id in ids:
            components.append(root)
            ids = [_id for _id in ids if _id != root._id]
        
        if hasattr(root,"children"):
            for block in root.children:
                components.extend(GenParamGetter.get_components_by_ids(block, ids))
        return components
    
    def compare_components_with_ids(components: list[gr.Blocks], ids: list[int]):
        try:
            return len(components) == len(ids) and all(component._id == _id for component, _id in zip(components, ids))
        except:
            return False

    def get_params_components(demo: gr.Blocks, app):
        # Merge-only needs no txt2img parameters. Bind it before inspecting
        # generation buttons so a changed Forge UI cannot disable Merge!.
        if not GenParamGetter.merge_events_assigned:
            with demo:
                components.merge.click(
                    fn=None,
                    inputs=[],
                    outputs=[components.merge_status],
                    js="() => 'Merge requested. Waiting for processing to start...'",
                    queue=False,
                    show_progress=False,
                )
                components.merge.click(
                    fn=merge_only,
                    inputs=components.msettings,
                    outputs=[components.submit_result,components.currentmodel,components.merge_status],
                )
            GenParamGetter.merge_events_assigned = True
            print("SuperMerger Neo: Merge! handlers registered")

        if GenParamGetter.txt2img_gen_button is None or GenParamGetter.img2img_gen_button is None:
            print("SuperMerger Neo: generation buttons not found; merge-only remains available")
            return

        for _id, _is_txt2img in zip([GenParamGetter.txt2img_gen_button._id, GenParamGetter.img2img_gen_button._id], [True, False]):
            if hasattr(demo,"dependencies"):
                #dependencies: list[dict] = [x for x in demo.dependencies if x["trigger"] == "click" and _id in x["targets"]]
                dependencies: list[dict] = [x for x in demo.dependencies if _id in x["targets"]]
                g4 = False
                
            else:
                dependencies: list[dict] = [x for x in demo.config["dependencies"] if x["targets"][0][1] == "click" and _id in x["targets"][0]]
                g4 = True
            
            dependency: dict = None

            for d in dependencies:
                if len(d["outputs"]) == 4 or len(d["outputs"]) == 5:
                    dependency = d
                    print("GenParamsGetter detected!")

            if dependency is None:
                print("SuperMerger Neo: generation parameters not found; merge-only remains available")
                return
            
            if g4:
                params = [demo.blocks[x] for x in dependency['inputs']]
                if _is_txt2img:
                    components.paramsnames = [x.label if hasattr(x,"label") else "None" for x in params]

                if _is_txt2img:
                    components.txt2img_params = params
                else:
                    components.img2img_params = params
            else:
                if dependency is None:continue

                params = [params for params in demo.fns if GenParamGetter.compare_components_with_ids(params.inputs, dependency["inputs"])]

                if _is_txt2img:
                    components.paramsnames = [x.label if hasattr(x,"label") else "None" for x in params[0].inputs]

                if _is_txt2img:
                    components.txt2img_params = params[0].inputs 
                else:
                    components.img2img_params = params[0].inputs
        
        if not GenParamGetter.events_assigned:
            with demo:
                components.mergeandgen.click(
                    fn=smergegen,
                    inputs=[*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,components.currentmodel,components.dtrue,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal]
                )

                components.gen.click(
                    fn=simggen,
                    inputs=[*components.genparams,*components.hiresfix,components.currentmodel,components.id_sets,gr.Textbox(value="No id",visible=False),*components.txt2img_params],
                    outputs=[*components.imagegal],
                )

                components.s_reserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.numaframe]
                )

                components.s_reserve1.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.numaframe]
                )

                components.gengrid.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="normal",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.s_startreserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value=" ",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.rand_merge.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="random",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.frompromptb.click(
                    fn=frompromptf,
                    inputs=[*components.txt2img_params],
                    outputs=components.sml_loranames,
                )
            GenParamGetter.events_assigned = True

if __package__ == "GenParamGetter":
    script_callbacks.on_app_started(GenParamGetter.get_params_components)
