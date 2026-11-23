import gc
import hashlib
import json
import math
import os
import re
import csv
import sys
import traceback
from io import BytesIO
import gradio as gr
import launch
import modules.shared as shared
import numpy as np
import safetensors.torch
import scripts.mergers.components as components
import torch
from modules import extra_networks, scripts, sd_models, launch_utils
from modules.ui import create_refresh_button
from safetensors.torch import load_file, save_file
from scripts.kohyas import extract_lora_from_models as ext
from scripts.A1111 import networks as nets
from scripts.mergers.model_util import filenamecutter, savemodel
from scripts.mergers.bake_lifecycle import protect_bake
from scripts.mergers.mergers import extract_super, unload_forge, q_dequantize, q_quantize, qdtyper, prefixer, BLOCKIDFLUX
from tqdm import tqdm

forge = hasattr(sd_models, "forge_model_reload") or launch_utils.git_tag()[0:2] == "f2" or launch_utils.git_tag() == "neo"

selectable = []
pchanged = False
MAX_UI_LORA_CHOICES = 500

def lora_matches_query(name, lora_on_disk, query):
    terms = [term.lower() for term in str(query or "").split() if term.strip()]
    if not terms:
        return True
    filename = getattr(lora_on_disk, "filename", "") if lora_on_disk is not None else ""
    haystack = f"{name} {filename}".lower().replace("\\", "/")
    return all(term in haystack for term in terms)

def normalize_lora_choice(name):
    name = str(name or "")
    return name[:name.find("[")] if "[" in name else name


def selected_loras_to_loranames(selected, ratio):
    selected = selected or []
    out = []
    for item in selected:
        name = normalize_lora_choice(item).strip()
        if name:
            out.append(f"{name}:{ratio}")
    return ",".join(out)

def limited_lora_choices(lora_module, query=""):
    matches = []
    total = 0
    for name, lora_on_disk in lora_module.available_loras.items():
        total += 1
        if lora_matches_query(name, lora_on_disk, query):
            matches.append(name)
        if len(matches) >= MAX_UI_LORA_CHOICES:
            break
    if total > MAX_UI_LORA_CHOICES or query:
        print(f"SuperMerger Neo: showing {len(matches)} LoRA choices (limit {MAX_UI_LORA_CHOICES}, query='{query or ''}').")
    return matches

CUDA = torch.device("cuda:0")
CPU = torch.device("cpu")

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKIDANIMA=["BASE"] + [f"B{x:02d}" for x in range(28)]
BLOCKNUMS = [12,17,20,26,29,61]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26,BLOCKIDANIMA]
ANIMA_KEY_MARKERS = (
    "lora_unet_blocks_",
    "lora_unet_patch_embed",
    "lora_unet_final_layer",
    "lora_unet_t_embedder",
    "lora_te_text_encoders_",
    "lora_te_llm_adapter_",
)
ANIMA_QWEN_PREFIXES = (
    "text_encoders_qwen25_7b_",
    "text_encoders_qwen3_06b_transformer_",
    "text_encoders_qwen3_transformer_",
)
ANIMA_QWEN_ALIAS_TO_DOT = {
    "text_encoders_qwen25_7b_": "text_encoders.qwen25_7b.",
    "text_encoders_qwen3_06b_transformer_": "text_encoders.qwen3_06b.transformer.",
    "text_encoders_qwen3_transformer_": "text_encoders.qwen3.transformer.",
}
ANIMA_QWEN_MODEL_LAYER_PREFIXES = (
    ("text_encoders_qwen25_7b_model_layers_", "text_encoders.qwen25_7b.model.layers."),
    ("text_encoders_qwen3_06b_transformer_model_layers_", "text_encoders.qwen3_06b.transformer.model.layers."),
    ("text_encoders_qwen3_transformer_model_layers_", "text_encoders.qwen3.transformer.model.layers."),
)

def to26(ratios):
    if len(ratios) in (26, 29) or len(ratios) > 40 : return ratios
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    return output

def to61s(ratioss):
    out = []
    for ratios in ratioss:
        if len(ratios) == 61:
            out.append(ratios)
        else:
            out.append(ratios + [ratios[0]] * (61 - len(ratios)))
    return out


def default_ratio_for_length(length, value):
    return [float(value)] * length

def detect_ratio_length_from_text(text):
    count = text.count(",") + 1
    if count in BLOCKNUMS:
        return count
    return 26

def default_ratio_length_for_lora(filename):
    try:
        header = load_state_header(filename, torch.float)
        if any(is_anima_lora_key(key) for key in header.keys()):
            return 29
    except Exception:
        pass
    return 26

def split_lora_entries(text):
    entries = []
    current = []
    angle_depth = 0
    for char in text:
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        if char == "," and angle_depth == 0:
            entry = "".join(current).strip()
            if entry:
                entries.append(entry)
            current = []
        else:
            current.append(char)
    entry = "".join(current).strip()
    if entry:
        entries.append(entry)
    return entries

def parse_lora_prompt_entry(entry):
    entry = entry.strip()
    if not (entry.startswith("<") and entry.endswith(">")):
        return None

    content = entry[1:-1].strip()
    parts = [part.strip() for part in content.split(":")]
    if len(parts) < 2 or parts[0].lower() not in ("lora", "lyco"):
        return None

    name = parts[1]
    multiplier = parts[2] if len(parts) > 2 and parts[2] else "1"
    lbw_value = None
    for part in parts[3:]:
        if part.startswith("lbw="):
            lbw_value = part.replace("lbw=", "", 1)
            break
    return name, multiplier, lbw_value

def parse_lora_merge_entries(text):
    parsed = []
    for entry in split_lora_entries(text):
        prompt_entry = parse_lora_prompt_entry(entry)
        if prompt_entry is not None:
            name, multiplier, lbw_value = prompt_entry
            parsed.append([name, multiplier, lbw_value] if lbw_value else [name, multiplier])
            continue

        if ":" in entry:
            parsed.append([part.strip() for part in entry.split(":")])
        elif parsed:
            parsed[-1].append(entry.strip())
        elif entry:
            parsed.append([entry.strip(), "1"])
    return parsed

def ratio_at(ratios, index):
    if index < len(ratios):
        return ratios[index]
    return ratios[0] if len(ratios) > 0 else 1.0

def is_anima_lora_key(key):
    return any(marker in key for marker in ANIMA_KEY_MARKERS)

def is_lycoris_state_dict(state_dict):
    lyco_suffixes = (
        ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
        ".lokr_w1", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
        ".oft_blocks", ".oft_diag", ".w_norm", ".b_norm",
    )
    return any(key.endswith(lyco_suffixes) for key in state_dict.keys())

def is_anima_checkpoint_key(key):
    return (
        key.startswith("model.diffusion_model.blocks.")
        or key.startswith("model.diffusion_model.patch_embed.")
        or key.startswith("model.diffusion_model.final_layer.")
        or key.startswith("model.diffusion_model.t_embedder.")
        or key.startswith("net.blocks.")
        or key.startswith("net.patch_embed.")
        or key.startswith("net.final_layer.")
        or key.startswith("net.t_embedder.")
        or key.startswith("net.llm_adapter.")
        or key.startswith("text_encoders.")
        or key.startswith("llm_adapter.")
        or ".model.layers." in key
        or "_model_layers_" in key
        or key.startswith("model.layers.")
        or key.startswith("model_layers_")
    )

def register_key_alias(keychanger, alias, original_key):
    if not alias:
        return
    existing = keychanger.get(alias)
    if existing is None:
        keychanger[alias] = original_key
        return
    # Prefer weight tensors when multiple checkpoint tensors share the same alias.
    if existing.endswith(".bias") and original_key.endswith(".weight"):
        keychanger[alias] = original_key

def suffix_after_prefix(value, prefix):
    if value.startswith(prefix):
        return value[len(prefix):]
    return None

def strip_tensor_suffix(key):
    for suffix in (".weight", ".bias", ".alpha", "_weight", "_bias", "_alpha"):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return key

def anima_mlp_alias_variants(value):
    variants = {value}
    replacements = (
        (".mlp.0.", ".mlp.layer1."),
        (".mlp.2.", ".mlp.layer2."),
        ("_mlp_0_", "_mlp_layer1_"),
        ("_mlp_2_", "_mlp_layer2_"),
    )
    changed = True
    while changed:
        changed = False
        current = list(variants)
        for item in current:
            for source, target in replacements:
                # Module names also end directly in mlp.0 / mlp.layer1.
                if item.endswith(source[:-1]):
                    variant = item[:-len(source[:-1])] + target[:-1]
                    if variant not in variants:
                        variants.add(variant)
                        changed = True
                if item.endswith(target[:-1]):
                    variant = item[:-len(target[:-1])] + source[:-1]
                    if variant not in variants:
                        variants.add(variant)
                        changed = True
                if source in item and item.replace(source, target) not in variants:
                    variants.add(item.replace(source, target))
                    changed = True
                if target in item and item.replace(target, source) not in variants:
                    variants.add(item.replace(target, source))
                    changed = True
    return variants

def _anima_join_tokens(tokens):
    parts = []
    i = 0
    while i < len(tokens):
        if i + 3 < len(tokens) and tokens[i] == "adaln" and tokens[i + 1] == "modulation" and tokens[i + 2] in {"cross", "self"} and tokens[i + 3] == "attn":
            parts.append(f"adaln_modulation_{tokens[i + 2]}_attn")
            i += 4
            continue
        if i + 2 < len(tokens) and tokens[i] == "adaln" and tokens[i + 1] == "modulation":
            parts.append(f"adaln_modulation_{tokens[i + 2]}")
            i += 3
            continue
        if i + 1 < len(tokens) and tokens[i] in {"cross", "self"} and tokens[i + 1] == "attn":
            parts.append(f"{tokens[i]}_attn")
            i += 2
            continue
        if i + 1 < len(tokens) and tokens[i] in {"q", "k"} and tokens[i + 1] == "norm":
            parts.append(f"{tokens[i]}_norm")
            i += 2
            continue
        if i + 1 < len(tokens) and tokens[i] in {"q", "k", "v", "output"} and tokens[i + 1] == "proj":
            parts.append(f"{tokens[i]}_proj")
            i += 2
            continue
        parts.append(tokens[i])
        i += 1
    return parts

def anima_dot_name_from_alias(alias):
    prefixes = ("model_diffusion_model_", "diffusion_model_", "net_")
    for prefix in prefixes:
        if alias.startswith(prefix):
            suffix = alias[len(prefix):]
            tokens = _anima_join_tokens(suffix.split("_"))
            return prefix.rstrip("_") + "." + ".".join(tokens)
    if alias.startswith("llm_adapter_"):
        suffix = alias[len("llm_adapter_"):]
        tokens = _anima_join_tokens(suffix.split("_"))
        return "llm_adapter." + ".".join(tokens)
    if alias.startswith("text_encoders_"):
        suffix = alias[len("text_encoders_"):]
        tokens = _anima_join_tokens(suffix.split("_"))
        return "text_encoders." + ".".join(tokens)
    for prefix, dot_prefix in ANIMA_QWEN_ALIAS_TO_DOT.items():
        if alias.startswith(prefix):
            suffix = alias[len(prefix):]
            tokens = _anima_join_tokens(suffix.split("_"))
            return dot_prefix + ".".join(tokens)
    if alias.startswith("model_layers_"):
        suffix = alias[len("model_layers_"):]
        tokens = _anima_join_tokens(suffix.split("_"))
        return "model.layers." + ".".join(tokens)
    return None

def anima_key_aliases_from_checkpoint_key(key):
    aliases = set()
    base_key = strip_tensor_suffix(key)
    normalized = base_key.replace(".", "_")
    aliases.add(base_key)
    aliases.add(normalized)
    aliases.update(anima_mlp_alias_variants(base_key))
    aliases.update(anima_mlp_alias_variants(normalized))

    if normalized.startswith("model_diffusion_model_"):
        suffix = normalized.split("model_diffusion_model_", 1)[1]
        aliases.add(suffix)
        aliases.add(f"diffusion_model_{suffix}")
        aliases.add(f"net_{suffix}")
        model_dot = anima_dot_name_from_alias(normalized)
        if model_dot:
            aliases.add(model_dot)
            model_suffix = suffix_after_prefix(model_dot, "model.diffusion_model.")
            if model_suffix:
                aliases.add(model_suffix)
        suffix_dot = anima_dot_name_from_alias(f"diffusion_model_{suffix}")
        if suffix_dot:
            aliases.add(suffix_dot)
        net_dot = anima_dot_name_from_alias(f"net_{suffix}")
        if net_dot:
            aliases.add(net_dot)

    if normalized.startswith("net_"):
        suffix = normalized.split("net_", 1)[1]
        aliases.add(suffix)
        aliases.add(f"diffusion_model_{suffix}")
        aliases.add(f"model_diffusion_model_{suffix}")
        net_dot = anima_dot_name_from_alias(normalized)
        if net_dot:
            aliases.add(net_dot)
            net_suffix = suffix_after_prefix(net_dot, "net.")
            if net_suffix:
                aliases.add(net_suffix)
        suffix_dot = anima_dot_name_from_alias(f"diffusion_model_{suffix}")
        if suffix_dot:
            aliases.add(suffix_dot)
        model_dot = anima_dot_name_from_alias(f"model_diffusion_model_{suffix}")
        if model_dot:
            aliases.add(model_dot)

    if normalized.startswith("text_encoders_"):
        aliases.add(normalized)
        aliases.add(normalized.split("text_encoders_", 1)[1])
        text_dot = anima_dot_name_from_alias(normalized)
        if text_dot:
            aliases.add(text_dot)
            text_suffix = suffix_after_prefix(text_dot, "text_encoders.")
            if text_suffix:
                aliases.add(text_suffix)
        for prefix in ANIMA_QWEN_PREFIXES:
            if normalized.startswith(prefix):
                aliases.add(normalized.split(prefix, 1)[1])
                aliases.add(normalized.replace(prefix, ""))
                qwen_dot = anima_dot_name_from_alias(normalized)
                if qwen_dot:
                    aliases.add(qwen_dot)
                    qwen_suffix = suffix_after_prefix(qwen_dot, "text_encoders.")
                    if qwen_suffix:
                        aliases.add(qwen_suffix)
                    if ".model.layers." in qwen_dot:
                        aliases.add("model.layers." + qwen_dot.split(".model.layers.", 1)[1])
                        aliases.add("layers." + qwen_dot.split(".model.layers.", 1)[1])

    if normalized.startswith("llm_adapter_"):
        aliases.add(normalized)
        llm_dot = anima_dot_name_from_alias(normalized)
        if llm_dot:
            aliases.add(llm_dot)

    if normalized.startswith("model_layers_") or "_model_layers_" in normalized:
        if normalized.startswith("model_layers_"):
            model_layers_name = normalized
        else:
            model_layers_name = "model_layers_" + normalized.split("_model_layers_", 1)[1]
        aliases.add(model_layers_name)
        aliases.add(model_layers_name.split("model_", 1)[1])
        aliases.add(f"layers_{model_layers_name.split('model_layers_', 1)[1]}")
        model_dot = anima_dot_name_from_alias(model_layers_name)
        if model_dot:
            aliases.add(model_dot)
            model_layers_suffix = suffix_after_prefix(model_dot, "model.layers.")
            if model_layers_suffix:
                aliases.add(model_layers_suffix)

    return aliases

def module_key_from_fullkey(fullkey):
    return strip_tensor_suffix(fullkey)

def candidate_checkpoint_module_keys(msd_key):
    candidates = []

    def add(value):
        if value and value not in candidates:
            candidates.append(value)

    base = strip_tensor_suffix(msd_key)
    add(base)

    if "." in base:
        add(base.replace(".", "_"))
    if "_" in base:
        dot_name = anima_dot_name_from_alias(base)
        if dot_name:
            add(dot_name)

    if base.startswith("diffusion_model_"):
        suffix = base[len("diffusion_model_"):]
        add(f"model_diffusion_model_{suffix}")
        add(f"net_{suffix}")
        add(f"diffusion_model.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
        add(f"model.diffusion_model.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
        add(f"net.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
    elif base.startswith("diffusion_model."):
        suffix = base[len("diffusion_model."):]
        add(f"model.diffusion_model.{suffix}")
        add(f"net.{suffix}")
    elif base.startswith("model_diffusion_model_"):
        suffix = base[len("model_diffusion_model_"):]
        add(f"diffusion_model_{suffix}")
        add(f"net_{suffix}")
    elif base.startswith("model.diffusion_model."):
        suffix = base[len("model.diffusion_model."):]
        add(f"diffusion_model.{suffix}")
        add(f"net.{suffix}")
    elif base.startswith("net_"):
        suffix = base[len("net_"):]
        add(f"diffusion_model_{suffix}")
        add(f"model_diffusion_model_{suffix}")
        add(f"diffusion_model.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
        add(f"model.diffusion_model.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
    elif base.startswith("net."):
        suffix = base[len("net."):]
        add(f"diffusion_model.{suffix}")
        add(f"model.diffusion_model.{suffix}")
    elif base.startswith("layers_"):
        suffix = base[len("layers_"):]
        add(f"model_layers_{suffix}")
        add(f"model.layers.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
        suffix_dot = ".".join(_anima_join_tokens(suffix.split('_')))
        for underscored_prefix, dotted_prefix in ANIMA_QWEN_MODEL_LAYER_PREFIXES:
            add(f"{underscored_prefix}{suffix}")
            add(f"{dotted_prefix}{suffix_dot}")
    elif base.startswith("model_layers_"):
        suffix = base[len("model_layers_"):]
        add(f"layers_{suffix}")
        add(f"model.layers.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
        suffix_dot = ".".join(_anima_join_tokens(suffix.split('_')))
        for underscored_prefix, dotted_prefix in ANIMA_QWEN_MODEL_LAYER_PREFIXES:
            add(f"{underscored_prefix}{suffix}")
            add(f"{dotted_prefix}{suffix_dot}")
    elif base.startswith("model.layers."):
        suffix = base[len("model.layers."):]
        add(f"model_layers_{suffix.replace('.', '_')}")
        add(f"layers_{suffix.replace('.', '_')}")
        suffix_us = suffix.replace('.', '_')
        for underscored_prefix, dotted_prefix in ANIMA_QWEN_MODEL_LAYER_PREFIXES:
            add(f"{underscored_prefix}{suffix_us}")
            add(f"{dotted_prefix}{suffix}")
    elif any(base.startswith(underscored_prefix) for underscored_prefix, _ in ANIMA_QWEN_MODEL_LAYER_PREFIXES):
        suffix = base.split("_model_layers_", 1)[1]
        add(f"model_layers_{suffix}")
        add(f"layers_{suffix}")
        add(f"model.layers.{'.'.join(_anima_join_tokens(suffix.split('_')))}")
    elif any(base.startswith(dotted_prefix) for _, dotted_prefix in ANIMA_QWEN_MODEL_LAYER_PREFIXES):
        suffix = base.split(".model.layers.", 1)[1]
        add(f"model_layers_{suffix.replace('.', '_')}")
        add(f"layers_{suffix.replace('.', '_')}")
        add(f"model.layers.{suffix}")
    expanded = []
    for candidate in candidates:
        for variant in anima_mlp_alias_variants(candidate):
            if variant not in expanded:
                expanded.append(variant)
    return expanded

def resolve_checkpoint_weight_key(msd_key, keychanger, state_dict=None, expected_shape=None):
    fallback = None
    for candidate in candidate_checkpoint_module_keys(msd_key):
        if candidate not in keychanger:
            continue
        resolved_key = keychanger[candidate]
        if fallback is None:
            fallback = (resolved_key, candidate)
        if state_dict is not None and expected_shape is not None:
            tensor = state_dict.get(resolved_key)
            if tensor is not None and tuple(tensor.shape) == tuple(expected_shape):
                return resolved_key, candidate
        else:
            return resolved_key, candidate
    if fallback is not None:
        return fallback
    return None, msd_key

def f_changediffusers(version):
    launch.run_pip(f"install diffusers=={version}", f"diffusers ver {version}")

def on_ui_tabs():
    import lora
    global selectable
    selectable = limited_lora_choices(lora)
    sml_path_root = scripts.basedir()
    LWEIGHTSPRESETS="\
    NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
    INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
    OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
    OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
    OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
    ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    lbwpath = os.path.join(sml_path_root,"scripts", "lbwpresets.txt")
    lbwpathn = os.path.join(sml_path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
    dimpath = os.path.join(sml_path_root,"extensions","sd-webui-supermerger-neo","loradims.csv")
    sml_lbwpresets=""

    if os.path.isfile(lbwpath):
        with open(lbwpath,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    elif os.path.isfile(lbwpathn):
        with open(lbwpathn,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    else:
        sml_lbwpresets=LWEIGHTSPRESETS

    try:
        import diffusers
        d_ver = diffusers.__version__
    except:
        d_ver = None

    with gr.Blocks(analytics_enabled=False) :
        sml_submit_result = gr.Textbox(label="Message", value="")
        with gr.Row(equal_height=False):
            with gr.Column():
                sml_cpmerge = gr.Button(elem_id="model_merger_merge", value="Merge to Checkpoint(Model A)",variant='primary')
                sml_merge = gr.Button(elem_id="model_merger_merge", value="Merge LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_settings = gr.CheckboxGroup(["same to Strength", "overwrite"], label="settings")
                    sml_filename = gr.Textbox(label="filename(option)",lines=1,visible =True,interactive  = True)  
                sml_metasettings = gr.Radio(value = "create new",choices = ["create new","create new without output_name", "merge","save all", "use first lora"], label="metadata")
                with gr.Row(equal_height=False):
                    save_precision = gr.Radio(label = "save precision",choices=["float","fp16","bf16"],value = "fp16",type="value")
                    calc_precision = gr.Radio(label = "calc precision(fp16:cuda only)" ,choices=["float","fp16","bf16"],value = "float",type="value")
                    device = gr.Radio(label = "device",choices=["cuda","cpu"],value = "cuda",type="value")
            with gr.Column():
                sml_makelora = gr.Button(elem_id="model_merger_merge", value="Make LoRA (alpha * Model A - beta * Model B)",variant='primary')
                sml_extract = gr.Button(elem_id="model_merger_merge", value="Extract from two LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model A",interactive=True)
                    create_refresh_button(sml_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    sml_model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model B",interactive=True)
                    create_refresh_button(sml_model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=1)
                    beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=1)
                    smooth = gr.Slider(label="gamma(smooth)", minimum=-1, maximum=20, step=0.1, value=1)
        
        sml_dim = gr.Radio(label = "remake dimension",choices = ["no","auto",4,8,16,32,64,128,256,512,768,1024],value = "no",type = "value") 
        sml_loranames = gr.Textbox(label='LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,...(":blocks" is option, not necessary)',lines=1,value="",visible =True)
        sml_loratypes = gr.CheckboxGroup(show_label=False, choices= ["LoRA", "LoCon", "Others"], value=["LoRA", "LoCon", "Others"])
        sml_dims = gr.CheckboxGroup(label = "1.X/2.X",choices=[],value = [],type="value",interactive=True,visible = False)
        sml_dims_xl = gr.CheckboxGroup(label = "XL",choices=[],value = [],type="value",interactive=True,visible = False)
        sml_dims_flux = gr.CheckboxGroup(label = "Flux",choices=[],value = [],type="value",interactive=True,visible = False)
        with gr.Row(equal_height=False):
            sml_calcdim = gr.Button(elem_id="calcloras", value="Calculate LoRA dimensions",variant='primary')
            sml_calcsets = gr.CheckboxGroup(choices=["Save as CSV","Load from CSV"],show_label=False)
            sml_update = gr.Button(elem_id="calcloras", value="update list",variant='primary')
            sml_lratio = gr.Slider(label="default LoRA multiplier", minimum=-1.0, maximum=2, step=0.1, value=1)
            sml_apply_ratio = gr.Button(value="選択LoRAへ強度を反映")
        gr.Markdown("焼き込みは **LoRAname:ratio 欄の値**を使います。例: `my_lora:-0.5` は差分を50%引きます。スライダー変更後は「選択LoRAへ強度を反映」を押してください。個別の強度や階層設定を入力済みの場合、その欄を直接編集できます。")

        with gr.Row():
            sml_selectall = gr.Button(elem_id="sml_selectall", value="select all",variant='primary')
            sml_deselectall = gr.Button(elem_id="slm_deselectall", value="deselect all",variant='primary')
            components.frompromptb = gr.Button(elem_id="slm_deselectall", value="get from prompt",variant='primary')
            hidenb = gr.Checkbox(value = False,visible = False)
        with gr.Row():
            sml_lora_search = gr.Textbox(label="LoRA search", placeholder="name/path terms, space = AND", lines=1)
            sml_lora_search_btn = gr.Button(value="search LoRA", variant='primary')
            sml_lora_clear_btn = gr.Button(value="clear")
        sml_loras = gr.CheckboxGroup(label = "LoRAs on disk",choices = selectable,type="value",interactive=True,visible = True)
        sml_loraratios = gr.TextArea(label="",value="",visible =True,interactive  = True)
        with gr.Row():
            sml_load_lbw_presets = gr.Button(value="load block weight presets")
            sml_clear_lbw_presets = gr.Button(value="clear block weight presets")

        sml_selectall.click(fn = lambda:gr.update(value = selectable),outputs = [sml_loras])
        sml_deselectall.click(fn = lambda:gr.update(value =[]),outputs = [sml_loras])
        sml_load_lbw_presets.click(fn=lambda: sml_lbwpresets, outputs=[sml_loraratios])
        sml_clear_lbw_presets.click(fn=lambda: "", outputs=[sml_loraratios])

        with gr.Row():
            changediffusers = gr.Button(elem_id=f"change_diffusers_version", value=f"change diffusers version(now:{d_ver})",variant='primary')
            dversion = gr.Textbox(label="diffusers version",lines=1,visible =True,interactive  = True)  
        components.sml_loranames = [sml_loras, sml_loranames, hidenb]

        changediffusers.click(
            fn=f_changediffusers,
            inputs=[dversion],
            outputs=[sml_submit_result]
        )

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loras,sml_lratio,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = True,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_extract.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loras,sml_lratio,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = False,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a,sml_model_b,sml_dim,sml_filename,sml_settings,alpha,beta,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=pluslora,
            inputs=[sml_loranames,sml_loras,sml_lratio,sml_loraratios,sml_settings,sml_filename,sml_model_a,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )

        ldict = {}

        def toselect(input):
            out = []
            for name, vals in input.items():
                if (not isinstance(vals, list)) or len(vals) != 3: continue
                dim, ltype, sdver = vals
                add = [] if dim == "LyCORIS" else [str(dim)]
                if ltype != "LoRA": add +=[ltype]
                if sdver != "1.X/2.X": add += [sdver]
                out.append(f"{name}[{','.join(add)}]" if add != ["","",""] else f"{name}")
            return out

        def limited_lora_dict(source):
            out = {}
            for name, vals in source.items():
                out[name] = vals
                if len(out) >= MAX_UI_LORA_CHOICES:
                    break
            return out

        def normalize_lora_choice(name):
            name = str(name or "")
            return name[:name.find("[")] if "[" in name else name

        def selected_loras_to_loranames(selected, ratio):
            selected = selected or []
            out = []
            for item in selected:
                name = normalize_lora_choice(item).strip()
                if name:
                    out.append(f"{name}:{ratio}")
            return ",".join(out)

        sml_apply_ratio.click(fn=selected_loras_to_loranames, inputs=[sml_loras, sml_lratio], outputs=[sml_loranames])

        def updateloras(query="", selected=None):
            lora.list_available_loras()
            query = str(query or "").strip()
            selected_names = []
            for item in selected or []:
                name = normalize_lora_choice(item)
                if name and name in lora.available_loras and name not in selected_names:
                    selected_names.append(name)
                    if name not in ldict:ldict[name] = ["","",""]

            names = []
            matched = 0
            total = len(lora.available_loras)
            for name, lora_on_disk in lora.available_loras.items():
                if not lora_matches_query(name, lora_on_disk, query):
                    continue
                matched += 1
                if len(names) < MAX_UI_LORA_CHOICES:
                    if name not in ldict:ldict[name] = ["","",""]
                    names.append(name)

            global selectable
            visible_names = selected_names + [name for name in names if name not in selected_names]
            visible = {name: ldict[name] for name in visible_names if name in ldict}
            selectable = toselect(visible)
            selected_values = toselect({name: ldict[name] for name in selected_names if name in ldict})
            if query:
                message = f"LoRA search: showing {len(names)} of {matched} matches for '{query}' (total {total}, limit {MAX_UI_LORA_CHOICES}, selected {len(selected_names)} kept)."
            else:
                message = f"LoRA list: showing first {len(names)} of {total} LoRAs. Use LoRA search to find items beyond the first {MAX_UI_LORA_CHOICES}. Selected {len(selected_names)} kept."
            return gr.update(choices = selectable, value=selected_values), message

        def search_loras(query, selected):
            return updateloras(query, selected)

        def clear_lora_search(selected):
            choices, message = updateloras("", selected)
            return "", choices, message

        sml_update.click(fn = updateloras, inputs=[sml_lora_search, sml_loras], outputs = [sml_loras, sml_submit_result])
        sml_lora_search_btn.click(fn=search_loras, inputs=[sml_lora_search, sml_loras], outputs=[sml_loras, sml_submit_result])
        sml_lora_search.submit(fn=search_loras, inputs=[sml_lora_search, sml_loras], outputs=[sml_loras, sml_submit_result])
        sml_lora_clear_btn.click(fn=clear_lora_search, inputs=[sml_loras], outputs=[sml_lora_search, sml_loras, sml_submit_result])

        def makedimlist(ver):
            outs = []
            outs_list = []
            for dim, _, sdver in ldict.values():
                if sdver == ver or (ver == "1.X/2.X" and dim == "unknown"):
                    if isinstance(dim, int):
                        if dim not in outs:
                            outs.append(dim)
                    else:
                        if dim not in outs_list:
                            outs_list.append(dim)
            outs = sorted(set(outs))
            return outs + outs_list

        def calculatedim(calcsets, device):
            # Load cached LoRA dimensions from CSV.
            if "Load from CSV" in calcsets:
                with open(dimpath, mode='r', encoding='utf-8') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        ldict[row[0]] = row[1:]

            print("listing dimensions...")
            for n in tqdm(lora.available_loras.items()):
                name = n[0] 
                if name in ldict and ldict[n[0]] != ["","",""]:
                    continue
                c_lora = lora.available_loras.get(n[0], None) 
                
                try:
                    d, t, s = dimgetter(c_lora.filename, device)
                except:
                    d, t, s = dimgetter(c_lora.filename)
                    
                ldict[name] = [d,t,s]

            # Save cached LoRA dimensions to CSV.
            if "Save as CSV" in calcsets:
                with open(dimpath, mode='w', encoding='utf-8', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for key, value in ldict.items():
                        csv_writer.writerow([key, *value])

            global selectable
            selectable = toselect(limited_lora_dict(ldict))
            return (gr.update(choices=selectable, value=[]), 
                    gr.update(visible=True, choices=makedimlist("1.X/2.X")),
                    gr.update(visible=True, choices=makedimlist("XL")),
                    gr.update(visible=True, choices=makedimlist("Flux"))
            )

        sml_calcdim.click(
            fn=calculatedim,
            inputs=[sml_calcsets, device],
            outputs=[sml_loras,sml_dims,sml_dims_xl,sml_dims_flux]
        )

        def dimselector(dims, dims_xl, dims_flux, ltypes):
            rl={}
            if "Others" in ltypes:ltypes += ["LyCORIS", "unknown"]
            for name, vals in ldict.items():
                dim, ltype, sdver = vals
                if sdver == "1.X/2.X" and dim in dims and ltype in ltypes:
                    rl[name] = vals
                if sdver == "XL" and dim in dims_xl and ltype in ltypes:
                    rl[name] = vals
                if sdver == "Flux" and dim in dims_flux and ltype in ltypes:
                    rl[name] = vals

            global selectable
            selectable = toselect(limited_lora_dict(rl))

            return gr.update(choices = selectable, value =[])

        def llister(names,ratio, hiden):
          if hiden:return gr.update()
          if names ==[] : return ""
          else:
            for i,n in enumerate(names):
              if "[" in n:names[i] = n[:n.find("[")]
            return f":{ratio},".join(names)+f":{ratio} "

        hidenb.change(fn=lambda x: False, outputs = [hidenb])
        sml_loras.change(fn=llister,inputs=[sml_loras,sml_lratio, hidenb],outputs=[sml_loranames])     
        sml_dims.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_loratypes],outputs=[sml_loras])  
        sml_dims_xl.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_loratypes],outputs=[sml_loras]) 
        sml_dims_flux.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_loratypes],outputs=[sml_loras]) 
        sml_loratypes.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_loratypes],outputs=[sml_loras]) 

##############################################################
####### make LoRA from checkpoint

def makelora(model_a,model_b,dim,saveto,settings,alpha,beta,save_precision,calc_precision,metasets,device):
    print("make LoRA start")
    if model_a == "" or model_b =="":
      return "ERROR: No model Selected"
    gc.collect()

    try:
        currentinfo = shared.sd_model.sd_checkpoint_info
    except:
        currentinfo = None

    lowvramdealer() # Work around web UI low VRAM state issues.

    checkpoint_info = resolve_checkpoint_info(model_a)
    if checkpoint_info is None:
        return f"ERROR: Model not found: {model_a}"
    if resolve_checkpoint_info(model_b) is None:
        return f"ERROR: Model not found: {model_b}"
    load_model(checkpoint_info)

    model = shared.sd_model
    print(type(model).__name__)
    print("XL" in type(model).__name__)

    is_sdxl = type(model).__name__ == "StableDiffusionXL" or getattr(model,'is_sdxl', False)
    is_sd2 = type(model).__name__ == "StableDiffusion2" or getattr(model,'is_sd2', False)
    is_sd1 = type(model).__name__ == "StableDiffusion" or getattr(model,'is_sd1', False)
    is_flux = type(model).__name__ == "Flux" or getattr(model,'is_flux', False)

    print(f"Detected model type: SDXL: {is_sdxl}, SD2.X: {is_sd2}, SD1.X: {is_sd1}")

    if forge:
        unload_forge()
    else:
        sd_models.unload_model_weights()

    if saveto =="" : saveto = makeloraname(model_a,model_b)
    if not ".safetensors" in saveto :saveto  += ".safetensors"
    saveto = os.path.join(shared.cmd_opts.lora_dir,saveto)

    dim = 128 if type(dim) != int else int(dim)
    if os.path.isfile(saveto ) and not "overwrite" in settings:
        _err_msg = f"Output file ({saveto}) existed and was not saved"
        print(_err_msg)
        return _err_msg

    args = Kohya_extract_args(
        v2=is_sd2,
        v_parameterization=True,
        sdxl=is_sdxl,
        save_precision=save_precision,
        model_org=fullpathfromname(model_b),
        model_tuned=fullpathfromname(model_a),
        save_to=saveto,
        dim=dim,
        conv_dim=None,
        device=device,
        no_metadata=False,
        alpha = alpha,
        beta = beta
    )

    result = ext.svd(args)

    if currentinfo:
        load_model(currentinfo)
    return result

##############################################################
####### merge LoRAs

def lmerge(loranames,selected_loras,default_ratio,loraratioss,settings,filename,dim,save_precision,calc_precision,metasets,alpha,beta,smooth,merge,device):
    try:
        import lora
        loranames = (loranames or "").strip()
        if not loranames:
            loranames = selected_loras_to_loranames(selected_loras, default_ratio)
        if not loranames:
            return "ERROR: No LoRA Selected"
        parsed_lora_names = [x.split(":", 1)[0].strip() for x in loranames.split(",") if x.strip()]
        loras_on_disk = [lora.available_loras.get(name, None) for name in parsed_lora_names]
        if any([x is None for x in loras_on_disk]):
            lora.list_available_loras()

            loras_on_disk = [lora.available_loras.get(name, None) for name in parsed_lora_names]

        lnames = loranames.split(",")

        #LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,.
        #LoRAname1:ratio1:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,LoRAname2:ratio2:Blocks2,.

        temp = []
        for n in lnames:
            if ":" in n:
                temp.append(n.split(":"))
            else:
                temp[-1].append(n)

        lnames = temp

        loraratios=loraratioss.splitlines()
        ldict ={}

        for i,l in enumerate(loraratios):
            if ":" not in l or not any(l.count(",") == x - 1 for x in BLOCKNUMS) : continue
            ldict[l.split(":")[0]]=l.split(":")[1]

        ln, lr, ld, lt, lm, ls = [], [], [], [], [], [] # lm: metadata for each LoRA, ls: SD version
        dmax = 1

        for i,n in enumerate(lnames):
            if len(n) ==2:
                ratio = [float(n[1])]*26
            elif len(n) ==3:
                if n[2].strip() in ldict:
                    ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
                    ratio = to26(ratio)
                else:ratio = [float(n[1])]*26
            elif len(n[2:]) in BLOCKNUMS:
                ratio = [float(x) for x in n[2:]]
                ratio = to26(ratio)
            else:
                print("ERROR:Number of Blocks must be 12,17,20,26")
                ratio = [float(n[1])]*26
            c_lora = lora.available_loras.get(n[0], None) 
            ln.append(c_lora.filename)
            lr.append(ratio)
            d, t, s = dimgetter(c_lora.filename, device)
            if t == "LoCon" and isinstance(d, list):
                d = list(set(d))
                d = d[0]
            lt.append(t)
            ld.append(d)
            ls.append(s)
            if d != "LyCORIS" and isinstance(d, int):
                if d > dmax : dmax = d
            
            # Keep metadata for each LoRA.
            meta = prepare_merge_metadata( n[1], ",".join( [str(n) for n in ratio] ), c_lora )
            lm.append( meta )

        if filename =="":filename =loranames.replace(",","+").replace(":","_")
        if not ".safetensors" in filename:filename += ".safetensors"
        loraname = filename.replace(".safetensors", "")
        filename = os.path.join(shared.cmd_opts.lora_dir,filename)

        auto = True if dim == "auto" else False
    
        dim = int(dim) if dim != "no" and dim != "auto" else 0

        if merge:
            if "LyCORIS" in ld:
                if len(ld) !=1:
                    return "multiple merge of LyCORIS is not supported"
                sd = lycomerge(ln[0], lr[0], calc_precision, device)
            elif dim > 0:
                print("change demension to ", dim)
                sd = merge_lora_models_dim(ln, lr, dim,settings,device,calc_precision)
            elif auto and ld.count(ld[0]) != len(ld):
                print("change demension to ",dmax)
                sd = merge_lora_models_dim(ln, lr, dmax,settings,device,calc_precision)
            else:
                sd = merge_lora_models(ln, lr, settings, False, calc_precision, device)

            if os.path.isfile(filename) and not "overwrite" in settings:
                _err_msg = f"Output file ({filename}) existed and was not saved"
                print(_err_msg)
                return _err_msg
        else:
            a = merge_lora_models(ln[0:1], lr[0:1], settings, False, calc_precision, device)
            b = merge_lora_models(ln[1:2], lr[1:2], settings, False, calc_precision, device)
            sd = extract_two(a,b,alpha,beta,smooth)
        
        # Build metadata for the merged LoRA.
        metadata = create_merge_metadata( sd, lm, loraname, save_precision,metasets )

        save_to_file(filename,sd,sd, str_to_dtype(save_precision), metadata)
        sd = None
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        return "saved : "+filename
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        return exc_value

def merge_lora_models(models, ratios, sets, locon, calc_precision, device):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merge_dtype = str_to_dtype(calc_precision)
    merged_sd = {}
    fugou = 1
    for model, ratios in zip(models, ratios):
        keylist = LBLCOKS26

        print(f"merging {model}: {ratios}")
        lora_sd, metadata, isv2 = load_state_dict(model, merge_dtype, device)

        # get alpha and dim
        alphas = {}                             # alpha for current model
        dims = {}                               # dims for current model

        base_dims, base_alphas, dims, alphas = dimalpha(lora_sd, base_dims, base_alphas)

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if 'alpha' in key or "dora" in key:
                continue

            lora_module_name = key[:key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            ratio = ratios[blockfromkey(key, keylist, isv2)]
            #print(key,blockfromkey(key, keylist, isv2))
            
            if "same to Strength" in sets:
                ratio, fugou = (ratio ** 0.5, 1) if ratio > 0 else (abs(ratio) ** 0.5, -1)

            if "lora_down" in key:
                ratio = ratio * fugou

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert merged_sd[key].size() == lora_sd[key].size(), (
                    f"weights shape mismatch merging v1 and v2, different dims? "
                    f"/ weights shape mismatch; v1/v2 or different dimensions cannot be merged"
                    f" {merged_sd[key].size()} ,{lora_sd[key].size()}, {lora_module_name}"
                )
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale
        del lora_sd

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def merge_lora_models_dim(models, ratios, new_rank, sets, device, calc_precision):
    CHUNK_SIZE = 50

    isv2 = False
    merge_dtype = str_to_dtype(calc_precision)
    
    lora_sds = []
    print("Loading LoRA models...")
    for model in models:
        lora_sd, _, _isv2 = load_state_dict(model, merge_dtype, "cpu")
        isv2 = isv2 or _isv2
        lora_sds.append(lora_sd)

    all_lora_module_names = set()
    for lora_sd in lora_sds:
        for key in lora_sd.keys():
            if 'lora_down' in key:
                lora_module_name = key[:key.rfind(".lora_down")]
                all_lora_module_names.add(lora_module_name)
    
    all_lora_module_names = sorted(list(all_lora_module_names))
    total_modules = len(all_lora_module_names)
    total_chunks = (total_modules + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Found {total_modules} unique modules to merge. Processing in {total_chunks} chunks of {CHUNK_SIZE}.")

    merged_lora_sd = {}

    with tqdm(total=total_modules, desc="Overall Progress") as pbar_overall:
        for i in range(0, total_modules, CHUNK_SIZE):
            chunk = all_lora_module_names[i:i + CHUNK_SIZE]
            
            pbar_overall.set_description(f"Processing Chunk {i//CHUNK_SIZE + 1}/{total_chunks}")

            merged_sd_chunk = {}
            original_shapes_chunk = {}

            for lora_module_name in chunk:
                merged_weight = None
                
                for j, lora_sd in enumerate(lora_sds):
                    ratio = ratios[j]
                    down_key = lora_module_name + '.lora_down.weight'
                    if down_key not in lora_sd:
                        continue

                    down_weight = lora_sd[down_key].to(device, non_blocking=True)
                    up_weight = lora_sd[lora_module_name + '.lora_up.weight'].to(device, non_blocking=True)
                    
                    network_dim = down_weight.size(0)
                    alpha = lora_sd.get(lora_module_name + '.alpha', torch.tensor(network_dim)).to(device, non_blocking=True)
                    scale = (alpha / network_dim) if network_dim else 0

                    conv2d = len(down_weight.size()) == 4
                    if not conv2d:
                        diff = (up_weight @ down_weight)
                    else:
                        diff = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3)

                    block_ratio = ratio[blockfromkey(down_key, LBLCOKS26, isv2)]
                    fugou = 1
                    if "same to Strength" in sets:
                        block_ratio, fugou = (block_ratio ** 0.5, 1) if block_ratio > 0 else (abs(block_ratio) ** 0.5, -1)
                    
                    if merged_weight is None:
                        merged_weight = (block_ratio * diff * scale * fugou)
                    else:
                        merged_weight += (block_ratio * diff * scale * fugou)

                if merged_weight is not None:
                    merged_sd_chunk[lora_module_name] = merged_weight
                    if len(merged_weight.shape) == 4:
                        original_shapes_chunk[lora_module_name] = merged_weight.shape

            with torch.no_grad():
                for lora_module_name, mat in merged_sd_chunk.items():
                    mat = mat.to(torch.float)
                
                    conv2d = lora_module_name in original_shapes_chunk
                    if conv2d:
                        out_dim, in_dim, k_h, k_w = original_shapes_chunk[lora_module_name]
                        mat = mat.reshape(out_dim, -1)

                    U, S, Vh = torch.linalg.svd(mat)

                    U = U[:, :new_rank]
                    S = S[:new_rank]
                    U = U @ torch.diag(S)
                    Vh = Vh[:new_rank, :]

                    dist = torch.cat([U.flatten(), Vh.flatten()])
                    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
                    low_val = -hi_val
                    U = U.clamp(low_val, hi_val)
                    Vh = Vh.clamp(low_val, hi_val)

                    new_up_weight = U
                    new_down_weight = Vh

                    if conv2d:
                        out_dim, in_dim, k_h, k_w = original_shapes_chunk[lora_module_name]
                        new_up_weight = new_up_weight.unsqueeze(2).unsqueeze(3)
                        new_down_weight = new_down_weight.view(new_rank, in_dim, k_h, k_w)

                    merged_lora_sd[lora_module_name + '.lora_up.weight'] = new_up_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + '.lora_down.weight'] = new_down_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + '.alpha'] = torch.tensor(float(new_rank))

                    pbar_overall.update(1)

            del merged_sd_chunk, original_shapes_chunk, chunk
            gc.collect()
            torch.cuda.empty_cache()

    print("LoRA merge process completed.")
    return merged_lora_sd

def extract_two(a,b,pa,pb,ps):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merged_sd = {}
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model

    base_dims_a, base_alphas_a, dims, alphas_a = dimalpha(a, base_dims, base_alphas)
    base_dims_b, base_alphas_b, dims, alphas_b = dimalpha(b, base_dims, base_alphas)

    print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

    # merge
    print(f"merging...")
    for key in a.keys():
        if 'alpha' in key:
            continue

        lora_module_name = key[:key.rfind(".lora_")]

        base_alpha_a = base_alphas_a[lora_module_name]
        base_alpha_b = base_alphas_b[lora_module_name]
        alpha_a = alphas_a[lora_module_name]
        alpha_b = alphas_b[lora_module_name]

        scale_a = math.sqrt(alpha_a / base_alpha_a) 
        scale_b = math.sqrt(alpha_b / base_alpha_b)

        merged_sd[key] = extract_super(None,a[key] * scale_a,b[key] * scale_b,pa,pb,ps)

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def lycomerge(filename, ratios, calc_precision, device):
    merge_dtype = str_to_dtype(calc_precision)
    sd, metadata, isv2 = load_state_dict(filename, merge_dtype, device)

    if len(ratios) == 17:
      r0 = 1
      ratios = [ratios[0]] + [r0] + ratios[1:3]+ [r0] + ratios[3:5]+[r0] + ratios[5:7]+[r0,r0,r0] + [ratios[7]] + [r0,r0,r0] + ratios[8:]

    print("LyCORIS: " , ratios)

    keys_failed_to_match = []

    for lkey, weight in sd.items():
        ratio = 1
        picked = False
        if 'alpha' in lkey:
          continue

        fullkey = convert_diffusers_name_to_compvis(lkey,isv2)

        if "." not in fullkey:continue

        key, lora_key = fullkey.split(".", 1)

        for i,block in enumerate(LBLCOKS26):
            if block in key:
                ratio = ratios[i]
                picked = True
        if not picked: keys_failed_to_match.append(key)

        sd[lkey] = weight * math.sqrt(abs(float(ratio)))

        if "down" in lkey and ratio < 0:
          sd[key] = sd[key] * -1
        
    if len(keys_failed_to_match) > 0:
      print(keys_failed_to_match)
  
    return sd 

##############################################################
####### merge to checkpoint
@protect_bake
def pluslora(lnames,selected_loras,default_ratio,loraratios,settings,output,model,save_precision,calc_precision,metasets,device):
    if not model: return "ERROR: No model Selected"
    lnames = (lnames or "").strip()
    if not lnames:
        lnames = selected_loras_to_loranames(selected_loras, default_ratio)
    if lnames == "":return "ERROR: No LoRA Selected"
    import lora
    print("Plus LoRA start")
    add = ""

    entries = parse_lora_merge_entries(lnames)
    presets = {}
    for line in (loraratios or "").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            presets[key.strip()] = value.strip()
    names, filenames, lweis = [], [], []
    for entry in entries:
        name = entry[0]
        item = lora.available_loras.get(name, lora.available_lora_aliases.get(name))
        if item is None:
            return f"ERROR: LoRA not found: {name}"
        names.append(name)
        filenames.append(item.filename)
        lweis.append(resolve_bake_ratios(entry, presets, item.filename))

    modeln=filenamecutter(model,True)   
    dname = modeln
    for n in names:
      dname = dname + "+"+n

    checkpoint_info = resolve_checkpoint_info(model)
    if checkpoint_info is None:
        return f"ERROR: Model not found: {model}"

    if forge and hasattr(sd_models, "read_state_dict"):
        revert_target = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    print(f"Loading {model}")

    theta_0 = read_model_state_dict(checkpoint_info, device)
    dtype = qdtyper(theta_0)

    if dtype == "fp4" or dtype == "nf4":
        print(f"Changing dtype of {model} from {dtype} to float16")
        qkeys = list(theta_0.keys())
        q_dequantize(theta_0,dtype,device,torch.float16,False)

    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    isv2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()
    isflux = any("double_block" in k for k in theta_0.keys())
    need_revert = prefixer(theta_0) if isflux else False

    isanima = any(is_anima_checkpoint_key(k) for k in theta_0)

    if isflux:
        lweis = to61s(lweis)

    try:
        import networks
        is15 = True
    except:
        is15 = False

    keychanger = {}
    for key in theta_0.keys():
        skey = key.replace(".","_").replace("_weight","")
        if "conditioner_embedders_" in skey:
            keychanger[skey.split("conditioner_embedders_",1)[1]] = key
        else:
            if "wrapped" in skey:
                keychanger[skey.split("wrapped_",1)[1]] = key
            elif "clip_l" in skey or "t5xxl" in skey:
                keychanger[skey.replace("text_encoders_","")] = key
            elif "model_" in skey:
                keychanger[skey.split("model_",1)[1]] = key

    if isanima:
        for key in theta_0:
            for alias in anima_key_aliases_from_checkpoint_key(key):
                register_key_alias(keychanger, alias, key)

    lowvramdealer() # Work around web UI low VRAM state issues.

    if is15:
        if shared.sd_model is not None:
            orig_checkpoint = shared.sd_model.sd_checkpoint_info if hasattr(shared.sd_model, "sd_checkpoint_info") else None
        else:
            orig_checkpoint = None
        checkpoint_info = resolve_checkpoint_info(model)
        if checkpoint_info is None:
            return f"ERROR: Model not found: {model}"
        if orig_checkpoint != checkpoint_info:
            load_model(checkpoint_info)
        
        theta_0 = newpluslora(theta_0,filenames,lweis,names, calc_precision, isxl,isv2,isflux, keychanger)
        
        if dtype == "fp4" or dtype == "nf4":
            print(f"Changing dtype of {model} from float16 to {dtype}")
            q_quantize(theta_0,dtype,device,False)
        
            failedkeys = []
            for key in theta_0:
                if key not in qkeys:
                    failedkeys.append(key)

            print(f"Key Check : {'OK' if failedkeys == [] else str(len(failedkeys)) + ' keys failed'}")

        if need_revert:
            prefixer(theta_0, True)

        if orig_checkpoint:
            load_model(orig_checkpoint)
    else:
        theta_0 = oldpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2, keychanger, device)

    #usemodelgen(theta_0,model)
    validate_merged_checkpoint_state_dict(theta_0)
    settings = list(settings or [])
    settings.append(save_precision)
    settings.append("safetensors")
    result = savemodel(theta_0,dname,output,settings)

    # Neo cleanup (including exceptions) is owned by protect_bake. Keep the
    # original cache cleanup on non-Neo hosts.
    if hasattr(sd_models, "read_state_dict"):
        if hasattr(lora, "loaded_loras"):
            lora.loaded_loras.clear()
        if hasattr(sd_models, "checkpoints_loaded"):
            sd_models.checkpoints_loaded.clear()
        if forge:
            from modules.sd_models import FakeInitialModel
            sd_models.unload_model_weights()
            sd_models.checkpoint_info = FakeInitialModel()
            load_model(revert_target, reload=True)

    del theta_0
    gc.collect()
    return result + add

def oldpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2, keychanger, device):
    for name,filename, lwei in zip(names,filenames, lweis):
        print(f"loading: {name}")
        lora_sd, metadata, isv2 = load_state_dict(filename, torch.float, device)

        print(f"merging..." ,lwei)
        for key in lora_sd.keys():
            ratio = 1
            fullkey = convert_diffusers_name_to_compvis(key,isv2)

            msd_key, _ = fullkey.split(".", 1)
            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            for i,block in enumerate(LBLCOKS26):
                if block in fullkey or block in msd_key:
                    ratio = lwei[i]

            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[:key.index("lora_down")] + 'alpha'

                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key].to(device="cpu")
                up_weight = lora_sd[up_key].to(device="cpu")

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim
                # W <- W + U * D
                weight = theta_0[keychanger[msd_key]].to(device="cpu")

                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale

                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale
                    
                theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    return theta_0

def newpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2,isflux, keychanger):
    # SuperMerger stores per-module LBW ratios on the loaded LoRA objects.
    # Reusing cached objects can leak ratios across merge runs, so force reload.
    if hasattr(nets, "loaded_networks"):
        nets.loaded_networks.clear()
    if hasattr(nets, "networks_in_memory"):
        nets.networks_in_memory.clear()
    nets.load_networks(names, [1]* len(names),[1]* len(names), None, isxl, isv2)

    if len(nets.loaded_networks) != len(names):
        raise RuntimeError("Some LoRAs failed to load; checkpoint was not saved")
    if len(set(names)) != len(names):
        raise ValueError("Use each LoRA once per bake; combine repeated strengths first")
    for loaded, filename, weights, name in zip(nets.loaded_networks, filenames, lweis, names):
        if loaded.name != name or not loaded.modules:
            raise RuntimeError(f"No matching modules were loaded for LoRA: {name}")
        lbw(loaded, to26(weights), isv2, isflux)
        lora_sd, _, _ = load_state_dict(filename, torch.float32, "cpu")
        sync_module_scales_from_state_dict(loaded, lora_sd)
        del lora_sd

    for net in nets.loaded_networks:
        errormodules = []
        net.dyn_dim = None
        applied_groups = set()
        applied_count = 0
        for name,module in tqdm(net.modules.items(), desc=f"{net.name}"):
            fullkey = convert_diffusers_name_to_compvis(name,isv2)
            msd_key = module_key_from_fullkey(fullkey)

            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            qvk = ["_q_proj","_k_proj","_v_proj","_out_proj",".q_proj",".k_proj",".v_proj",".out_proj",".output_proj"]

            wkey, matched_key = resolve_checkpoint_weight_key(msd_key, keychanger, theta_0, module.shape)
            if wkey is not None:
                applied_count += 1
                bkey = wkey.replace("weight","bias")
                if bkey in theta_0.keys():
                    theta_0[wkey], theta_0[bkey]= plusweights(theta_0[wkey], module, bias = theta_0[bkey])
                else:
                    theta_0[wkey], _ = plusweights(theta_0[wkey] ,module)

            else:
                inkey, outkey = resolve_qkv_group_keys(name, keychanger)
                group = re.sub(r"(?:_|\.)(?:q|k|v|out|output)_proj$", "", name)
                if inkey is not None and inkey in keychanger and outkey in keychanger:
                    if group in applied_groups:
                        continue
                    applied_groups.add(group)
                    iw, ow = keychanger[inkey], keychanger[outkey]
                    bkey = ow.replace("weight", "bias")
                    theta_0[iw], theta_0[ow], result_bias = plusweightsqvk(
                        theta_0[iw], theta_0[ow], group, module, net, theta_0.get(bkey))
                    if bkey in theta_0:
                        theta_0[bkey] = result_bias
                    applied_count += 1
                else:
                    errormodules.append(msd_key)

        if not applied_count:
            raise RuntimeError(f"No checkpoint weights matched LoRA: {net.name}")
        if errormodules != []:
            print(f"Unmerged modules in {net.name} : {errormodules}")
        gc.collect()
    return theta_0

def plusweights(weight, module, bias=None):
    with torch.no_grad():
        label = getattr(module, "network_key", "?")
        # Zero must be a no-op even if an unused adapter contains invalid data.
        if float(getattr(module, "lbw_ratio", 1.0)) == 0:
            return weight, bias
        delta, extra_bias = bake_delta(module, weight)
        result = checked_bake_add(weight, delta, label)
        result_bias = bias
        if bias is not None and extra_bias is not None:
            result_bias = checked_bake_add(bias, extra_bias, label + ".bias")
        return result, result_bias

def plusweightsqvk(inweight, outweight, network_layer_name, module, net, bias=None):
    with torch.no_grad():
        separator = "." if "." in network_layer_name else "_"
        # One fused QKV matrix contains three independent weight slices.
        weights = list(inweight.chunk(3, dim=0))
        for i, projection in enumerate(("q", "k", "v")):
            candidate = net.modules.get(network_layer_name + separator + projection + "_proj")
            if candidate is not None:
                weights[i], _ = plusweights(weights[i], candidate)
        output = net.modules.get(network_layer_name + separator + "out_proj")
        if output is None:
            output = net.modules.get(network_layer_name + separator + "output_proj")
        result_out, result_bias = outweight, bias
        if output is not None:
            result_out, result_bias = plusweights(outweight, output, bias)
        return torch.cat(weights, dim=0), result_out, result_bias

def lbw(lora,lwei,isv2,isflux=False):
    errormodules = []

    blocks = list(LBLCOKS26)
    if isv2:
        blocks[0] = V2ENCODER

    for key in lora.modules.keys():
        ratio = lwei[0]
        picked = False

        if isflux:
            block = get_flux_blocks(key)
            if block in BLOCKIDFLUX:
                ratio = lwei[BLOCKIDFLUX.index(block)]
                picked = True

        else:
            anima_block = re.search(r"(?:^|[._])blocks[._](\d+)(?:[._]|$)", key)
            if anima_block:
                block_index = int(anima_block.group(1))
                ratio = lwei[min(block_index + 1, len(lwei) - 1)]
                picked = True
            else:
                anima_layer = (
                    re.search(r"^layers_(\d+)_", key)
                    or re.search(r"_model_layers_(\d+)_", key)
                    or re.search(r"\.model\.layers\.(\d+)\.", key)
                )
                if anima_layer:
                    layer_index = int(anima_layer.group(1))
                    ratio = lwei[min(layer_index + 1, len(lwei) - 1)]
                    picked = True
                elif key.startswith("layers_") or "model_layers_" in key or ".model.layers." in key:
                    ratio = lwei[0]
                    picked = True
                elif any(marker in key for marker in (
                    "diffusion_model.llm_adapter.norm",
                    "diffusion_model_t_embedding_norm",
                    "diffusion_model.t_embedding.norm",
                    "llm_adapter.norm",
                    "t_embedding_norm",
                )):
                    ratio = lwei[0]
                    picked = True
            for i,block in enumerate(blocks):
                if block in key:
                    if i == 26 or i == 27: i=0
                    ratio = lwei[i]
                    picked = True

        if not picked:
            errormodules.append(key)

        lora.modules[key].lbw_ratio = ratio

    if errormodules:
        print("BASE strength used for unmatched block names:", errormodules)
    else:
        print(f"{lora.name}: Successfully set the ratio {lwei} ")

    return lora

LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
    "NetworkModuleNorm": "w_norm",
}

def save_to_file(file_name, model, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)

CLAMP_QUANTILE = 0.99

def str_to_dtype(p):
  if p == 'float':
    return torch.float
  if p == 'fp16':
    return torch.float16
  if p == 'bf16':
    return torch.bfloat16
  return None


def get_safetensors_header(filename):
    import json
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        if metadata_len > 2 and json_start in (b'{"', b"{'"):
            json_data = json_start + file.read(metadata_len-2)
            return json.loads(json_data)

        # invalid safetensors
        return {}

def load_state_header(file_name, dtype):
  """load safetensors header if available"""
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = get_safetensors_header(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd

def load_state_dict(file_name, dtype, device = "cpu"):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name,device=device)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location=device)
        metadata = {}

    isv2 = False

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype = dtype, device = device)
            if "resblocks" in key:
                isv2 = True
    
    if isv2: print("SD2.X")

    return sd, metadata, isv2

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

def dimgetter(filename, device = "cpu"):
    lora_sd = load_state_header(filename, torch.float)
    alpha = None
    dim = None
    ltype = None

    if "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight" in lora_sd.keys():
      ltype = "LoCon"
      if type(lora_sd["lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight"]) is dict:
          lora_sd, _, _ = load_state_dict(filename, torch.float, device)
      _, _, dim, _ = dimalpha(lora_sd)

    if "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight" in lora_sd.keys():
        sdx = "XL"
        if type(lora_sd["lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight"]) is dict:
            lora_sd, _, _ = load_state_dict(filename, torch.float, device)
        _, _, dim, _ = dimalpha(lora_sd)
    elif "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.hada_w1_a" in lora_sd.keys():
        sdx = "XL"
    elif any("single_blocks" in key for key in lora_sd.keys()):
        sdx = "Flux"
    else:
        sdx = "1.X/2.X"

    if isinstance(dim, dict):
        dim = d2l(dim)

    for key, value in lora_sd.items():
  
        if alpha is None and 'alpha' in key:
            alpha = value
        if dim is None and 'lora_down' in key:
            if type(value) == torch.Tensor and len(value.size()) == 2:
                dim = value.size()[0]
            elif type(value) == dict:
                dim = value.get("shape",[0,0])[0]
        if "hada_" in key:
            dim,ltype = "LyCORIS","LyCORIS"
        if alpha is not None and dim is not None:
            break
    if alpha is None:
        alpha = dim
    if ltype == None:ltype = "LoRA"
    if dim :
      return dim, ltype, sdx
    else:
      return "unknown","unknown","unknown"
    
def d2l(dimdict):
    out = []
    for v in dimdict.values():
        if v not in out:
            out.append(v)
    return out[0] if len(out) == 1 else out

def blockfromkey(key,keylist,isv2 = False):
    fullkey = convert_diffusers_name_to_compvis(key,isv2)

    if "lora_unet" in fullkey:
        fullkey = fullkey.replace("lora_unet", "diffusion_model")
    elif "lora_te1_text_model" in fullkey:
        fullkey = fullkey.replace("lora_te1_text_model", "0_transformer_text_model")

    if "1_model_transformer_resblocks_" in fullkey:return 0

    for i,n in enumerate(keylist):
        if n in fullkey: return i

    print(f"ERROR:Block is not deteced:{fullkey}")
    return 0

def dimalpha(lora_sd, base_dims={}, base_alphas={}):
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model
    for key in lora_sd.keys():
        if 'alpha' in key:
            lora_module_name = key[:key.rfind(".alpha")]
            alpha = float(lora_sd[key].detach().cpu().numpy()) 
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
        elif "lora_down" in key:
            lora_module_name = key[:key.rfind(".lora_down")]
            dim = lora_sd[key].size()[0]
            dims[lora_module_name] = dim
            if lora_module_name not in base_dims:
                base_dims[lora_module_name] = dim

    for lora_module_name in dims.keys():
        if lora_module_name not in alphas:
            alpha = dims[lora_module_name]
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
    return base_dims, base_alphas, dims, alphas


def fullpathfromname(name):
    checkpoint_info = resolve_checkpoint_info(name)
    if checkpoint_info is None:
        raise ValueError(f"Model not found: {name}")
    return checkpoint_info.filename

def makeloraname(model_a,model_b):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    return "lora_"+model_a+"-"+model_b

V2ENCODER = "resblocks"

LBLCOKS26=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders",
"transformer_resblocks"]

###########################################################
##### metadata

def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash

def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]

def prepare_merge_metadata( ratio, blocks, fromLora ):
    """
    郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ邵ｺ・ｫ ratio, blocks 邵ｺ・ｪ邵ｺ・ｩ邵ｺ・ｮ隲繝ｻ・ｰ・ｱ郢ｧ蜑・ｽｻ莨懷・邵ｺ蜉ｱ窶ｻ邵ｺ鄙ｫ・･

    Parameters
    ----
    ratio : string
        name:ratio:blocks 邵ｺ・ｮ ratio 鬩幢ｽｨ陋ｻ繝ｻ
    blocks : string
        name:ratio:bloks 邵ｺ・ｮ blocks 鬩幢ｽｨ陋ｻ繝ｻ郢晢ｽｩ郢晏生ﾎ晉ｸｺ・ｧ邵ｺ・ｯ邵ｺ・ｪ邵ｺ荳岩ｻ陞ｳ貅倥Τ郢晢ｽｩ郢晢ｽ｡郢晢ｽｼ郢ｧ・ｿ)
    fromLora : NetworkOnDisk
        郢晄ｧｭ繝ｻ郢ｧ・ｸ陝・ｽｾ髮趣ｽ｡邵ｺ・ｮLoRA
    
    Returns
    ----
    dict[str, str]
        郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ
    """
    meta = fromLora.metadata
    meta["sshs_ratio"] = str.strip( ratio )
    meta["sshs_blocks"] = str.strip( blocks )
    meta["ss_output_name"] = str.strip( fromLora.name )

    return meta

BASE_METADATA = [
    "sshs_ratio", "sshs_blocks", "ss_output_name",
    "sshs_model_hash", "sshs_legacy_hash",
    "ss_network_module",
    "ss_network_alpha", "ss_network_dim",
    "ss_mixed_precision", "ss_v2",
    "ss_training_comment",
    "ss_sd_model_name", "ss_new_sd_model_hash",
    "ss_clip_skip",
    "ss_base_model_version"
]

MINIMUM_METADATA = [
    "ss_network_module","ss_network_alpha", "ss_network_dim","ss_v2","ss_sd_model_name", "ss_base_model_version"
]

def create_merge_metadata( sd, lmetas, lname, lprecision, metasets ):
    """
    LoRA郢晄ｧｭ繝ｻ郢ｧ・ｸ陟募ｾ後・郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ郢ｧ蜑・ｽｽ諛医・邵ｺ蜷ｶ・・

    Parameters
    ----
    sd : NetworkOnDisk
        郢晄ｧｭ繝ｻ郢ｧ・ｸ陟募ｾ後・LoRA
    lmetas : dict[str, str]
        郢晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ霈費ｽ檎ｹｧ險椀RA邵ｺ・ｮ郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ
    lname : str
        郢晄ｧｭ繝ｻ郢ｧ・ｸ陟募ｾ後・LoRA陷ｷ繝ｻ
    lprecision : str
        save precision 邵ｺ・ｮ陋滂ｽ､
    mergeAll : bool
        郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ邵ｺ・ｮ隹ｿ荵晢ｼ隴・ｽｹ邵ｲ繧・螺邵ｺ・ｰ邵ｺ蜉ｱ縺｡郢ｧ・ｰ隲繝ｻ・ｰ・ｱ邵ｺ・ｯ郢昴・縺・ｹ晢ｽｬ郢ｧ・ｯ郢晏現ﾎ懆惺髦ｪ窶ｲ陟墓ぁ莠ｫ邵ｺ・｡邵ｺ・ｧ郢晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ蜉ｱ竏ｪ邵ｺ繝ｻ
        True 陷茨ｽｨ郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ郢ｧ蜻茨ｽｮ荵昶・邵ｲ繧・・郢晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ・ｮ陜｣・ｴ陷ｷ蛹ｻ繝ｻTrue陜暦ｽｺ陞ｳ繝ｻ
        False 闕ｳﾂ鬩幢ｽｨ邵ｺ・ｮ郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ邵ｺ・ｮ邵ｺ・ｿ隹ｿ荵昶・
    
    Returns
    ----
    dict[str, str]
        郢晢ｽ｡郢ｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ
    """

    metadata = {}
    networkModule = None

    if "first" in metasets:
        # 陷雁･竊醍ｹｧ莉㌃ight郢晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ・ｪ郢ｧ蟲ｨ笳守ｸｺ・ｮ邵ｺ・ｾ邵ｺ・ｾ郢ｧ・ｳ郢晄鱒繝ｻ
        metadata = lmetas[0]
    elif "new" in metasets:
        for key in MINIMUM_METADATA:
            if key in lmetas[0].keys():
                metadata[key] = lmetas[0][key]
        
    else:
        # 髫阪・辟夂ｹ晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ・ｮ陜｣・ｴ陷ｷ蛹ｻ繝ｻ郢晄ｧｭ繝ｻ郢ｧ・ｸ邵ｺ蜉ｱ笳・ｹｧ・ｿ郢ｧ・ｰ邵ｺ・ｨ闕ｳ・ｻ髫補・ﾎ鍋ｹｧ・ｿ郢昴・繝ｻ郢ｧ・ｿ郢ｧ蜑・ｽｿ譎擾ｽｭ繝ｻ
        metadata = lmetas[0]
        tags = {}
        for i, lmeta in enumerate( lmetas ):
            meta = {}
            metadata[ f"sshs_cp{i}" ] = json.dumps( lmeta )

            # 隴崢陋ｻ譏ｴ繝ｻ network_module 郢ｧ蜑・ｽｿ譎・亜
            if networkModule is None and "ss_network_module" in lmeta:
                networkModule = lmeta["ss_network_module"]

            # 郢ｧ・ｿ郢ｧ・ｰ郢ｧ蛛ｵ繝ｻ郢晢ｽｼ郢ｧ・ｸ
            if "merge" in metasets:
                if "ss_tag_frequency" in lmeta:
                    ldict = lmeta["ss_tag_frequency"]
                    if "ss_tag_frequency" in metadata:
                        mdict = metadata["ss_tag_frequency"]
                        if type(ldict) is dict and type(mdict) is dict:
                            for key in ldict:
                                if key not in mdict:
                                    mdict[key] = ldict[key]

    # network_module邵ｺ荵晢ｽ鵜oRA驕橸ｽｮ陋ｻ・･陋ｻ・､陞ｳ螢ｹ笘・ｹｧ蜿･・ｰ・ｴ陷ｷ蛹ｻ窶ｲ陞溷｣ｹ・樒ｸｺ貅假ｽ∫ｸｲ竏ｵ諤呵崕譏ｴ竊馴囎荵昶命邵ｺ莉｣笳・ｹｧ繧・・邵ｺ・ｫ邵ｺ蜷ｶ・・
    if networkModule is not None:
        metadata["ss_network_module"] = networkModule

    # output陷ｷ髦ｪ竊恥recision邵ｲ窶･im邵ｺ・ｯ陞溽判蟲ｩ邵ｺ霈費ｽ檎ｸｺ貅ｷ蠎・妙・ｽ隲､・ｧ邵ｺ蠕娯旺郢ｧ繝ｻ
    if "without" not in metasets:
        metadata["ss_output_name"] = lname
    else:
        if "ss_output_name" in metadata:
            del metadata["ss_output_name"]
    metadata["ss_mixed_precision"] = lprecision

    # metadata邵ｺ・ｧ闖ｫ譎擾ｽｭ蛟･縲堤ｸｺ髦ｪ・玖厄ｽ｢陟台ｸ岩・陞溽判驪､
    for key in metadata:
        if type(metadata[key] ) is not str:
            metadata[key] = json.dumps( metadata[key] )
    # 郢昴・繝ｻ郢ｧ・ｿ陞溽判蟲ｩ邵ｺ・ｫ郢ｧ蛹ｻ・撹ash邵ｺ謔滂ｽ､蟲ｨ・冗ｹｧ荵昴・邵ｺ・ｧ髫ｪ閧ｲ・ｮ繝ｻ
    model_hash, legacy_hash = precalculate_safetensors_hashes( sd, metadata )
    metadata[ "sshs_model_hash" ] = model_hash
    metadata[ "sshs_legacy_hash" ] = legacy_hash

    return metadata


##############################################################
####### Get loranames from prompt
def frompromptf(*args):
    outst = []
    outss = []
    prompt = args[1]
    names, multis, lbws = loradealer(prompt, "", "")
    for name, multi, lbw in zip(names, multis, lbws):
        nml = [name,str(multi),lbw] if lbw is not None else [name,str(multi)]
        outst.append(":".join(nml))
        if name in selectable:
            outss.append(name)
    global pchanged
    pchanged = True
    return outss,",".join(outst), True

def loradealer(prompts,lratios,elementals):
    _, extra_network_data = extra_networks.parse_prompts([prompts])
    moduletypes = extra_network_data.keys()

    outnames = []
    outmultis = []
    outlbws = []

    for ltype in moduletypes:
        lorans = []
        lorars = []
        loraps = []
        multipliers = []
        elements = []
        if not (ltype == "lora" or ltype == "lyco") : continue
        for called in extra_network_data[ltype]:
            multiple = float(syntaxdealer(called.items,"unet=","te=",1))
            multipliers.append(multiple)
            lorans.append(called.items[0])
            loraps.append(syntaxdealer(called.items,"lbw=",None,2))

        if len(lorans) > 0:
            outnames.extend(lorans)
            outmultis.extend(multipliers)
            outlbws.extend(loraps)

    return outnames, outmultis, outlbws

def syntaxdealer(items,type1,type2,index): #type "unet=", "x=", "lwbe=" 
    target = [type1,type2] if type2 is not None else [type1]
    for t in target:
        for item in items:
            if t in item:
                return item.replace(t,"")
    if index > len(items) - 1 :return None
    return items[index] if "@" not in items[index] else 1

##############################################################
####### Extract lora from checkpoints args
class Kohya_extract_args:
    def __init__(
        self,
        v2=False,
        v_parameterization=None,
        sdxl=False,
        save_precision=None,
        model_org=None,
        model_tuned=None,
        save_to=None,
        dim=4,
        conv_dim=None,
        device=None,
        no_metadata=False,
        alpha = 1,
        beta = 1
    ):
        self.v2 = v2
        self.v_parameterization = v_parameterization
        self.sdxl = sdxl
        self.save_precision = save_precision
        self.model_org = model_org
        self.model_tuned = model_tuned
        self.save_to = save_to
        self.dim = dim
        self.conv_dim = conv_dim
        self.device = device
        self.no_metadata = no_metadata
        self.alpha = alpha
        self.beta = beta

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    #for flux
    if match(m, r"lora_unet_double_blocks_(\d+)_(img|txt)_(attn|mlp|mod)_(proj|qkv|lin|\d+)(.*)"):
        block_type = m[1]  # img or txt
        module_type = m[2]  # attn, mlp, mod
        specific_module = m[3]  # proj, qkv, lin, or numeric index

        # Create suffix for specific module types
        if module_type == "attn":
            if specific_module == "proj":
                suffix = "proj.weight"
            elif specific_module == "qkv":
                suffix = "qkv.weight"
            else:
                suffix = f"norm.{specific_module}_norm.scale"
        elif module_type == "mlp":
            suffix = f"{specific_module}.weight"
        elif module_type == "mod":
            suffix = f"lin.weight"
        else:
            suffix = specific_module

        return f"model.diffusion_model.double_blocks.{m[0]}.{block_type}_{module_type}.{suffix}"

    if match(m, r"lora_unet_single_blocks_(\d+)_(linear\d+|modulation_lin)(.*)"):
        block_index = m[0]  # single block index
        module_name = m[1]  # linear1, linear2, or modulation_lin

        # Create suffix for module types
        if "linear" in module_name:
            suffix = f"{module_name}.weight"
        elif module_name == "modulation_lin":
            suffix = "modulation.lin.weight"
        else:
            suffix = module_name

        return f"model.diffusion_model.single_blocks.{block_index}.{suffix}"

    return key

def lowvramdealer():
    try:
        from modules import lowvram
        lowvram.module_in_gpu = None # Work around web UI low VRAM state issues.
    except:
        pass

def resolve_checkpoint_info(name):
    name = str(name or "").strip()
    if not name:
        return None

    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    if checkpoint_info is not None:
        return checkpoint_info

    normalized = name.replace("/", "\\")
    normalized_no_hash = normalized.split(" [")[0]
    for info in sd_models.checkpoints_list.values():
        candidates = {
            getattr(info, "name", ""),
            getattr(info, "title", ""),
            getattr(info, "short_title", ""),
            getattr(info, "model_name", ""),
            getattr(info, "name_for_extra", ""),
        }
        candidates.update(str(x) for x in getattr(info, "ids", []) if x)
        for candidate in candidates:
            c = str(candidate).replace("/", "\\")
            c_no_hash = c.split(" [")[0]
            if normalized in (c, c_no_hash) or normalized_no_hash in (c, c_no_hash):
                return info

    return None

def get_flux_blocks(key):
    if "vae" in key:
        return "VAE"
    if "t5xxl" in key:
        return "T5"
    if "clip_l" in key:
        return "CLIP"
    
    match = re.search(r'\_(\d+)\_', key)
    if "double_blocks" in key:
        return f"D{match.group(1).zfill(2) }"
    if "single_blocks" in key:
        return f"S{match.group(1).zfill(2) }"
    if "_in" in key:
        return "IN"
    if "final_layer" in key:
        return "OUT"
    return "Not Merge"

def read_model_state_dict(checkpoint_info, device):
    if forge or not hasattr(sd_models, "read_state_dict"):
        from backend.utils import load_torch_file
        load_model(checkpoint_info)
        return load_torch_file(checkpoint_info.filename,device=CUDA if "cuda" in device else CPU)
    else:
        return sd_models.read_state_dict(checkpoint_info.filename,map_location=device)
    
def load_model(checkpoint_info, reload = False):
    if checkpoint_info is None:
        raise ValueError("Model not found: checkpoint_info is None")
    if forge:
        from modules.sd_models import forge_model_reload, model_data
        from modules_forge.main_entry import forge_unet_storage_dtype_options
        unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))
        forge_model_params = dict(
            checkpoint_info=checkpoint_info,
            additional_modules=shared.opts.forge_additional_modules,
            unet_storage_dtype=unet_storage_dtype
        )
        if reload:
            model_data.forge_hash = None
        model_data.forge_loading_parameters = forge_model_params
        forge_model_reload()
    else:
        sd_models.load_model(checkpoint_info)





def strip_network_weight_suffix(key):
    for suffix in (
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_mid.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".alpha",
        ".scale",
        ".bias",
        ".dora_scale",
        ".w_norm",
        ".b_norm",
        ".oft_blocks",
        ".oft_diag",
        ".rescale",
    ):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return key


def sync_module_scales_from_state_dict(net, lora_sd):
    synced = 0
    for module in net.modules.values():
        base_key = strip_network_weight_suffix(module.network_key)
        down_key = None
        for candidate in (
            base_key + ".lora_down.weight",
            base_key + ".lora_A.weight",
        ):
            if candidate in lora_sd:
                down_key = candidate
                break

        alpha_key = next((candidate for candidate in (base_key + ".alpha", base_key + "_alpha") if candidate in lora_sd), None)
        scale_key = next((candidate for candidate in (base_key + ".scale", base_key + "_scale") if candidate in lora_sd), None)

        if down_key is not None:
            module.dim = lora_sd[down_key].shape[0]
        if alpha_key is not None:
            module.alpha = float(lora_sd[alpha_key].detach().cpu().item())
        if scale_key is not None:
            module.scale = float(lora_sd[scale_key].detach().cpu().item())
        synced += 1

    print(f"{net.name}: synchronized alpha/dim metadata for {synced} modules")


def resolve_qkv_group_keys(name, keychanger):
    qkv_suffixes = ["_q_proj", "_k_proj", "_v_proj", "_out_proj", "_output_proj", ".q_proj", ".k_proj", ".v_proj", ".out_proj", ".output_proj"]
    for suffix in qkv_suffixes:
        if suffix in name:
            base = name.rsplit(suffix, 1)[0]
            if suffix.startswith("."):
                inkey = base + ".in_proj"
                out_candidates = [base + ".out_proj", base + ".output_proj"]
            else:
                inkey = base + "_in_proj"
                out_candidates = [base + "_out_proj", base + "_output_proj"]
            outkey = next((candidate for candidate in out_candidates if candidate in keychanger), out_candidates[0])
            return inkey, outkey
    return None, None


def validate_tensor_is_finite(tensor, label):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected while merging LoRA: {label}")


def validate_merged_checkpoint_state_dict(state_dict):
    checked = 0
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            continue
        validate_tensor_is_finite(value, key)
        checked += 1
    print(f"SuperMerger: validated {checked} floating tensors after LoRA merge")


def resolve_bake_ratios(entry, presets, filename):
    name, strength = entry[:2]
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError(f"LoRA strength must be finite: {name}")
    if len(entry) == 2:
        return default_ratio_for_length(default_ratio_length_for_lora(filename), strength)
    spec = ",".join(entry[2:]).removeprefix("lbw=").strip()
    spec = presets.get(spec, spec)
    try:
        blocks = [float(x) for x in spec.split(",")]
    except ValueError:
        raise ValueError(f"Unknown LoRA block preset: {spec}") from None
    if len(blocks) not in BLOCKNUMS or not all(math.isfinite(x) for x in blocks):
        raise ValueError("LoRA block weights must contain 12, 17, 20, 26, 29 or 61 finite values")
    return to26([strength * x for x in blocks])


def checked_bake_add(weight, delta, label):
    """Compute before copying: catch fp16 output overflow as well as bad input."""
    validate_tensor_is_finite(delta, label)
    if tuple(weight.shape) != tuple(delta.shape):
        raise RuntimeError(f"LoRA shape mismatch for {label}: {tuple(weight.shape)} vs {tuple(delta.shape)}")
    result = (weight.float() + delta.float().to(weight.device)).to(weight.dtype)
    validate_tensor_is_finite(result, label)
    return result


def bake_delta(module, weight):
    ratio = float(getattr(module, "lbw_ratio", 1.0))
    if not math.isfinite(ratio):
        raise ValueError("LoRA strength must be finite")
    result = module.calc_updown(weight.to(dtype=torch.float32))
    delta, bias = result if isinstance(result, tuple) else (result, None)
    if weight.ndim == 4 and weight.shape[1] == 9 and delta.shape[1] == 4:
        delta = torch.nn.functional.pad(delta, (0, 0, 0, 0, 0, 5))
    return delta * ratio, None if bias is None else bias * ratio
