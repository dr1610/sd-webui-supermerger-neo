"""Offline regression tests: real adapter classes and safetensors, no GPU model reload."""
import ast
import __future__
from contextlib import redirect_stdout
import gc
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import safetensors
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
TARGET = Path(os.environ.get('BAKE_TEST_SOURCE') or ROOT.parent)

def module(name, **values):
    value = types.ModuleType(name)
    value.__dict__.update(values)
    sys.modules[name] = value
    return value

# Only Forge's host services are stubbed; module parsing, adapter math,
# strength/block selection, and the baking loop are production code.
shared = module('modules.shared', opts=types.SimpleNamespace(), sd_model=None)
devices = module('modules.devices', cpu=torch.device('cpu'), dtype=torch.float16, torch_gc=lambda: None)
sd_models = module('modules.sd_models', forge_model_reload=lambda: None)
errors = module('modules.errors', display=lambda *a, **kw: None)
host_scripts = module('modules.scripts', basedir=lambda: str(TARGET))
launch_utils = module('modules.launch_utils', git_tag=lambda: 'neo')
module('modules', __path__=[], shared=shared, devices=devices, sd_models=sd_models,
       errors=errors, scripts=host_scripts, launch_utils=launch_utils,
       cache=types.SimpleNamespace(), hashes=types.SimpleNamespace())
module('modules.ui', versions_html=lambda: 'Forge Neo')
module('scripts', __path__=[str(TARGET/'scripts')])
module('scripts.A1111', __path__=[str(TARGET/'scripts/A1111')])

network = importlib.import_module('scripts.A1111.network')
names = ('lora', 'hada', 'ia3', 'lokr', 'full', 'norm', 'glora', 'oft')
adapters = {name: importlib.import_module('scripts.A1111.network_' + name) for name in names}

def definitions(path, namespace, decorators=True):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.FunctionDef) and not decorators:
                node.decorator_list = []
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)

nets = dict(torch=torch, re=re, os=os, network=network, shared=shared,
            sd_models=sd_models, launch_utils=launch_utils, devices=devices,
            errors=errors, logging=logging, Union=__import__('typing').Union,
            gr=types.SimpleNamespace(Warning=lambda *a: None),
            sd_hijack=types.SimpleNamespace(model_hijack=types.SimpleNamespace(comments=[])),
            **{'network_' + k: v for k, v in adapters.items()})
definitions(TARGET/'scripts/A1111/networks.py', nets)
net_api = types.SimpleNamespace(**nets)
net_api.loaded_networks = nets['loaded_networks']
net_api.networks_in_memory = nets['networks_in_memory']

def quiet_tqdm(iterator=None, **kw):
    return iterator

bake = dict(torch=torch, re=re, math=math, os=os, gc=gc, nets=net_api,
            load_file=load_file, save_file=save_file, json=json,
            safetensors=safetensors,
            tqdm=quiet_tqdm, shared=shared, sd_models=sd_models, launch_utils=launch_utils,
            BLOCKIDFLUX=['CLIP', 'T5', 'IN']+[f'D{i:02}' for i in range(19)]+[f'S{i:02}' for i in range(38)]+['OUT'],
            CUDA=torch.device('cuda'), CPU=torch.device('cpu'))
definitions(TARGET/'scripts/mergers/pluslora.py', bake, decorators=False)

class BakingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.base = torch.arange(16).float().reshape(4,4)/16
        self.down = torch.arange(1,9).float().reshape(2,4)/8
        self.up = torch.arange(1,9).float().reshape(4,2)/16
        net_api.loaded_networks.clear()
        net_api.networks_in_memory.clear()
        nets['available_networks'].clear()
        nets['available_network_aliases'].clear()
        shared.opts = types.SimpleNamespace()
        self.output = io.StringIO()
        self.silence = redirect_stdout(self.output)
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def create(self, key='lora_unet_input_blocks_0_0', kind='lora', name='test', **extras):
        if kind == 'lora':
            weights = {'lora_down.weight': self.down, 'lora_up.weight': self.up, 'alpha': torch.tensor(1.)}
        elif kind == 'lokr':
            weights = {'lokr_w1_a': torch.tensor([[.1],[.3]]), 'lokr_w1_b': torch.tensor([[.2,.4]]),
                       'lokr_w2_a': torch.tensor([[.3],[.5]]), 'lokr_w2_b': torch.tensor([[.6,.8]]), 'alpha': torch.tensor(1.)}
        elif kind == 'hada':
            weights = {'hada_w1_a': self.up, 'hada_w1_b': self.down,
                       'hada_w2_a': self.up*2, 'hada_w2_b': self.down/2, 'alpha': torch.tensor(1.)}
        else:
            raise ValueError(kind)
        weights.update(extras)
        filename = self.path / (name + '.safetensors')
        save_file({key+'.'+suffix: value.contiguous() for suffix,value in weights.items()}, str(filename))
        return filename, weights

    def load_adapter(self, filename, source_key, state_key='model.diffusion_model.input_blocks.0.0.weight', name='test'):
        mapped = state_key.removesuffix('.weight').replace('.', '_').removeprefix('model_')
        target = torch.nn.Linear(4,4, bias=False)
        mapping = {mapped: target}
        if 'anima_aliases_from_module_name' in nets:
            for alias in nets['anima_aliases_from_module_name'](state_key.removesuffix('.weight').replace('.', '_')):
                mapping[alias] = target
        shared.sd_model = types.SimpleNamespace(network_layer_mapping=mapping)
        on_disk = types.SimpleNamespace(filename=str(filename))
        nets['read_state_dict'] = lambda path: load_file(str(path))
        net = nets['load_network'](name, on_disk, False, False)
        net.dyn_dim = None
        return net

    def test_plain_lora_negative_and_zero(self):
        filename, _ = self.create()
        expected_delta = self.up @ self.down / 2
        for ratio in (-1., -.5, 0., .5, 1.):
            net = self.load_adapter(filename, '')
            bake['lbw'](net, [ratio]*26, False)
            result, _ = bake['plusweights'](self.base.clone(), next(iter(net.modules.values())))
            torch.testing.assert_close(result, self.base + expected_delta*ratio)

    def test_lokr_factorized_negative(self):
        filename, weights = self.create(kind='lokr')
        net = self.load_adapter(filename, '')
        delta = torch.kron(weights['lokr_w1_a'] @ weights['lokr_w1_b'], weights['lokr_w2_a'] @ weights['lokr_w2_b'])
        for ratio in (-1., -.5, 0., 1.):
            bake['lbw'](net, [ratio]*26, False)
            result, _ = bake['plusweights'](self.base.clone(), next(iter(net.modules.values())))
            torch.testing.assert_close(result, self.base + delta*ratio)

    def test_lora_weights_not_modified_by_strength(self):
        filename, _ = self.create()
        net = self.load_adapter(filename, '')
        adapter = next(iter(net.modules.values()))
        before = adapter.up_model.weight.clone()
        bake['lbw'](net, [-.5]*26, False)
        bake['lbw'](net, [1.]*26, False)
        torch.testing.assert_close(adapter.up_model.weight, before)

    def test_hada_negative(self):
        filename, _ = self.create(kind='hada')
        net = self.load_adapter(filename, '')
        bake['lbw'](net, [-.5]*26, False)
        result, _ = bake['plusweights'](self.base.clone(), next(iter(net.modules.values())))
        delta = (self.up@self.down).square()/2
        torch.testing.assert_close(result, self.base - .5*delta)

    def test_inline_blocks_keep_negative_global_strength(self):
        for text in ('test:-0.5:' + ','.join(['.2']*26), '<lora:test:-0.5:lbw=' + ','.join(['.2']*26) + '>'):
            entries = bake['parse_lora_merge_entries'](text)
            ratios = bake['resolve_bake_ratios'](entries[0], {}, 'unused')
            self.assertEqual(ratios, [-.1]*26)

    def test_named_block_preset_and_nonfinite_input(self):
        result = bake['resolve_bake_ratios'](['test','-.5','ALL'], {'ALL': ','.join(['.2']*26)}, 'unused')
        self.assertEqual(result, [-.1]*26)
        for entry in (['test','nan'], ['test','1','missing'], ['test','1']+['inf']*26):
            with self.assertRaises(ValueError):
                bake['resolve_bake_ratios'](entry, {}, 'unused')

    def test_unmatched_block_uses_base_strength(self):
        obj = types.SimpleNamespace(network_key='other', calc_updown=lambda w: (torch.ones_like(w), None))
        net = types.SimpleNamespace(name='unmatched', modules={'unknown': obj})
        bake['lbw'](net, [-.5]*26, False)
        result, _ = bake['plusweights'](self.base.clone(), obj)
        torch.testing.assert_close(result, self.base - .5)

    def test_norm_bias_receives_same_negative_ratio(self):
        target = torch.nn.LayerNorm(4)
        weights = network.NetworkWeights('norm.w_norm', 'norm', {'w_norm': torch.ones(4), 'b_norm': torch.ones(4)*2}, target)
        net = network.Network('norm', None)
        obj = adapters['norm'].ModuleTypeNorm().create_module(net, weights)
        obj.lbw_ratio = -.5
        result, bias = bake['plusweights'](torch.zeros(4), obj, torch.zeros(4))
        torch.testing.assert_close(result, torch.full((4,), -.5))
        torch.testing.assert_close(bias, torch.full((4,), -1.))
        weights = weights._replace(w={'w_norm': torch.ones(4)})
        self.assertIsNotNone(adapters['norm'].ModuleTypeNorm().create_module(net, weights))

    def test_dora_negative_scales_the_completed_delta(self):
        base = self.base[:,:3].contiguous()
        values = {'lora_up.weight': self.up, 'lora_down.weight': self.down[:,:3],
                  'alpha': torch.tensor(1.), 'dora_scale': torch.tensor([[1.],[2.],[3.],[4.]])}
        net = network.Network('dora', None)
        net.dyn_dim = None
        weights = network.NetworkWeights('dora.lora_up.weight', 'diffusion_model_dora', values, torch.nn.Linear(3,4,bias=False))
        adapter = adapters['lora'].ModuleTypeLora().create_module(net, weights)
        merged = base + self.up @ self.down[:,:3] / 2
        full_delta = torch.stack([row / row.norm() * (i+1) for i,row in enumerate(merged)]) - base
        for strength in (-1., -.5, 0., 1.):
            adapter.lbw_ratio = strength
            result, _ = bake['plusweights'](base.clone(), adapter)
            torch.testing.assert_close(result, base + strength*full_delta)

    def test_missing_network_does_not_save_unchanged_checkpoint(self):
        with patch.object(net_api, 'load_networks', lambda *a: net_api.loaded_networks.clear()):
            with self.assertRaisesRegex(RuntimeError, 'failed to load'):
                bake['newpluslora']({'weight':self.base.clone()}, ['missing'], [[1.]*26], ['missing'], 'float', False, False, False, {})

    def test_qkv_baking_loop_does_not_apply_the_group_twice(self):
        def make_module(key, value):
            return types.SimpleNamespace(shape=(4,4), network_key=key, lbw_ratio=1., calc_updown=lambda w: (torch.full_like(w,value), None))
        net = types.SimpleNamespace(name='fused', modules={'attn_q_proj':make_module('q',1.), 'attn_v_proj':make_module('v',3.)})
        def load(*args):
            net_api.loaded_networks[:] = [net]
        state = {'in.weight': torch.zeros(12,4), 'out.weight': torch.zeros(4,4)}
        with patch.object(net_api, 'load_networks', load), patch.dict(bake, load_state_dict=lambda *a: ({}, {}, False)):
            result = bake['newpluslora'](state, ['fake'], [[-.5]*26], ['fused'], 'float', False, False, False,
                                         {'attn_in_proj':'in.weight', 'attn_out_proj':'out.weight'})
        torch.testing.assert_close(result['in.weight'][:4], torch.full((4,4), -.5))
        torch.testing.assert_close(result['in.weight'][8:], torch.full((4,4), -1.5))

    def test_inpaint_tuple_padding(self):
        obj = types.SimpleNamespace(lbw_ratio=-.5, network_key='inpaint', calc_updown=lambda w: (torch.ones(2,4,1,1), None))
        result, _ = bake['plusweights'](torch.zeros(2,9,1,1), obj)
        torch.testing.assert_close(result[:,:4], torch.full((2,4,1,1), -.5))
        self.assertEqual(torch.count_nonzero(result[:,4:]).item(), 0)

    def test_finite_and_fp16_overflow_abort(self):
        for delta in (float('nan'), 1e10):
            obj = types.SimpleNamespace(lbw_ratio=1., network_key='bad', calc_updown=lambda w: (torch.full_like(w, delta), None))
            original = torch.zeros(2,2,dtype=torch.float16)
            with self.assertRaises(RuntimeError):
                bake['plusweights'](original, obj)
            self.assertEqual(torch.count_nonzero(original).item(), 0)

    def test_fp16_lokr_products_calculate_in_fp32(self):
        filename, weights = self.create(kind='lokr')
        # An fp16 intermediate would be inf, but the alpha-scaled delta fits.
        weights = {k: (torch.full_like(v, 300.).half() if k != 'alpha' else torch.tensor(1e-8)) for k,v in weights.items()}
        save_file({'lora_unet_input_blocks_0_0.'+k:v for k,v in weights.items()}, str(filename))
        net = self.load_adapter(filename, '')
        bake['lbw'](net, [-1.]*26, False)
        result, _ = bake['plusweights'](torch.zeros(4,4), next(iter(net.modules.values())))
        torch.testing.assert_close(result, torch.full((4,4), -81.), rtol=1e-5, atol=1e-5)

    def test_qkv_partial_group_applied_once(self):
        objects = {}
        for proj, value in [('q', 1.), ('v', 3.)]:
            objects['attn_'+proj+'_proj'] = types.SimpleNamespace(lbw_ratio=-.5, network_key=proj,
                calc_updown=lambda w, value=value: (torch.full_like(w, value), None))
        net = types.SimpleNamespace(modules=objects)
        result, out, _ = bake['plusweightsqvk'](torch.zeros(12,4), torch.zeros(4,4), 'attn', None, net)
        torch.testing.assert_close(result[:4], torch.full((4,4), -.5))
        self.assertEqual(torch.count_nonzero(result[4:8]).item(), 0)
        torch.testing.assert_close(result[8:], torch.full((4,4), -1.5))
        self.assertEqual(torch.count_nonzero(out).item(), 0)

    def test_anima_alias_and_block_ratio(self):
        key = 'lora_unet_blocks_27_mlp_0'
        filename, _ = self.create(key)
        state_key = 'model.diffusion_model.blocks.27.mlp.layer1.weight'
        net = self.load_adapter(filename, key, state_key=state_key)
        ratios = [0.]*29
        ratios[28] = -.75
        bake['lbw'](net, ratios, False)
        adapter = next(iter(net.modules.values()))
        result, _ = bake['plusweights'](self.base.clone(), adapter)
        torch.testing.assert_close(result, self.base - .75*(self.up@self.down)/2)
        keychanger = {}
        for alias in bake['anima_key_aliases_from_checkpoint_key'](state_key):
            bake['register_key_alias'](keychanger, alias, state_key)
        resolved, _ = bake['resolve_checkpoint_weight_key']('model_diffusion_model_blocks_27_mlp_0', keychanger, {state_key:self.base}, self.base.shape)
        self.assertEqual(resolved, state_key)

    def test_missing_neo_options_do_not_crash(self):
        nets['networks_in_memory']['old'] = object()
        nets['purge_networks_from_memory']()
        self.assertEqual(nets['networks_in_memory'], {})
        nets['list_available_networks'] = lambda: None
        nets['update_available_networks_by_names'] = lambda n: None
        nets['load_networks'](['missing'])
        self.assertEqual(nets['loaded_networks'], [])

    def test_newpluslora_real_file_roundtrip_and_repeated_bake(self):
        filename, _ = self.create()
        state_key = 'model.diffusion_model.input_blocks.0.0.weight'
        keychanger = {'diffusion_model_input_blocks_0_0': state_key}
        def actual_load(names, *args):
            # Simulate cache reuse if the baker forgets to clear its cache.
            net = net_api.networks_in_memory.get('test')
            if net is None:
                net = self.load_adapter(filename, '')
                net_api.networks_in_memory['test'] = net
            net_api.loaded_networks[:] = [net]
        with patch.object(net_api, 'load_networks', actual_load):
            for strength in (-1., -.5, 0., 1., -.5):
                result = bake['newpluslora']({state_key:self.base.clone()}, [str(filename)], [[strength]*26], ['test'], 'float', False, False, False, keychanger)
                output_path = self.path/'baked.safetensors'
                save_file(result, str(output_path))
                saved = load_file(str(output_path))[state_key]
                torch.testing.assert_close(saved, self.base + strength*(self.up@self.down)/2)

if __name__ == '__main__':
    selected = os.environ.get('BAKE_TEST_FILTER')
    suite = (unittest.TestSuite(BakingTests(name) for name in selected.split(',')) if selected
             else unittest.defaultTestLoader.loadTestsFromTestCase(BakingTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(source=str(TARGET), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))
    (ROOT/(TARGET.name+'-tests.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    sys.exit(not result.wasSuccessful())
