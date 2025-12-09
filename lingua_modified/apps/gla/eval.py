# Copyright (c) Meta Platforms, Inc. and affiliates.

# This file is adapted from the original implementation in:
# https://github.com/facebookresearch/lingua/
# Released under the BSD 3-Clause License by:
# Mathurin Videau*, Badr Youbi Idrissi*, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, and David Lopez-Paz.
#
# Modifications made by Zeyuan Allen-Zhu include:
# - removed bos_token from lm_eval loglikelihood tests --- this matches the behavior of lm_eval's original HFLM code
# - added support to use HFLM's wrapper around our model for lm_eval evaluation
#
# These modifications are licensed under the Apache 2.0 license, as stated in the root LICENSE file.
#

import os,socket
if True:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("Running in offline mode for Huggingface")

import sys, os, gc
# print(sys.executable)
import socket
# print(f"Running on host: {socket.gethostname()}")
# print(f"Env MASTER_ADDR = {os.environ.get('MASTER_ADDR', 'Not set')}")
# print(f"Env MASTER_PORT = {os.environ.get('MASTER_PORT', 'Not set')}")
# print(f"Env LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
# print(f"Env LD_PRELOAD = {os.environ.get('LD_PRELOAD', 'Not set')}")
# print(f"Env HF_HOME = {os.environ.get('HF_HOME', 'Not set')}")
# print(f"Env TRITON_CACHE_DIR = {os.environ.get('TRITON_CACHE_DIR', 'Not set')}")

import atexit, tempfile, shutil, os

import torch.nn as nn
import torch
from lm_eval.models.huggingface import HFLM

import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from omegaconf import OmegaConf
import torch
from apps.gla.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.gla.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.data import init_choice_state, setup_sources
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from lingua.metrics import (
    GPUMemoryMonitor,
)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234

@dataclass
class ValidationArgs:
    max_steps: Optional[int] = None # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    use_val_from_train_src: bool = True # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list) # Other sources to eval on

@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation

    # Zeyuan's new 
    no_eval: bool = False  # If True, after processing folders, exit withotu eval



def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)



class MockAccelerator:
    def __init__(self):
        if torch.distributed.get_world_size()>32:
            self.init_group = torch.distributed.new_group(ranks=list(range(32)))
        else:
            self.init_group = torch.distributed.group.WORLD
    def gather(self, tensor):  
        l = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size(group=self.init_group))]
        torch.distributed.all_gather(l, tensor, group=self.init_group)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier(group=self.init_group)    

    def unwrap_model(self, model):   
        return model

class HFLikeModelAdapter(nn.Module):

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.config = "fake"
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                self.device = p.device
                print(f"Model parameter {n} is on device {self.device}")
                break
        #self.device = model.trans.lm_head.weight.device
        self.tie_weights = lambda: self
        self.gpu_memory_monitor = GPUMemoryMonitor("cuda")
        self.counter = 0
        self.gpu_memory_monitor.reset_peak_stats()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.counter+=1
        if self.counter%100==0:
            gpu_mem_stats = self.gpu_memory_monitor.get_peak_stats()
            print(f"Past 100 iterations, GPU mem: {gpu_mem_stats.max_active_pct:.1f}%, pow: {gpu_mem_stats.power_draw/1000}W; GC will be applied", flush=True)
            gc.collect()
            self.counter=0
            self.gpu_memory_monitor.reset_peak_stats()

        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if hasattr(self.model, 'traditional_forward'):
                    output = self.model(input_ids, **kwargs)
                else:
                    output = self.model(input_ids, attention_mask=attention_mask, **kwargs)

        # Make sure output has the expected .logits attribute
        if not hasattr(output, "logits"):
            if isinstance(output, torch.Tensor):
                output.logits = output
        return output

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.counter+=1
        if self.counter%100==0:
            gpu_mem_stats = self.gpu_memory_monitor.get_peak_stats()
            print(f"Past 100 iterations, GPU mem: {gpu_mem_stats.max_active_pct:.1f}%, pow: {gpu_mem_stats.power_draw/1000}W; GC will be applied", flush=True)
            gc.collect()
            self.counter=0
            self.gpu_memory_monitor.reset_peak_stats()
        assert 'max_length' in kwargs or 'max_new_tokens' in kwargs
        if 'max_length' in kwargs: assert 'max_new_tokens' not in kwargs
        kwargs['use_cache'] = True  # in case this is forgotton

        kwargs['pad_token_id'] = self.tokenizer.bos_token_id  # niah dataset的时候它们会给出pad_token_id=0

        with torch.inference_mode():
            if False:
                res = self.model.generate( input_ids=input_ids, attention_mask=attention_mask, eos_token_id=self.tokenizer.bos_token_id, **kwargs,)
            else:
                with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
                    res = self.model.generate( input_ids=input_ids, attention_mask=attention_mask, eos_token_id=self.tokenizer.bos_token_id, **kwargs,)
        return res

    # Only delegate specific attributes we know we need
    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
        
    

def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0
    for src in train_cfg.data.sources:
        path = os.path.join(train_cfg.data.root_dir, src)
        srcs[path] = 1.0

    multi_state = init_choice_state("", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl")
    path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state['current_iter'] > 0 or (val_args.max_steps is not None and step >= val_args.max_steps):
                break
            content_key = "text" if ("text" in content) else "content"
            texts.append(content[content_key])
        
        _, loglikelihood, _ = generator.generate(texts)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            tmp = ll.sum().item()
            metrics['nll'].append(tmp)
            metrics['nll_per_token'].append(tmp / len(ll))
            metrics['nll_per_char'].append(tmp / len(texts[i]))

            metrics['avg_seqlen'].append(len(ll))
        
        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics

def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        #if not consolidate_path.exists() and get_global_rank() == 0:
        if get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    if True:
        consolidate_path = str(consolidate_path)
        torch.distributed.barrier()
        logger.info("Loading model")
        model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
            consolidate_path,
            model_cls=LMTransformer,
            model_args_cls=LMTransformerArgs,
        )
        logger.info("Model loaded")
        model.eval()
        # for n,p in model.named_parameters():
        #     print(f"{n}, shape={p.shape}, mean={p.mean().item()}, std={p.std().item()}")

        model1 = HFLikeModelAdapter(model.trans, tokenizer)
        model2 = HFLM(pretrained=model1, tokenizer=tokenizer, backend="causal", max_length=2048)  
        if torch.distributed.is_initialized():
            model2._rank = torch.distributed.get_rank()
            model2.accelerator = MockAccelerator()
            model2._world_size = torch.distributed.get_world_size(group=model2.accelerator.init_group)

        print(asdict(cfg.harness))

        do_evaluate = True
        if torch.distributed.is_initialized() and torch.distributed.get_rank()>=32:
            do_evaluate = False
        if do_evaluate:
            results = simple_evaluate(
                model=model2,
                #device=device,
                #tasks=tasks,
                #verbosity="INFO",
                **asdict(cfg.harness)
            )
        else:
            results = {}


        val_results =  None
        if cfg.validation:
            val_results = eval_on_val(generator, cfg.validation, train_cfg)
        if get_global_rank() == 0:
            with open(Path(cfg.dump_dir) / "results.json", "w") as f:
                f.write(json.dumps(results))
            logger.info(f"All evaluation results: {results['results']}")
            if val_results is not None:
                with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                    f.write(json.dumps(val_results))
                logger.info(f"All validation results: {val_results}")
        if cfg.metric_log_dir and get_global_rank() == 0:
            metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

            logger.info(f"Writing metric logs to {metric_log_path}")
            timestamp = {
                "created_at": datetime.utcnow().isoformat(),
            }
            if cfg.global_step is not None:
                timestamp["global_step"] = cfg.global_step
            print(
                json.dumps(timestamp | results["results"]),
                file=open(metric_log_path, mode="a"),
                flush=True,
            )

            val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
            if val_results is not None:
                print(
                    json.dumps(timestamp | val_results),
                    file=open(val_log_path, mode="a"),
                    flush=True,
                )
        
        #del generator



def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """

    # When using Triton, it attempts to locate prebuilt kernels in a cache
    # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # it to a local directory it would belong to the first user who created it
    # and it would fail for the job of any other successive user assigned to
    # that machine. To avoid all this mess we use a temporary per-process cache.
    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    print(f"Using Triton cache dir: {triton_cache_dir}")

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
