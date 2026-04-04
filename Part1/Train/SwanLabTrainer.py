import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from transformers import Trainer

# comput_loss
from typing_extensions import override
from transformers.trainer import deepspeed_sp_compute_loss, _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import threading
import queue
import re
import time
import os
import pickle
from pathlib import Path

try:
    from utils.Layer_Hidden import Layer_Hidden_Train
    from utils.Coe_Scores_Batch import CoEScoreInfo_Train as CoEScoreInfo_Batch
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.Layer_Hidden import Layer_Hidden_Train
    from utils.Coe_Scores_Batch import CoEScoreInfo_Train as CoEScoreInfo_Batch


import swanlab

class SwanLabTrainer(Trainer):
    def __init__(self, *args, swanlab_project="test_physics_lm", swanlab_experiment_name="test", swanlab_description="gpt test Training with SwanLab",  test_falg = False,CoE_Flag=True, Train_type=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_type = Train_type
        
        # 1. 开启一个 SwanLab 实验
        # 将 TrainingArguments 中的超参数存入 swanlab.config
        swanlab.init(
            project=swanlab_project,
            experiment_name=swanlab_experiment_name,
            description=swanlab_description,
            config=self.args.to_dict()
        )

        self.CoE_Flag=CoE_Flag
        self.step_count = 0 

        self.input = None
        self.output = None
        self.test_falg = test_falg

        self.rank = self.accelerator.process_index if hasattr(self, 'accelerator') else 0

        if CoE_Flag:

            if not hasattr(self, 'args') or not self.args.resume_from_checkpoint: 

                self.save_layer_hidden_root = os.path.join(
                    self.args.output_dir.replace("output", "coe_train_result"),
                    "Layer_Hidden_Train"
                )
            else:
                try:
                    base = self.args.resume_from_checkpoint.replace("output", "coe_train_result")

                    self.save_layer_hidden_root = os.path.join(
                        os.path.dirname(base),
                        "Layer_Hidden_Train"
                    )
                    print(f"[CoeTrainer] 从 checkpoint 路径构建 CoE 保存路径: {self.save_layer_hidden_root}")
                except Exception as e:
                    print(f"[CoeTrainer] 从 checkpoint 路径构建 CoE 保存路径失败: {e}")
                    self.save_layer_hidden_root = os.path.join(
                        self.args.output_dir.replace("output", "coe_train_result"),
                        "Layer_Hidden_Train"
                    )
                    print(f"[CoeTrainer] 已使用默认路径: {self.save_layer_hidden_root}")
            os.makedirs(self.save_layer_hidden_root, exist_ok=True)


            self._try_load_coe_state()


            # ========异步保存===========
            self._save_queue = queue.Queue(maxsize=128) 
            def _async_save_worker():
                while True:
                    item = self._save_queue.get()
                    if item is None:
                        break
                    
                    try:
                        task_type = item.get("type")
                        
                        if task_type == "tensor":
                            # 处理 Tensor 保存
                            torch.save(item["data"], item["path"])
                            
                        elif task_type == "coe":
                            # 处理 CoE 保存
                            with open(item["path"], "wb") as f:
                                pickle.dump(item["data"], f)
                                
                            # 顺便在后台处理文件删除，避免阻塞
                            for rm_path in item.get("remove_paths", []):
                                if os.path.exists(rm_path):
                                    os.remove(rm_path)
                    except Exception as e:
                        print(f"[AsyncWorker] 保存出错: {e}")
                    finally:
                        self._save_queue.task_done()

            self._save_thread = threading.Thread(
                target=_async_save_worker,
                daemon=True
            )
            self._save_thread.start()

            print(f"[CoeTrainer] Rank {self.rank} 初始化完成")

    def _submit_async_layer_save(self, tensor, path):
        """提交 Tensor 保存任务"""
        self._save_queue.put({"type": "tensor", "data": tensor, "path": path})

    def _submit_async_coe_save(self, coe_data, path, remove_paths):
        """提交 CoE 列表保存及旧文件清理任务"""
        self._save_queue.put({
            "type": "coe", 
            "data": coe_data, 
            "path": path, 
            "remove_paths": remove_paths
        })



    def _try_load_coe_state(self):
        """
        尝试从保存目录中加载最新的 Coe 和 step_count
        """
        # 1. 构建保存路径 (逻辑需与保存代码完全一致)
        # 注意：这里假设 self.args.output_dir 已经被父类初始化
        if not hasattr(self, 'args') or not self.args.resume_from_checkpoint:
            print("[CoeTrainer] args 中未设置 resume_from_checkpoint，无法加载 CoE 状态。")
            return

        resume_dir = self.args.resume_from_checkpoint
        print(f"[CoeTrainer] 尝试从 {resume_dir} 加载步数...")

        dirname = os.path.basename(resume_dir)   # checkpoint-34000

        pattern = re.compile(r"checkpoint-(\d+)")
        match = pattern.match(dirname)

        max_step = 0
        if match:
            max_step = int(match.group(1))

        self.step_count = max_step

        print(f"[CoeTrainer] Rank {self.rank}: 步数已设置为 {max_step}")

    def _should_skip_custom_loss(self, model: nn.Module) -> bool:
        # 评估/验证阶段 prediction_step 会调用 compute_loss，此时直接复用父类逻辑。
        return not bool(getattr(model, "training", False))


    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:

        if self._should_skip_custom_loss(model):
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        
        # 复制 inputs 以防止原始字典中的 labels 被弹出
        # inputs_for_loss = inputs.copy()
        # labels = inputs.get("labels")

        grad_acc_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
        is_last_accum_forward = True
        if model.training and grad_acc_steps > 1:
            # accelerate 会在真正需要梯度同步/更新的 micro-step 上将 sync_gradients 设为 True。
            # 这样也能正确覆盖 epoch 尾部不足 grad_acc_steps 的最后一个 micro-step。
            is_last_accum_forward = bool(getattr(self.accelerator, "sync_gradients", True))

        # 正在做梯度累加且不是最后一次反向传播前向：走父类快速路径，跳过所有额外计算。
        if not is_last_accum_forward:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # 仅在真实参数更新步递增 step_count，保持其语义与 global update step 一致。
        self.step_count += 1

        # ==================== 下面是真正的更新步或无梯度累加才会进入的代码 ====================

        if self.step_count == 1 and self.rank == 0:
            print("input_ids.shape:", inputs['input_ids'].shape)
            print("input的labels.shape:", inputs['labels'].shape)


        if self.test_falg and self.rank==0:
            print("============测试模式=============")
            if self.step_count == 1 and self.rank == 0:
                print("input_ids:", inputs['input_ids'].tolist())
                # print("labels:", inputs['labels'].tolist())
            torch.cuda.synchronize()
            comput_loss_start = time.time()

        current_labels = inputs.get("labels")

        if current_labels is None:
            print("================\nWarning: current_labels is None.\n================")

        # 原始compute_loss==============================================================================

        # 调用父类的 compute_loss，一定要用 return_outputs=True 拿到 outputs 以便计算 acc
        # loss, outputs = super().compute_loss(
        #     model, 
        #     inputs_for_loss, 
        #     return_outputs=True, 
        #     num_items_in_batch=num_items_in_batch
        # )
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled and self.model.training:
            return deepspeed_sp_compute_loss(self.accelerator, model, inputs, return_outputs, pc)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        # outputs = model(**inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                print(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
        # ==============================================================================

        # CoE

        if self.step_count == 1 and self.rank == 0:
            print(f"=== Step {self.step_count} ===")
            print("模型输出 keys:", outputs.keys())
            if hasattr(outputs, "logits"):
                print("logits shape:", outputs.logits.shape)
            else:
                print("Warning: outputs 中没有 logits 字段，无法打印形状。")

            print("loss:", loss.item())

        if self.test_falg and self.rank==0:
            torch.cuda.synchronize()
            print("raw_compute_loss",time.time()-comput_loss_start)
            layer_hidden_start = time.time()
            print("labels 的形状",current_labels.shape)
            print(f"labels 中 -100 比例",(current_labels == -100).float().mean())


    

        

        layer_hidden_state = Layer_Hidden_Train(outputs.hidden_states, labels = current_labels,eos_token_id=getattr(self.model.config, 'eos_token_id', None),pad_token_id=getattr(self.model.config, 'pad_token_id', None), steps = self.step_count, rank = self.accelerator.process_index, input_ids = inputs.get('input_ids'),train_type=self.train_type)
        # (Batch_Real, Layer, Hidden_Dim)
        if layer_hidden_state is not None:

            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                print("layer_hidden_time",time.time()-layer_hidden_start)


            if self.test_falg and self.rank==0:
                self.output = outputs 

            outputs.hidden_states = None  # 释放内存

            save_path = os.path.join(
                self.save_layer_hidden_root,
                f"Step{self.step_count}_Rank{self.rank}.pt"
            )

            tensor_to_save = (
                layer_hidden_state[:10]
                .detach()
                .to(torch.bfloat16)   # 强烈建议压缩
                .cpu()
            )
            if self.step_count == 1 and self.rank == 0:
                print(f"准备保存 Layer Hidden State: {tensor_to_save.shape}, layer_hidden_state 原始形状: {layer_hidden_state.shape}")

            
            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                save_layer_start = time.time()
            self._submit_async_layer_save(tensor_to_save, save_path)
            # torch.save(tensor_to_save, save_path)
            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                print("save_layer_time",time.time()-save_layer_start)
            tensor_to_save = None

            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                coe_start = time.time()


            z_ang_mean, a_in_mean, a_mid_mean, a_out_mean = CoEScoreInfo_Batch(layer_hidden_state).compute_CoE()


            metrics = {
                "CoE/Z_A_Mean": z_ang_mean,
                "CoE/A_In_Mean": a_in_mean,
                "CoE/A_Mid_Mean": a_mid_mean,
                "CoE/A_Out_Mean": a_out_mean
            }

            if self.rank == 0:
                swanlab.log(metrics, step=self.step_count)

            layer_hidden_state = None  # 释放内存

            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                print("coe_add",time.time()-coe_start)
        else:
            print(f"Step {self.step_count} Rank {self.rank}: Layer_Hidden_Train 返回 None，未保存 Layer Hidden State 也未计算 CoE 指标。")


        # ==============================================================================

        # Acc

        # 计算 token 的 Accuracy
        acc = 0.0
        if current_labels is not None and hasattr(outputs, "logits"):
            logits = outputs.logits
            
            # 对于 Causal LM (自回归)，预测下一个 token，因此需要 shift (错位)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = current_labels[..., 1:].contiguous()
            
            # 预测类别
            preds = shift_logits.argmax(dim=-1)

            if self.test_falg and self.rank==0:
                print("shift_labels:", shift_labels.tolist())
                print("preds:", preds.tolist())
            
            # 去除 padding (通常为 -100) 的影响
            valid_mask = (shift_labels != -100)
            
            # 计算预测正确的 token 数量
            correct_preds = (preds == shift_labels) & valid_mask
            
            # 计算正确率
            if valid_mask.sum().item() > 0:
                acc = correct_preds.sum().item() / valid_mask.sum().item()

        # 3. 在训练步长内记录指标
        # 这里只有在模型处于训练模式且在可接受的设备环境中时记录 (防止多卡环境重复记录)
        # 您还可以利用 self.state.global_step 加入 logging_steps 的判别
        if self.rank == 0:  # 仅主进程记录，避免多卡重复
            swanlab.log({
                "loss": loss.item(), 
                "acc": acc
            }, step=self.step_count)

        return (loss, outputs) if return_outputs else loss
    
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            step_start = time.time()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            print("step_time=",time.time()-step_start)

        return loss

    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, torch.Tensor | Any],
    #     prediction_loss_only: bool,
    #     ignore_keys: list[str] | None = None,
    # ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    #     """
    #     Perform an evaluation step on `model` using `inputs`.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to evaluate.
    #         inputs (`dict[str, torch.Tensor | Any]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #         prediction_loss_only (`bool`):
    #             Whether or not to return the loss only.
    #         ignore_keys (`list[str]`, *optional*):
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions.

    #     Return:
    #         tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
    #         logits and labels (each being optional).
    #     """
    #     has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
    #     # For CLIP-like models capable of returning loss values.
    #     # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
    #     # is `True` in `model.forward`.
    #     return_loss = inputs.get("return_loss")
    #     if return_loss is None:
    #         return_loss = self.can_return_loss
    #     loss_without_labels = len(self.label_names) == 0 and return_loss

    #     inputs = self._prepare_inputs(inputs)
    #     if ignore_keys is None:
    #         if hasattr(self.model, "config"):
    #             ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
    #         else:
    #             ignore_keys = []

    #     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    #     if has_labels or loss_without_labels:
    #         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
    #         if len(labels) == 1:
    #             labels = labels[0]
    #     else:
    #         labels = None

    #     with torch.no_grad():
    #         if is_sagemaker_mp_enabled():
    #             raw_outputs = smp_forward_only(model, inputs)
    #             if has_labels or loss_without_labels:
    #                 if isinstance(raw_outputs, dict):
    #                     loss_mb = raw_outputs["loss"]
    #                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
    #                 else:
    #                     loss_mb = raw_outputs[0]
    #                     logits_mb = raw_outputs[1:]

    #                 loss = loss_mb.reduce_mean().detach().cpu()
    #                 logits = smp_nested_concat(logits_mb)
    #             else:
    #                 loss = None
    #                 if isinstance(raw_outputs, dict):
    #                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
    #                 else:
    #                     logits_mb = raw_outputs
    #                 logits = smp_nested_concat(logits_mb)
    #         else:
    #             if has_labels or loss_without_labels:
    #                 with self.compute_loss_context_manager():
    #                     num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
    #                     loss, outputs = self.compute_loss(
    #                         model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
    #                     )
    #                 loss = loss.detach().mean()

    #                 if isinstance(outputs, dict):
    #                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
    #                 else:
    #                     logits = outputs[1:]
    #             else:
    #                 loss = None
    #                 with self.compute_loss_context_manager():
    #                     outputs = model(**inputs)
    #                 if isinstance(outputs, dict):
    #                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
    #                 else:
    #                     logits = outputs

    #     if prediction_loss_only:
    #         return (loss, None, None)

    #     logits = nested_detach(logits)
    #     if len(logits) == 1:
    #         logits = logits[0]

    #     return (loss, logits, labels)