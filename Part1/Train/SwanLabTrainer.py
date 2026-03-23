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

import sys
sys.path.append("/ruilab/jxhe/CoE_Monitor/utils")

from Layer_Hidden import Layer_Hidden_Train
from Coe_Scores_Batch import CoEScoreInfo_Train as CoEScoreInfo_Batch


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


    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        
        # 复制 inputs 以防止原始字典中的 labels 被弹出
        # inputs_for_loss = inputs.copy()
        # labels = inputs.get("labels")

        self.step_count += 1
        
        # 当正在进行梯度累加且不是最后一次反向传播前向时，直接调用父类方法，跳过所有额外计算以提升速度
        if self.args.gradient_accumulation_steps > 1 and self.step_count % self.args.gradient_accumulation_steps != 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # 针对当前步计算实际更新后的真实 Global Step
        real_global_step = int(self.step_count // self.args.gradient_accumulation_steps)

        # ==================== 下面是真正的更新步或无梯度累加才会进入的代码 ====================

        if real_global_step == 1 and self.rank == 0:
            print("input_ids.shape:", inputs['input_ids'].shape)
            print("input的labels.shape:", inputs['labels'].shape)


        if self.test_falg and self.rank==0:
            print("============测试模式=============")
            if real_global_step == 1 and self.rank == 0:
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

        if real_global_step == 1 and self.rank == 0:
            print(f"=== Step {real_global_step} ===")
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


    

        

        layer_hidden_state = Layer_Hidden_Train(outputs.hidden_states, labels = current_labels,eos_token_id=getattr(self.model.config, 'eos_token_id', None),pad_token_id=getattr(self.model.config, 'pad_token_id', None), steps = real_global_step, rank = self.accelerator.process_index, input_ids = inputs.get('input_ids'),train_type=self.train_type)
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
                f"Step{real_global_step}_Rank{self.rank}.pt"
            )

            tensor_to_save = (
                layer_hidden_state[:10]
                .detach()
                .to(torch.bfloat16)   # 强烈建议压缩
                .cpu()
            )
            if real_global_step == 1 and self.rank == 0:
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


            z_ang_mean, a_in_mean, a_mid_mean, a_out_mean = CoEScoreInfo_Batch(layer_hidden_state).compute_CoE_Ang()


            metrics = {
                "CoE/Z_A_Mean": z_ang_mean,
                "CoE/A_In_Mean": a_in_mean,
                "CoE/A_Mid_Mean": a_mid_mean,
                "CoE/A_Out_Mean": a_out_mean
            }

            if self.rank == 0:
                swanlab.log(metrics, step=real_global_step)

            layer_hidden_state = None  # 释放内存

            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                print("coe_add",time.time()-coe_start)
        else:
            print(f"Step {real_global_step} Rank {self.rank}: Layer_Hidden_Train 返回 None，未保存 Layer Hidden State 也未计算 CoE 指标。")


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
            }, step=real_global_step)

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
