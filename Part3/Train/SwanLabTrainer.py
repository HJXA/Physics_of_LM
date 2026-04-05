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

        # 统一训练类型开关：仅支持 PT / SFT。
        self.train_type = str(Train_type or "PT").upper()
        if self.train_type not in {"PT", "SFT"}:
            raise f"[Trainer警告] 未识别的 Train_type={Train_type}"


        # 语义化布尔开关，便于后续逻辑分支可读。
        self.is_pt_mode = self.train_type == "PT"
        self.is_sft_mode = self.train_type == "SFT"
        
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

        # ====== test_falg 诊断计数器 ======
        # eval 轮次编号（从 1 开始计数）。
        self.eval_round = 0
        # 当前 eval 轮次内已执行的步数。
        self.per_eval_steps = 0
        # 记录上一次 compute_loss 看到的模型模式，便于识别 train/eval 切换边界。
        self._last_model_training_flag: Optional[bool] = None

        # 梯度累加诊断：micro-step 总次数。
        self._test_micro_step_total = 0
        # 梯度累加诊断：进入“真实更新步逻辑”的次数。
        self._test_update_step_total = 0
        # 梯度累加诊断：因非最后累加步而跳过自定义逻辑的次数。
        self._test_skipped_accum_forward_total = 0

        self.rank = self.accelerator.process_index if hasattr(self, 'accelerator') else 0

        if self.test_falg and self.rank == 0:
            print(
                f"[Trainer模式] 当前训练类型(train_type)={self.train_type}。"
            )

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
        is_training = bool(getattr(model, "training", False))

        if self.test_falg:
            # 首次调用只记录状态，不触发“切换边界”逻辑。
            if self._last_model_training_flag is None:
                self._last_model_training_flag = is_training
                if not is_training:
                    self.eval_round = 1
                    self.per_eval_steps = 0
                    if self.rank == 0:
                        print(f"[评估测试] 首次进入第 {self.eval_round} 轮评估，轮内步数(per_eval_steps)已重置为 0")
            else:
                # train -> eval：新一轮 eval 开始，步数清零并轮次 +1。
                if self._last_model_training_flag and (not is_training):
                    self.eval_round += 1
                    self.per_eval_steps = 0
                    if self.rank == 0:
                        print(f"[评估测试] 进入第 {self.eval_round} 轮评估，轮内步数(per_eval_steps)已重置为 0")

                # eval -> train：上一轮 eval 结束，打印总结并清零轮内步数。
                elif (not self._last_model_training_flag) and is_training:
                    if self.rank == 0:
                        print(f"[评估测试] 第 {self.eval_round} 轮评估结束，本轮步数(per_eval_steps)={self.per_eval_steps}，现重置为 0")
                    self.per_eval_steps = 0

            # 仅在 eval 模式内累计本轮步数。
            if not is_training:
                self.per_eval_steps += 1

            # 更新“上一次模式”记录。
            self._last_model_training_flag = is_training

            # 只在主进程打印，避免多卡重复日志。
            if self.rank == 0:
                if is_training:
                    print(
                        f"训练进度日志: 全局更新步(step_count)={self.step_count}，当前模型状态=训练，"
                        f"是否跳过自定义loss计算={not is_training}"
                    )
                else:
                    print(
                        f"训练进度日志: 全局更新步(step_count)={self.step_count}，当前模型状态=评估/验证，"
                        f"评估轮数(eval_round)={self.eval_round}，该轮步数(per_eval_steps)={self.per_eval_steps}，"
                        f"是否跳过自定义loss计算={not is_training}"
                    )

        return not is_training


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

        if self.test_falg and self.rank == 0:
            self._test_micro_step_total += 1
            current_global_step = int(getattr(self.state, "global_step", 0))
            print(
                f"[梯度累加测试][compute_loss] 微步统计: "
                f"累计前向次数(micro_step_total)={self._test_micro_step_total}, "
                f"梯度累加配置(grad_acc_steps)={grad_acc_steps}, "
                f"是否为最后累加前向(is_last_accum_forward)={is_last_accum_forward}, "
                f"更新前全局步(global_step_before)={current_global_step}, "
                f"更新前自定义步(step_count_before)={self.step_count}"
            )

        # 正在做梯度累加且不是最后一次反向传播前向：走父类快速路径，跳过所有额外计算。
        if not is_last_accum_forward:
            if self.test_falg and self.rank == 0:
                self._test_skipped_accum_forward_total += 1
                print(
                    f"[梯度累加测试][compute_loss] 当前为非更新微步(micro-step)，已跳过自定义逻辑；"
                    f"累计跳过次数(skipped_accum_forward_total)={self._test_skipped_accum_forward_total}"
                )
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # 仅在真实参数更新步递增 step_count，保持其语义与 global update step 一致。
        prev_step_count = self.step_count
        self.step_count += 1

        if self.test_falg and self.rank == 0:
            self._test_update_step_total += 1
            expected_step_count = int(getattr(self.state, "global_step", 0)) + 1
            step_match = self.step_count == expected_step_count
            print(
                f"[梯度累加测试][compute_loss] 进入更新步逻辑: "
                f"step_count(全局更新步计数) {prev_step_count}->{self.step_count}, "
                f"期望step_count(=global_step+1)={expected_step_count}, "
                f"对齐校验(step_match)={step_match}, "
                f"累计更新步次数(update_step_total)={self._test_update_step_total}"
            )

        # ==================== 下面是真正的更新步或无梯度累加才会进入的代码 ====================

        if self.step_count == 1 and self.rank == 0:
            print("首步输入张量形状: input_ids.shape=", inputs['input_ids'].shape)
            print("首步标签张量形状: labels.shape=", inputs['labels'].shape)


        if self.test_falg and self.rank==0:
            print("============测试模式日志============")
            if self.step_count == 1 and self.rank == 0:
                print("首步input_ids明细:", inputs['input_ids'].tolist())
                print("首步labels明细:", inputs['labels'].tolist())
            torch.cuda.synchronize()
            comput_loss_start = time.time()

        current_labels = inputs.get("labels")

        if current_labels is None:
            print("================\n警告: current_labels 为 None，后续部分统计将不可用。\n================")

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
                    "Trainer 提示: 已定义 `compute_loss_func`，但当前 labels=None；"
                    "自定义损失函数仍会被调用，请确认这符合预期。"
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
            print(f"=== 第 {self.step_count} 个更新步日志 ===")
            print("模型输出字段(keys):", outputs.keys())
            if hasattr(outputs, "logits"):
                print("logits 张量形状(logits.shape):", outputs.logits.shape)
            else:
                print("警告: outputs 中没有 logits 字段，无法打印 logits 形状。")

            print("当前损失值(loss):", loss.item())

        if self.test_falg and self.rank==0:
            torch.cuda.synchronize()
            print("原始 compute_loss 阶段耗时(秒):", time.time()-comput_loss_start)
            layer_hidden_start = time.time()
            print("当前标签张量形状(labels.shape):", current_labels.shape)
            print("标签中忽略位(-100)占比:", (current_labels == -100).float().mean())


    

        

        layer_hidden_state = Layer_Hidden_Train(outputs.hidden_states, labels = current_labels,eos_token_id=getattr(self.model.config, 'eos_token_id', None),pad_token_id=getattr(self.model.config, 'pad_token_id', None), steps = self.step_count, rank = self.accelerator.process_index, input_ids = inputs.get('input_ids'),train_type=self.train_type)
        # (Batch_Real, Layer, Hidden_Dim)
        if layer_hidden_state is not None:

            if self.test_falg and self.rank==0:
                torch.cuda.synchronize()
                print("Layer_Hidden_Train 计算耗时(秒):", time.time()-layer_hidden_start)


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
                print("Layer Hidden 异步提交耗时(秒):", time.time()-save_layer_start)
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
                print("CoE 指标计算与记录耗时(秒):", time.time()-coe_start)
        else:
            print(f"更新步(step_count)={self.step_count}, 进程(rank)={self.rank}: Layer_Hidden_Train 返回 None，未保存 Layer Hidden State，且未计算 CoE 指标。")


        # ==============================================================================

        # Acc

        # 计算 token 的 Accuracy
        acc = 0.0
        if current_labels is not None and hasattr(outputs, "logits"):
            logits = outputs.logits
            
            # Causal LM（自回归）统一采用 next-token 预测，因此 PT/SFT 都需要 shift。
            # PT 和 SFT 的区别在 labels 的“mask”，不在“shift”，shift都是自动做的。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = current_labels[..., 1:].contiguous()
            
            # 预测类别
            preds = shift_logits.argmax(dim=-1)

            if self.test_falg and self.rank==0:
                print("错位后真实标签(shift_labels):", shift_labels.tolist())
                print("错位后预测类别(preds):", preds.tolist())
            
            # 有效监督位掩码：labels != -100 的位置才参与准确率统计。
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
        before_global_step = int(getattr(self.state, "global_step", 0))
        before_step_count = self.step_count

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            step_start = time.time()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            after_global_step = int(getattr(self.state, "global_step", 0))
            after_step_count = self.step_count
            is_update_step = after_global_step > before_global_step

            if is_update_step:
                inc_ok = after_step_count == (before_step_count + 1)
                align_ok = after_step_count == after_global_step
                print(
                    f"[梯度累加测试][training_step] 本次为更新步(UPDATE): "
                    f"global_step {before_global_step}->{after_global_step}, "
                    f"step_count {before_step_count}->{after_step_count}, "
                    f"step_count递增+1校验(inc_ok)={inc_ok}, "
                    f"step_count与global_step对齐校验(align_ok)={align_ok}"
                )
            else:
                stable_ok = after_step_count == before_step_count
                print(
                    f"[梯度累加测试][training_step] 本次为非更新微步(micro-step): "
                    f"global_step {before_global_step}->{after_global_step}, "
                    f"step_count {before_step_count}->{after_step_count}, "
                    f"step_count稳定性校验(stable_ok)={stable_ok}"
                )

            print("training_step 总耗时(秒):", time.time()-step_start)

        return loss
