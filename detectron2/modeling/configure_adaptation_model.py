import numpy as np
import torch
import logging
import warnings
import fvcore.common.checkpoint as fvcore_checkpoint
import torch.nn as nn
from detectron2.layers import FrozenBatchNorm2d
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger


def configure_model(cfg, trainer, model=None, revert=True, lr=None, weight_path=None):
    # revert to the source trained weight
    if model is None or revert:
        model = trainer.build_model(cfg)
        # warnings.filterwarnings("ignore", category=UserWarning)
        # torch.nn.modules.module.register_parameter_buffer_warning_hook(lambda name, is_buffer: False)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if weight_path is None else weight_path, resume=False
        )
        #if cfg.TEST.ADAPTATION.WHERE == 'adapter' and "resnet" in cfg.MODEL.BACKBONE.NAME:
        #    model.backbone.bottom_up.add_adapter(cfg.TEST.ADAPTER.TYPE, cfg.TEST.ADAPTER.THRESHOLD, scalar=cfg.TEST.ADAPTER.SCALAR)
        #    model.to(model.device)
        model.initialize()

    model.eval()
    model.requires_grad_(False)
    lr_ = cfg.SOLVER.BASE_LR * max(cfg.SOLVER.IMS_PER_BATCH_TEST // 4, 1) if lr is None else lr
    params = []
    bn_params = []
    warnings.filterwarnings("ignore", message=".*are not found in the checkpoint.*")
    logging.getLogger("fvcore.common.checkpoint").setLevel(logging.ERROR)
    if cfg.TEST.ADAPTATION.WHERE == 'smi':
        if "resnet" in cfg.MODEL.BACKBONE.NAME:
            for stage in model.backbone.bottom_up.stages:
                for block in model.backbone.bottom_up.res5:
                    if hasattr(block.conv1, 'down_proj'):# conv1
                        block.conv1.down_proj.requires_grad_(True)
                        params += list(block.conv1.down_proj.parameters())
                    if hasattr(block.conv1, 'up_proj'):
                        block.conv1.up_proj.requires_grad_(True)
                        params += list(block.conv1.up_proj.parameters())
                    block.conv1.requires_grad_(True)# 开启 conv1 本身的梯度更新
                    params += list(block.conv1.parameters())
                    if hasattr(block.conv2, 'down_proj'):# conv2
                        block.conv2.down_proj.requires_grad_(True)
                        params += list(block.conv2.down_proj.parameters())
                    if hasattr(block.conv2, 'up_proj'):
                        block.conv2.up_proj.requires_grad_(True)
                        params += list(block.conv2.up_proj.parameters())
                    block.conv2.requires_grad_(True)
                    params += list(block.conv2.parameters())
                    if hasattr(block.conv3,'down_proj'):# conv3
                        block.conv3.down_proj.requires_grad_(True)
                        params += list(block.conv3.down_proj.parameters())
                    if hasattr(block.conv3, 'up_proj'):
                        block.conv3.up_proj.requires_grad_(True)
                        params += list(block.conv3.up_proj.parameters())
                    block.conv3.requires_grad_(True)
                    params += list(block.conv3.parameters())
            for stage in model.backbone.bottom_up.stages:
                for block in model.backbone.bottom_up.res4:
                    if hasattr(block.conv1, 'down_proj'):# conv1
                        block.conv1.down_proj.requires_grad_(True)
                        params += list(block.conv1.down_proj.parameters())
                    if hasattr(block.conv1, 'up_proj'):
                        block.conv1.up_proj.requires_grad_(True)
                        params += list(block.conv1.up_proj.parameters())
                    block.conv1.requires_grad_(True)# 开启 conv1 本身的梯度更新
                    params += list(block.conv1.parameters())
                    if hasattr(block.conv2, 'down_proj'):# conv2
                        block.conv2.down_proj.requires_grad_(True)
                        params += list(block.conv2.down_proj.parameters())
                    if hasattr(block.conv2, 'up_proj'):
                        block.conv2.up_proj.requires_grad_(True)
                        params += list(block.conv2.up_proj.parameters())
                    block.conv2.requires_grad_(True)
                    params += list(block.conv2.parameters())
                    if hasattr(block.conv3,'down_proj'):# conv3
                        block.conv3.down_proj.requires_grad_(True)
                        params += list(block.conv3.down_proj.parameters())
                    if hasattr(block.conv3, 'up_proj'):
                        block.conv3.up_proj.requires_grad_(True)
                        params += list(block.conv3.up_proj.parameters())
                    block.conv3.requires_grad_(True)
                    params += list(block.conv3.parameters())
    if cfg.TEST.ADAPTATION.WHERE == 'adapter':
        if "resnet" in cfg.MODEL.BACKBONE.NAME:
            #if hasattr(model.backbone.bottom_up.stem.conv1, 'parallel_conv'):
            #    model.backbone.bottom_up.stem.conv1.parallel_conv.requires_grad_(True)
            #    params += list(model.backbone.bottom_up.stem.conv1.parallel_conv.parameters())
            for stage in model.backbone.bottom_up.stages:
                for block in stage:
                    if hasattr(block, 'adapter') and block.adapter is not None:
                        block.adapter.requires_grad_(True)
                        params += list(block.adapter.parameters())
                        if hasattr(block.conv1, 'down_proj'):
                            block.conv1.down_proj.requires_grad_(True)
                            block.conv1.up_proj.requires_grad_(True)
                            params += list(block.conv1.down_proj.parameters())
                            params += list(block.conv1.up_proj.parameters())
                        # if hasattr(block.conv1, 'scalar') and cfg.TEST.ADAPTER.SCALAR == 'learnable_scalar':
                        #     block.conv1.scalar.requires_grad_(True)
                        #     params.append(block.conv1.scalar)
                        # if hasattr(block.conv1, 'scale'):
                        #     block.conv1.scale.requires_grad_(True)
                        #     block.conv1.shift.requires_grad_(True)
                        #     params += [block.conv1.scale, block.conv1.shift]
                        # if hasattr(block.conv1, 'lora_A'):
                        #     block.conv1.lora_A.requires_grad_(True)
                        #     block.conv1.lora_B.requires_grad_(True)
                        #     params += [block.conv1.lora_A, block.conv1.lora_B]
                        # if hasattr(block.conv1, 'adapter_norm'):
                        #     block.conv1.adapter_norm.track_running_stats = False
                        #     block.conv1.adapter_norm.requires_grad_(True)
                        #     params += list(block.conv1.adapter_norm.parameters())
            #     if cfg.TEST.ADAPTATION.NORM:
            #         block.conv1.norm.weight.requires_grad = True
            #         block.conv1.norm.bias.requires_grad = True
            #         bn_params += [block.conv1.norm.weight, block.conv1.norm.bias]
            # for m_name, m in model.backbone.bottom_up.named_modules():
            #     if cfg.TEST.ADAPTATION.NORM:
            #         if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            #             m.adapt_training = False
            #             m.weight.requires_grad = True
            #             m.bias.requires_grad = True
            #             if m.weight.requires_grad:
            #                 bn_params += [m.weight, m.bias]
        elif "swin" in cfg.MODEL.BACKBONE.NAME:
            for layer in model.backbone.bottom_up.layers:
                for block in layer.blocks:
                    if hasattr(block, 'adapter') and block.adapter is not None:
                        block.adapter.requires_grad_(True)
                        params += list(block.adapter.parameters())

    elif cfg.TEST.ADAPTATION.WHERE == 'full':
        if cfg.TEST.ADAPTATION.GLOBAL_ALIGN == "BN":
            for m in model.modules():
                if isinstance(m, FrozenBatchNorm2d):
                    # force use of batch stats in train and eval modes
                    m.adapt_training = True
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            return model, None

        for m_name, m in model.backbone.bottom_up.named_modules():
            #if cfg.TEST.ADAPTATION.NORM == 'adapt':
            if True:
                if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.adapt_training = False
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    if m.weight.requires_grad:
                        #bn_params += [m.weight, m.bias]
                        params += [m.weight, m.bias]
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                # except patch embedding
                if "patch_embed" in m_name and "attn" in m_name:
                    continue
                m.weight.requires_grad = True
                params += [m.weight]
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params += [m.bias]
    elif cfg.TEST.ADAPTATION.WHERE == 'normalization':
        for m_name, m in model.backbone.bottom_up.named_modules():
            if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.adapt_training = False
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                if m.weight.requires_grad:
                    params += [m.weight, m.bias]
    elif cfg.TEST.ADAPTATION.WHERE == 'head':
        model.roi_heads.box_head.requires_grad_(True)
        params += list(model.roi_heads.box_head.parameters())

    if cfg.TEST.ADAPTATION.NORM in ["DUA", "NORM"]:
        if "resnet" in cfg.MODEL.BACKBONE.NAME:
            for m_name, m in model.named_modules():
                if isinstance(m, FrozenBatchNorm2d) and 'stem' not in m_name:
                    # force use of batch stats in train and eval modes
                    m.adapt_type = cfg.TEST.ADAPTATION.NORM  # "DUA" or "NORM"
                    # Original DUA Hyperparam
                    if cfg.TEST.ADAPTATION.NORM == "DUA":
                        m.min_momentum_constant = cfg.TEST.ADAPTATION.BN_MIN_MOMENTUM_CONSTANT
                        m.decay_factor = cfg.TEST.ADAPTATION.BN_DECAY_FACTOR
                        m.mom_pre = cfg.TEST.ADAPTATION.BN_MOM_PRE
                    elif cfg.TEST.ADAPTATION.NORM == "NORM":
                        m.source_sum = cfg.TEST.ADAPTATION.BN_SOURCE_NUM
        elif "swin" in cfg.MODEL.BACKBONE.NAME:
            for m_name, m in model.named_modules():
                # 处理Swin Transformer中的LayerNorm层
                if isinstance(m, nn.LayerNorm) and 'patch_embed' not in m_name:
                    # 为LayerNorm层添加适配功能
                    m.adapt_type = cfg.TEST.ADAPTATION.NORM  # "DUA" 或 "NORM"

                    # DUA模式下的超参数
                    if cfg.TEST.ADAPTATION.NORM == "DUA":
                        m.min_momentum_constant = cfg.TEST.ADAPTATION.BN_MIN_MOMENTUM_CONSTANT
                        m.decay_factor = cfg.TEST.ADAPTATION.BN_DECAY_FACTOR
                        m.mom_pre = cfg.TEST.ADAPTATION.BN_MOM_PRE

                    # NORM模式下的源域统计量
                    elif cfg.TEST.ADAPTATION.NORM == "NORM":
                        m.source_sum = cfg.TEST.ADAPTATION.BN_SOURCE_NUM
        if not cfg.TEST.ONLINE_ADAPTATION:
            return model, None, None



    if cfg.SOLVER.TYPE == "SGD":
        sgd_args = [{"params": params}]
        if len(bn_params) > 0:
            sgd_args.append({"params": bn_params, "lr": cfg.SOLVER.BASE_LR_BN}) 
        optimizer = torch.optim.SGD(sgd_args, lr_, momentum=cfg.SOLVER.MOMENTUM,
                                    nesterov=cfg.SOLVER.NESTEROV, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.TYPE == "AdamW":
        adamw_args = {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR,
            "betas": (0.9, 0.999),
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        }
        optimizer = torch.optim.AdamW(**adamw_args)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)

    if cfg.TEST.ADAPTATION.TYPE is not None and "mean-teacher" in cfg.TEST.ADAPTATION.TYPE:
        import copy
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        teacher_model.online_adapt = False
        model.online_adapt = False
        model.training = True
        model.proposal_generator.training = True
        model.roi_heads.training = True
    else:
        teacher_model = None
    
    return model, optimizer, teacher_model

