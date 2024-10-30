from __future__ import division

import argparse
import cv2
import torch
import sklearn
import mmcv
import copy
import os
import time
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

warnings.filterwarnings("ignore")

from mmcv.utils import TORCH_VERSION, digit_version


def parse_args():
    """
    用于解析命令行参数
    """
    '''
    add_argument(key, type, default, help):
    parser.add_argument('--example', type=int, default=5, help='an example parameter')
    --example为key，期望一个整数值，默认值是5。help 提供了该参数的描述。
    action: 这个参数表示当这个参数在命令行中出现时采取的动作
    action = 'store' 表示保存参数值，这是默认的行为
    action = 'store_const' 表示保存一个被定义为参数的常量值
    action = 'store_true` 和 action = 'store_false' 用于保存True和False值
    action = DictAction 用于保存字典类型的参数
    '''
    parser = argparse.ArgumentParser(description='Train a detector')     # 创建一个ArgumentParser对象，description参数是一个描述这个参数解析器的字符串
    parser.add_argument('config', help='train config file path')         # 训练配置文件的路径
    parser.add_argument('--work-dir', help='the dir to save logs and models') # 保存日志和模型的目录
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')  # 从哪个检查点文件恢复
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training') # 训练过程中是否进行验证
    group_gpus = parser.add_mutually_exclusive_group()                 # 创建一个互斥组，只能选择其中一个参数
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')  # 使用的gpu数量
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)') # 使用的gpu的id
    parser.add_argument('--seed', type=int, default=0, help='random seed') # 随机种子
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.') # 是否设置CUDNN后端的确定性选项
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')      # 用于覆盖配置文件中的一些设置
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')     # 覆盖配置文件中的一些设置
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')       # 使用的启动器
    parser.add_argument('--local_rank', type=int, default=0)  # 本地排名，用于分布式训练时指定本地进程的排名
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')  # 根据gpu数量自动调整学习率
    #* 解析命令行参数并返回一个包含参数值的 Namespace 对象。
    args = parser.parse_args()  

    # 如果没有设置环境变量LOCAL_RANK，则设置为args.local_rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 如果--options和--cfg-options都被指定，则抛出异常
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args() #* 解析命令行参数

    cfg = Config.fromfile(args.config) #* 解析配置文件

    #* 一下进行命令行参数和配置文件的整合
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # *从字符串列表中导入模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # *从字符串列表中导入模块，更新Register
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # 没有指定plugin_dir，使用config所在目录
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            #* 从插件中导入custom_train_model
            from projects.mmdet3d_plugin.uniad.apis.train import custom_train_model 
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 根据优先级设置工作目录：命令行参数 > 配置文件 > 默认值
    #* 这里work_dir在.sh脚本里面
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 如果提供了恢复检查点的路径，则更新配置文件
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # 修复AdamW的bug
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' 
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    #* 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    #* 保存合并更新后的配置文件
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # 初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    #* logger保存在
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # 初始化 meta 字典，用于记录一些重要信息
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()]) 
    dash_line = '-' * 60 + '\n'                                            # 输出格式调整
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)    # 记录环境信息
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text          # 环境信息和配置内容存储在元信息字典中

    # 基础信息记录
    logger.info(f'Distributed training: {distributed}')  # 是否分布式训练
    logger.info(f'Config:\n{cfg.pretty_text}')           # 配置文件内容

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    ##############################################################################
    #* 按照CFG构建模型并初始化权重
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    ###############################################################################
    # 输出模型信息
    logger.info(f'Model:\n{model}')
    
    #* 构建训练数据集，并将其添加到 datasets 列表中。
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:       # # 如果工作流包含两个阶段（train，val），则深拷贝验证数据集配置。
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:         
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False       # 设置验证数据集的 test_mode 为 False，这样不会影响后续的 AP/AR 计算。
        datasets.append(build_dataset(val_dataset)) #* 构建验证数据集，并将其添加到 datasets 列表中。
    
    # 如果配置文件中包含 checkpoint_config，则将一些元数据信息（如版本信息、配置文件内容、类别和调色板）保存到检查点配置中。
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # 为模型添加 CLASSES 属性，方便后续的可视化操作。
    model.CLASSES = datasets[0].CLASSES
    ###############################################################################
    # *正式开始训练模型
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
