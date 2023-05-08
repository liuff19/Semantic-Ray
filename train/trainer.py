import os
import json
import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint

from dataset.train_dataset import RendererDataset
from network.loss import name2loss
from network.renderer import Renderer
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger, reset_learning_rate, MultiGPUWrapper, DummyLoss
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import simple_collate_fn, dummy_collate_fn


class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 1.0e-4,
            "decay_step": 100000,
            "decay_rate": 0.5,
            "lr_min": 1.0e-5,
            "optim_args": {}
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "worker_num": 8,
    }

    def _init_dataset(self):
        self.train_set = RendererDataset(self.cfg['train_dataset_cfg'], True)
        self.train_set = DataLoader(
            self.train_set, 1, True, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        if isinstance(self.cfg['val_set_list'], list):
            for val_set_cfg in self.cfg['val_set_list']:
                name, val_cfg = val_set_cfg['name'], val_set_cfg['cfg']
                val_set = RendererDataset(val_cfg, False)
                val_set = DataLoader(
                    val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
                self.val_set_list.append(val_set)
                self.val_set_names.append(name)
                print(f'{name} val set len {len(val_set)}')
        elif isinstance(self.cfg['val_set_list'], str):
            val_scenes = np.loadtxt(self.cfg['val_set_list'], dtype=str).tolist()
            for name in val_scenes:
                val_cfg = {'val_database_name': name}
                val_set = RendererDataset(val_cfg, False)
                val_set = DataLoader(
                    val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
                self.val_set_list.append(val_set)
                self.val_set_names.append(name)
                print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        self.network = Renderer(self.cfg).cuda()
        if 'fc_layer_only' in self.cfg and self.cfg['fc_layer_only']:
            for param in self.network.parameters():
                param.requires_grad = False
            for param in self.network.agg_net.agg_impl.semantic_fc.parameters():
                param.requires_grad = True
            for param in self.network.agg_net.agg_impl.rgb_fc.parameters():
                param.requires_grad = True
            for param in self.network.agg_net.agg_impl.out_geometry_fc.parameters():
                param.requires_grad = True
        
        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        self.train_network = self.network
        self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'adamw':
            self.optimizer = AdamW
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](
            self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(
            self.optimizer, self.network, self.cfg['lr_cfg']['optim_args'])

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        self.pth_fn = os.path.join(self.model_dir, 'model.pth')
        self.best_pth_fn = os.path.join(self.model_dir, 'model_best.pth')
        self.database_statistics = {}

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        # load model
        if 'load_pretrain' in self.cfg:
            model_path = self.cfg['load_pretrain']
            best_para, start_step = self._load_model(model_path=model_path, load_optimizer=False, load_step=False)
        else:
            best_para, start_step = self._load_model()
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], bar_format='{r_bar}')
        pbar.update(start_step)
        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info = {}
            outputs = self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v

            loss = 0
            for k, v in log_info.items():
                if k.startswith('loss'):
                    loss = loss+torch.mean(v)

            loss.backward()
            if 'max_grad_norm' in self.cfg:
                norm = clip_grad_norm_(self.network.parameters(), self.cfg['max_grad_norm'])
                log_info['grad_norm'] = norm
            self.optimizer.step()
            if ((step+1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step+1, 'train')

            if (step + 1) % self.cfg['val_interval'] == 0 or (step+1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                for vi in range(len(self.val_set_list)):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, self.val_set_list[vi], step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    val_para += val_para_cur
                val_para /= len(self.val_set_list)
                if val_para > best_para:
                    print(
                        f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para = val_para
                    self._save_model(step+1, best_para, self.best_pth_fn)
                self._log_data(val_results, step+1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step+1) % self.cfg['save_interval'] == 0:
                self._save_model(step+1, best_para, self.pth_fn.replace('.pth', f'_step{step+1}.pth'))

            if 'max_grad_norm' in self.cfg:
                pbar.set_postfix(loss=loss.item(), lr=lr, grad_norm=norm.item())
            else:
                pbar.set_postfix(loss=loss.item(), lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()
        
    def eval(self, model_path):
        self._init_dataset()
        self._init_network()
        _, step = self._load_model(model_path, load_optimizer=False)
        val_results = {}
        val_para = 0
        for vi, val_set in enumerate(self.val_set_list):
            val_results_cur, val_para_cur = self.val_evaluator(
                self.network, self.val_losses + self.val_metrics, val_set, step,
                self.model_name, val_set_name=self.val_set_names[vi])
            val_para += val_para_cur
            print('Key metric: ', val_para_cur)
            for k, v in val_results_cur.items():
                v = np.mean(v)
                if k in val_results:
                    val_results[k].append(v)
                else:
                    val_results[k] = [v]
        val_para /= len(self.val_set_list)
        print(val_results)
        print('Key metric (mean): ', val_para)
        with open(os.path.join(self.model_dir, 'val_results.json'), 'w') as f:
            f.write(str(val_results).replace('\'', '\"'))

    def _load_model(self, model_path=None, load_optimizer=True, load_step=True):
        best_para, start_step = 0, 0
        if model_path is not None:
            checkpoint = torch.load(model_path)
        elif os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
        else:
            return 0, 0
        if load_step:
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
        else:
            start_step = 0
            best_para = 0
        self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'==> resuming from step {start_step} best para {best_para}')
        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = float(v)
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
