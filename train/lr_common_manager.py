class LearningRateManager():
    def __init__(self, cfg):
        if 'scale_new' in cfg:
            self.scale_new = cfg['scale_new']
        else:
            self.scale_new = None

    def set_lr_for_all(self, optimizer, lr):
        if self.scale_new is not None:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * self.scale_new
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network, optim_args):
        if 'scale_new' is not None:
            # may specify different lr for different parts
            # use group to set learning rate
            old = []
            new = []
            for name, param in network.named_parameters():
                if '_new' in name:
                    new.append(param)
                else:
                    old.append(param)
            return optimizer([{'params': old}, {'params': new}], lr=1e-3, **optim_args)
        else:
            return optimizer(network.parameters(), lr=1e-3, **optim_args)

    def __call__(self, optimizer, step, *args, **kwargs):
        pass


class ExpDecayLR(LearningRateManager):
    def __init__(self, cfg):
        super(ExpDecayLR, self).__init__(cfg)
        self.lr_init = cfg['lr_init']
        self.decay_step = cfg['decay_step']
        self.decay_rate = cfg['decay_rate']
        self.lr_min = cfg['lr_min']

    def __call__(self, optimizer, step, *args, **kwargs):
        lr = max(self.lr_init*(self.decay_rate **
                 (step//self.decay_step)), self.lr_min)
        self.set_lr_for_all(optimizer, lr)
        return lr


class ExpDecayLRRayFeats(ExpDecayLR):
    def construct_optimizer(self, optimizer, network):
        paras = network.parameters()
        return optimizer([para for para in paras] + network.ray_feats, lr=1e-3)


class WarmUpExpDecayLR(LearningRateManager):
    def __init__(self, cfg):
        super(WarmUpExpDecayLR, self).__init__(cfg)
        self.lr_warm = cfg['lr_warm']
        self.warm_step = cfg['warm_step']
        self.lr_init = cfg['lr_init']
        self.decay_step = cfg['decay_step']
        self.decay_rate = cfg['decay_rate']
        self.lr_min = 1e-5

    def __call__(self, optimizer, step, *args, **kwargs):
        if step < self.warm_step:
            lr = self.lr_warm
        else:
            lr = max(self.lr_init*(self.decay_rate **
                     ((step-self.warm_step)//self.decay_step)), self.lr_min)
        self.set_lr_for_all(optimizer, lr)
        return lr


name2lr_manager = {
    'exp_decay': ExpDecayLR,
    'exp_decay_ray_feats': ExpDecayLRRayFeats,
    'warm_up_exp_decay': WarmUpExpDecayLR,
}
