import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import SimpleCosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import ot
import pandas as pd
from torch import nn
import copy
EPSILON = 1e-8
seed = 1993
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
epochs = 30
lrate = 0.3
milestones = [int(epochs*(1/4)), int(epochs*(2/4))]
lrate_decay = 0.1
batch_size = 512
memory_size = 6400
T = 2



class COIL(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._network = SimpleCosineIncrementalNet("cosine_resnet18", False)
        self._device = args['device']
        self.data_manager=None
        self.nextperiod_initialization=None
        self.sinkhorn_reg=args['sinkhorn']
        self.calibration_term=args['calibration_term']
        self.args=args

    # 自行產出程式碼
    def save_checkpoint(self):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'known_classes':self._known_classes,
            'model': self._network,
            "data_memory": self._data_memory, 
            "targets_memory": self._targets_memory
        }
        # torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_task))
        torch.save(save_dict,"COIL_model.pkl")
        # torch.save(self._network, filename)
    # 自行產出程式碼
    def after_task(self):
        self._known_classes = self._total_classes
    # 自行產出程式碼
    def before_task(self):
        self.nextperiod_initialization=self.solving_ot()
        if self.nextperiod_initialization is None:
            return 'no_new'
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        return 'good'

    # 參考https://github.com/zhoudw-zdw/MM21-Coil 並進行修正
    def solving_ot(self):
        with torch.no_grad():
            if self._total_classes==self.data_manager.get_total_classnum():
                print('training over, no more ot solving')
                return None
            cur_task_size = len(self.df) - self._known_classes
            if cur_task_size <= 0:
                return None
            each_time_class_num=cur_task_size
            self._extract_class_means(self.data_manager,0,self._total_classes)
            former_class_means=torch.tensor(self._ot_prototype_means[:self._total_classes])
            next_period_class_means=torch.tensor(self._ot_prototype_means[self._total_classes:self._total_classes+each_time_class_num])
            Q_cost_matrix=torch.cdist(former_class_means,next_period_class_means,p=self.args['norm_term'])
            #solving ot
            _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.0
            _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.0
            T=ot.sinkhorn(_mu1_vec,_mu2_vec,Q_cost_matrix,self.sinkhorn_reg) 
            T=T.clone().detach().float().cuda()
            transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight, p=2, dim=1))
            oldnorm=(torch.norm(self._network.fc.weight,p=2,dim=1))
            newnorm=(torch.norm(transformed_hat_W*len(former_class_means),p=2,dim=1))
            meannew=torch.mean(newnorm)
            meanold=torch.mean(oldnorm)
            gamma=meanold/meannew
            self.calibration_term=gamma
            self._ot_new_branch=transformed_hat_W*len(former_class_means)*self.calibration_term
        return transformed_hat_W*len(former_class_means)*self.calibration_term

    # 參考https://github.com/zhoudw-zdw/MM21-Coil
    def solving_ot_to_old(self):
        cur_task_size = len(self.df) - self._known_classes
        current_class_num=cur_task_size
        self._extract_class_means_with_memory(self.data_manager,self._known_classes,self._total_classes)
        former_class_means=torch.tensor(self._ot_prototype_means[:self._known_classes])
        next_period_class_means=torch.tensor(self._ot_prototype_means[self._known_classes:self._total_classes])
        Q_cost_matrix=torch.cdist(next_period_class_means,former_class_means,p=self.args['norm_term'])+EPSILON #in case of numerical err
        _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.
        _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.
        T=ot.sinkhorn(_mu2_vec,_mu1_vec,Q_cost_matrix,self.sinkhorn_reg) 
        T = T.clone().detach().float().cuda()
        transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight[-current_class_num:,:], p=2, dim=1))
        return transformed_hat_W*len(former_class_means)*self.calibration_term

    # 參考https://github.com/zhoudw-zdw/MM21-Coil
    def incremental_train(self, data_manager):
        self._cur_task += 1
        cur_task_size = len(self.df) - self._known_classes
        self._total_classes = self._known_classes + cur_task_size
        # print(f'total_classes: {self._total_classes}')
        if self.nextperiod_initialization is not None:
            print(f'self.nextperiod_initialization: {self.nextperiod_initialization.size()}')
        self._network.update_fc(self._total_classes, self.nextperiod_initialization)
        self.data_manager=data_manager

        print('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        self.lamda = self._known_classes / self._total_classes
        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        self._train(self.train_loader)
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)
    # 參考https://github.com/zhoudw-zdw/MM21-Coil
    def _train(self, train_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        # 自行產出程式碼
        optimizer = optim.AdamW(self._network.parameters(), lr=lrate,weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader,  optimizer, scheduler)
    # 參考https://github.com/zhoudw-zdw/MM21-Coil
    def _update_representation(self, train_loader,  optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            weight_ot_init=max(1.-(epoch/2)**2,0)
            weight_ot_co_tuning=(epoch/epochs)**2.
        
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output=self._network(inputs)
                logits = output['logits']
                onehots = target2onehot(targets, self._total_classes)

                clf_loss = F.cross_entropy(logits, targets)
                if self._old_network is not None:
                    
                    old_logits = self._old_network(inputs)['logits'].detach()
                    hat_pai_k = F.softmax(old_logits / T, dim=1)
                    log_pai_k = F.log_softmax(logits[:, :self._known_classes] / T, dim=1)
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    
                    if epoch<1:
                        features=F.normalize(output['features'], p=2, dim=1)
                        current_logit_new=F.log_softmax(logits[:, self._known_classes:] / T, dim=1)
                        new_logit_by_wnew_init_by_ot = F.linear(features, F.normalize(self._ot_new_branch, p=2, dim=1))
                        new_logit_by_wnew_init_by_ot = F.softmax(new_logit_by_wnew_init_by_ot / T, dim=1)
                        new_branch_distill_loss = -torch.mean(torch.sum(current_logit_new * new_logit_by_wnew_init_by_ot, dim=1))
                        
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) +\
                        0.001*(weight_ot_init*new_branch_distill_loss)
                    else:
                        features=F.normalize(output['features'], p=2, dim=1)
                        if i%30==0:
                            with torch.no_grad():
                                self._ot_old_branch=self.solving_ot_to_old()
                        old_logit_by_wold_init_by_ot = F.linear(features, F.normalize(self._ot_old_branch, p=2, dim=1))
                        old_logit_by_wold_init_by_ot = F.log_softmax(old_logit_by_wold_init_by_ot / T, dim=1)
                        old_branch_distill_loss = -torch.mean(torch.sum(hat_pai_k* old_logit_by_wold_init_by_ot, dim=1))
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) +\
                        self.args['reg_term']*(weight_ot_co_tuning*old_branch_distill_loss)
                else:
                    loss = clf_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            prog_bar.set_description(info)

        print(info)


