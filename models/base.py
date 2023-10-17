import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
EPSILON = 1e-8
batch_size = 512
seed = 1993
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class BaseLearner(object):
    def __init__(self):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

    def save_checkpoint(self, filename):
        pass
    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
                                                   decimals=2)

        return ret



    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        
        dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.to(self._device)
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            
            _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)
    
    
    def _extract_class_means(self,data_manager,low,high): #Y
        self._ot_prototype_means = np.zeros((data_manager.get_total_classnum(), self._network.feature_dim))
        with torch.no_grad():
            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / (np.linalg.norm(class_mean))
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()

    def _extract_class_means_with_memory(self,data_manager,low,high): #Y
        
        self._ot_prototype_means = np.zeros((data_manager.get_total_classnum(), self._network.feature_dim))
        memoryx,memoryy=self._data_memory,self._targets_memory
        memoryx = np.array(self._data_memory).flatten()
        memoryy= np.array(self._targets_memory).flatten()
        with torch.no_grad():
            for class_idx in range(0,low):
                idxes = np.where(np.logical_and(memoryy >= class_idx, memoryy < class_idx+1))[0]
                
                data, targets = memoryx[idxes],memoryy[idxes]
                _,_,idx_dataset=data_manager.get_dataset([], source='train', appendent=(data,targets) ,mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean

            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()
        
    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        dummy_data = np.array(dummy_data).flatten()
        dummy_targets= np.array(dummy_targets).flatten()
        self._class_means = np.zeros((self._total_classes, self._network.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            mm = m
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            if mm > len(idx_loader.dataset):
                mm = len(idx_loader.dataset)
            selected_exemplars = []
            exemplar_vectors = []  
            for k in range(1, mm+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(mm, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

