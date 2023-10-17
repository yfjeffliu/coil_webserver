import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import  ITRI_GOODS
seed = 1993
# random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class DataManager(object):
    def __init__(self,   seed, init_cls, increment, longtail=0):
        self.dataset_name = "itri_goods"
        self._setup_data(self.dataset_name, 0, seed)
        # assert init_cls <= len(self._class_order), 'No enough classes.'
        # self._increments = [init_cls,init_cls]
        
        # while sum(self._increments) + increment <= len(self._class_order):
        #     self._increments.append(increment)
        
        # offset = len(self._class_order) - sum(self._increments)
        # if offset > 0:
        #     self._increments.append(offset)
        self.longtail = longtail
        
        self.longtaillist = get_img_num_per_cls(1300, 100, 'exp', 0.1)

    # @property
    # def nb_tasks(self):
    #     return len(self._increments)

    # def get_task_size(self, task):
    #     try:
    #         a = self._increments[task]
    #     except:
    #         a = 1
    #     return a

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose(
                [*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])

        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx+1)
            
            if self.longtail == 1 and mode == 'train':
                data.append(class_data[:self.longtaillist[idx]])
                targets.append(class_targets[:self.longtaillist[idx]])
            else:
                data.append(class_data)
                targets.append(class_targets)
        
        # return
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(np.array(appendent_data).flatten())
            targets.append(np.array(appendent_targets).flatten())
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)



    def _setup_data(self, dataset_name, shuffle, seed):
        idata = ITRI_GOODS()
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        # order = [i for i in range(50)]
        order = [i for i in range(len(np.unique(self._train_targets)))]
        
        order = idata.class_order
        self._class_order = order
        # logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order)
        

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    
    return ITRI_GOODS()
    


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_img_num_per_cls(img_max, cls_num, imb_type, imb_factor):
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls
