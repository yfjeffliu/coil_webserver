import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

# random.seed(seed)
np.random.seed(1993)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



class ITRI_GOODS(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        dir_train = './data/'
        train_dset = datasets.ImageFolder(dir_train)
        print("===================")
        print(f'Training Size: {len(train_dset)}')
        
        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
