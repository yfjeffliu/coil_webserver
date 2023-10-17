import sys
import logging
import copy
import torch
from utils.data_manager import DataManager
from models.COIL import COIL
import pandas as pd
from utils.toolkit import count_parameters

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    args['seed'] = seed_list
    args['device'] = device
    _train(args)


def _train(args):
    longtail='Longtail' if args['longtail']==1 else 'Normal'
    logfilename = '{}_{}_{}_{}_{}_{}_{}'.format(args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'],longtail)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    _set_device(args)
    data_manager = DataManager( args['seed'], args['init_cls'], args['increment'],args['longtail'])
    
    model = COIL(args)
    ## 若有訓練過模型，則將模型載入
    if args["check_point"] is not None:
        data = torch.load(open(args["check_point"],"rb"))
        model._network = data["model"]
        model._known_classes = data['known_classes']
        model._total_classes = model._known_classes
        model._cur_task = data["tasks"]
        model._data_memory = data ["data_memory"], 
        model._targets_memory = data["targets_memory"]
    model.data_manager = data_manager
    ## 讀取csv檔
    model.df = pd.read_csv('goods50.csv',encoding='cp950')
    if args["check_point"] is not None:
        isnew = model.before_task()
        if isnew == 'no_new':
            return
    model.incremental_train(data_manager)

    model.after_task()

    model.save_checkpoint()
    ## 訓練完成後，將新訓練的類別資料存入用來顯示的csv檔
    model.df.to_csv('goods50_show.csv',encoding='cp950',index=False)

def _set_device(args):
    device = torch.device('cuda')
    args['device'] = device
