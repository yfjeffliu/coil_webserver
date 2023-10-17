from trainer import train
import pandas as pd
import os
import torch
def main():
    ## 設定參數
    args = {}
    args["convnet_type"] = "cosine_resnet18"
    args["dataset"] = "itri_goods"
    args["shuffle"] = 0
    args["init_cls"] = 20
    args["increment"] = 1
    args["model_name"] = "test_ws"
    args["longtail"] = 0
    args["sinkhorn"] = 0.464
    args["seed"] = 1993
    args["calibration_term"] = 1.5
    args["norm_term"] = 3.
    args["reg_term"] = 1e-3
    args["device"] = '0'
    ## 若有訓練過模型，則將模型載入
    if (os.path.exists("COIL_model.pkl")):
        args["check_point"] = "COIL_model.pkl"
    else:
        args["check_point"] = None

    ## 訓練模型 開始時設定check_training.txt為1，結束時設定為0
    path = 'check_training.txt'
    f = open(path, 'w')
    f.write('1')
    f.close()
    train(args)
    f = open(path, 'w')
    f.write('0')
    f.close()
def check_new():
    ## 檢查是否有新資料
    if (os.path.exists("COIL_model.pkl")):
        record = "COIL_model.pkl"
        data = torch.load(open(record,"rb"))
        ## 計算新資料數量
        df = pd.read_csv('goods50.csv',encoding='cp950')
        new_size = len(df) - data['known_classes']
        if new_size > 0:
            ## 有新資料
            return True
        else:
            ## 無新資料
            return False
    else:
        record = None
        ## 檢查是否有資料
        df = pd.read_csv('goods50.csv',encoding='cp950')
        if len(df) ==0:
            ## 無資料
            return False
        else:
            ## 有資料
            return True

if __name__ == '__main__':
    main()
