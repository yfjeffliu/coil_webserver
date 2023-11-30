# 自行撰寫之程式碼
from flask import Flask, request
from flask import send_file
from flask import Flask, render_template
from flask import Flask, jsonify
import base64
import random
import os
import re
import csv
from datetime import datetime
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import torch
import time
from main import main, check_new
from count_down import countdown
import multiprocessing
headdir = './'
app = Flask(__name__)
# target_folder=headdir


@app.route('/getcsv', methods=['GET'])
def get_csv():
    #global target_folder
    filename = './'+'goods50_show.csv'
    data_list = []
    with open(filename,encoding='cp950') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_list.append(row)
    return jsonify(data_list)
    
## 接收圖片並辨識後回傳詳細資料
@app.route('/exa', methods=['POST'])
def examine():


    ## 防錯，預防無圖片上傳
    if 'image' not in request.files:
        return 'No image part', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    start = time.time()
    img = Image.open(file)
    ## 圖片轉換
    transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
    )
    img = transform1(img).to('cuda')
    ## 模型載入
    if not os.path.exists('./COIL_model.pkl'):
        return "尚未訓練模型"
    data = torch.load(open('./COIL_model.pkl',"rb"))
    model = data["model"].to('cuda')
    with torch.no_grad():
        outputs = model(img.unsqueeze(0))['logits']
    predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
    end_time = time.time()
    ## 計算花費時間
    print("花費時間:",end_time-start)
    ## 讀取csv檔
    df = pd.read_csv("./goods50.csv",encoding="cp950")
    ## 取得詳細結果並回傳
    predict_class = df.iloc[int(predicts)]
    return f"辨識結果: {predict_class['goods_name']}\n 重量: {predict_class['weight']} g \n 保存方式: {predict_class['tag']}"



## 開始訓練模型
@app.route('/start', methods=['GET'])
def outGet():
    ## 根據CSV goods50.csv 檔案中的gid數量，判斷是否有新資料
    new = check_new()
    if new:
        ## 檢查是否有模型訓練中
        path = 'check_training.txt'
        f = open(path, 'r')
        status = f.read()
        f.close()
        if status=='0':
            ## 開始訓練模型
            process = multiprocessing.Process(target=main)
            process.start()
            ## 開始倒數計時
            process2 = multiprocessing.Process(target=countdown)
            process2.start()
            return '模型開始訓練'
        else:
            return f'已有模型訓練中，大約還需要 {status} 分鐘'
    else:
        return '無新資料，無需訓練'
## 接收類別gid
@app.route('/text', methods=['POST'])
def upload_file():
    data = request.form.get('輸入類別')
    # 確保資料不為空
    if data:
        good_dict['gid'] = data
        return '數據已成功保存為文件。'
    else:
        return '未接收到有效數據。'
## 接收類別名稱
@app.route('/name', methods=['POST'])
def rname():
    input_name = request.form.get('輸入名稱')
    if input_name:
        good_dict['goods_name'] = input_name
        return "Data received"
    else:
        return "No data received.", 400    

## 接收重量
@app.route('/weight', methods=['POST'])
def rweight():
    t = datetime.now()
    input_name = request.form.get('輸入weight')
    if input_name:
        good_dict['weight'] = input_name
        return "Data received"
    else:
        return "No data received.", 400  
        
## 接收保存方式
@app.route('/tag', methods=['POST'])
def rtag():
    input_name = request.form.get('輸入tag')
    if input_name:
        good_dict['tag'] = input_name
        return "Data received"
    else:
        return "No data received.", 400  

## 預先設定字典資料
good_dict = {
    'gid':None,
    'goods_name':None,
    'weight':None,
    'tag':None,
}


## 接收圖片並儲存
@app.route('/upup', methods=['POST'])
def upload_delphi():
    t = datetime.now()
    z="%.3d" % (random.randint(1,999))
    d1="upimg"+t.strftime("%Y%m%d%H%M%S")+z+".jpg"
    df = pd.read_csv("./goods50.csv",encoding="cp950")
    ## 若gid不在csv檔中，則新增一筆資料，並設定圖片資料夾名稱
    if int(good_dict['gid']) not in list(df['gid']): 
        df.loc[len(df)] = good_dict
        img_path =  '{:03}'.format((len(df)-1)) +  '{:06}'.format(int(good_dict['gid']))
    else:
    ## 若gid在csv檔中，則取得該gid，並放入圖片資料夾
        img_path = '{:03}'.format(int(df[df['gid']==int(good_dict['gid'])].index.tolist()[0])) + '{:06}'.format(int(good_dict['gid']))
    if os.path.exists('./data/'+img_path):
        pass
    else:
        os.mkdir('./data/'+img_path)
    # 儲存csv檔
    df.to_csv("./goods50.csv",encoding="cp950",index=False)
    
    file = request.files['image']

    file.save('./data/'+img_path+'/'+d1)

    return '顯示辨識結果' 

@app.route('/preupload', methods=['GET'])
def pre_upload():
    return 'OK', 200
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
