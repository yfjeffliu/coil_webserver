# 自行撰寫之程式碼
import time
def countdown():
    path = 'check_training.txt'
    for min in range(90,0,-1):
        f = open(path, 'r')
        status = f.read()
        if status=='0':
            f.close()
            return
        f.close()
        f = open(path, 'w')
        f.write(str(min))
        f.close()
        time.sleep(60)
    
