from ultralytics import YOLO
import cv2


# 為了顯示中文而載入 PIL 相關函式庫
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 顯示中文字的自訂函數
def cfont(oimg,msg):
    fontpath = 'NotoSansTC-Regular.ttf'          # 設定字型路徑
    font = ImageFont.truetype(fontpath, 50)      # 設定字型與文字大小
    imgPil = Image.fromarray(oimg)                # 將 img 轉換成 PIL 影像
    draw = ImageDraw.Draw(imgPil)                # 準備開始畫畫
    draw.text((20, 50), msg, fill=(255, 255, 0), font=font)  # 畫入文字，\n 表示換行
    return np.array(imgPil)

# 判斷是否為順子的自訂函數
def sj(ca_num):
    mycard=[]
    for i in range(13):
        # 如果串列內容為1，將索引值新增到mycard
        if ca_num[i]==1:
            mycard.append(i)
    # 如果最大值-最小值是4或者牌組為10、J、Q、K、A
    if (max(mycard)-min(mycard)==4) or mycard==[0,9,10,11,12]:
        return True
    else:
        return False

# 載入模型
model = YOLO("poker_detect.pt")
# 52張牌類別
classNames =   ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']

# 檔案名稱
file='demo05.jpg'
img=cv2.imread(file)


# 調整照片尺寸
h,w=img.shape[0],img.shape[1]
scale=w/800
img = cv2.resize(img, (800,int(h/scale)))


# 將畫面送給模型推論，並回傳結果
results = model(img,iou=0.2,conf=0.5) 

# 用來記錄偵測到的牌類別索引
card=[]
# c_type紀錄牌的花色
c_type={'C':0,'D':0,'H':0,'S':0}
# c_num紀錄牌的數字，串列長度13，每個數字對應一格，串列內容0代表張數0
c_num=[0]*13
# 牌型
msg=''

# 以迴圈處理回傳結果(其實通常只有一個results)
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 將偵測到的類別索引增加到card串列
        card.append(int(box.cls[0]))

    # 因為一張牌有兩個偵測位置，利用set集合去除重複
    all_card=set(card)
    # 將得到的牌型轉成串列
    mycard=list(all_card)

    # 如果mycard張數是5張
    if len(mycard)==5:
        # 將牌一張張提出
        for i in mycard:
            # 以索引值從classNames取得牌的資訊，如10C
            card_name=classNames[i]
            #print(card_name)
                    
            # 從前面取到倒數第二個字，cn代表數字
            cn=card_name[:-1]
            # 將數字新增到c_num，代表某個數字有幾張牌
            if cn=='J':
                c_num[10]+=1
            elif cn=='Q':
                c_num[11]+=1
            elif cn=='K':
                c_num[12]+=1
            elif cn=='A':
                c_num[0]+=1
            else:
                c_num[int(cn)-1]+=1
                    
            # 取最後一個字，ct代表花色
            ct=card_name[-1]
            # 依花色計數
            c_type[ct]+=1
            
            #print(c_type)
            #print(c_num)
            
    #判斷牌型
    if max(c_num)==4:
        msg='鐵支'
    elif 3 in c_num and 2 in c_num:
        msg='葫蘆'
    elif 3 in c_num:
        msg='三條'
    elif 2 in c_num:
        if c_num.count(2)==2:
            msg='兔呸'
        else:
            msg="一對"
    elif 5 in c_type.values():
        if sj(c_num)==True:
            msg='同花順'
        else:
            msg='同花'
    elif sj(c_num)==True:
            msg='順子'
    else:
        msg='散牌'

    # 印出牌型
    print(msg)        
        
    # 回傳推論影像給image
    image=r.plot()
    # 將影像和牌型傳給cfont中文字函數
    image=cfont(image,msg)

    cv2.imshow('Result', image)

cv2.waitKey(0)
cv2.destroyAllWindows()