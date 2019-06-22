import requests
import os
import urllib
import json
#定义下载图片的函数
def downImg(imgUrl, dirPath, imgName):
    filename = os.path.join(dirPath, imgName)
    try:
        res = requests.get(imgUrl, timeout=15)
        if str(res.status_code)[0] == "4":
            print(str(res.status_code), ":", imgUrl)
            return False
    except Exception as e:
        print("抛出异常：", imgUrl)
        print(e)
        return False
    with open(filename, "wb") as f:
        f.write(res.content)
    return True

words = [["梵高作品",'FG'],['莫奈作品','MN'],['毕加索作品','BJS'],['达芬奇作品','DFQ']] #搜索关键字，如 ：梵高作品
trainPath = "train_data/"
#如果文件夹不存在，创建文件夹
if not os.path.exists(trainPath):
    os.mkdir(trainPath)
for word in words:
    dirPath = trainPath + word[1]
    # 如果文件夹不存在，创建文件夹
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    word = urllib.parse.quote(word[0]) #因为是中文，所以要进行urlencode转换
    pn = 30  #当前页的图片数量偏移量，如 60 表示当前页是第二页，图片数的偏移是60
    rn = 30  #每每页返回多少图片，如 30 表示每页三十张图片
    i = 1 #图片编号
    while pn <= 30 * 20: #获取20页的图片，总共600张，建议修改页数，爬去更多一点的图片
        try:
            url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=' + word + '&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=' + word + '=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=' + str(
                pn) + '&rn=' + str(rn) + '&gsm=3c&1550715038298='
            jsonBytes = requests.get(url, timeout=10).content  # 获取json数据-字节
            jsonData = jsonBytes.decode('utf-8')  # json数据-字节转字符串
            print("---------------------------------------------------------")
            jsonData = jsonData.replace("\\'", '') #不加这个字符串替换json.loads时会报错，意思是去掉字符串中的\'
            print(jsonData)
            print("---------------------------------------------------------")
            jsonObj = json.loads(jsonData)  # json数据-字符串转对象
            if 'data' in jsonObj:
                for item in jsonObj['data']:
                    if 'thumbURL' in item:
                        imgName = str(i) + ".jpg"
                        downImg(item['thumbURL'], dirPath, imgName)  # 下载图片
                        print(item['thumbURL'])
                        i += 1
            pn += rn  # 下一页
        except Exception as e:
            print(e)
