#resquests 用于网络请求
#urllib
#引入os，操作系统目录
import requests,urllib,json
import os

def get_data():
    i = 0
    classify = ['airplane','automobile','bird','cat',
                'dog','deer','frog','horse','ship','truck']
    urls = []
    for i in range(10):
        for j in range(10):
            # http: // www.cs.toronto.edu / ~kriz / cifar - 10 - python.tar.gz
            url = "http://www.cs.toronto.edu/~kriz/cifar-10-samples/{}{}.png".format(classify[i],j+1)
            urls.append(url)
    return urls

def download_images(urls):
    for i,url in enumerate(urls):
        image_name = url.split('/')[-1]
        print("No.{} images is downloading".format(i))
        urllib.request.urlretrieve(url,"images/" + image_name)

if __name__ == "__main__":
    download_images(get_data())