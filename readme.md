# 文本图片分类

功能比较特殊，仅用来判断图片中是否存在文本，不返回bounding box. 使用的是opencv 的EAST模型。 
权重链接：https://pan.baidu.com/s/1iei8J5YMWfNKqKdSNFSwFg 
提取码：u4ho

## 依赖

* opencv_python >=4.1
* numpy>=1.15
* python>=3.6

## 使用方法

```sh
python text_classify.py --input [xxx]

[xxx]填写文件名

