##基于视觉的钢琴转录算法实现##

###Description
该工程是基于视觉的钢琴转录算法，相比与之前的方法更加稳定，精度更高，可以处理各个视角下的钢琴转录，主要包括几个模块，键盘分割模块，按键定位模块，手的分割定位模块，钢琴按键按下分类模块等，包含的技术有语义分割，图像分类和传统的视觉算法，在公开数据集上white fscore=0.94,black fscore=0.98。

### Requirement
1. python-opencv
2. pytorch>=1.1.0
3. easydict

### module 
*	键盘分割采用的pspnet,在3rdparty/segmentaion下
*	钢琴按键分类在3rdparty/key_classification
*	其他的转录代码有main.py执行

### run
```
	python main.py --img_dir video_file/img_path
```


