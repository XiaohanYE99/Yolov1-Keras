darknet19.py 首先下载在Imagenet上预训练的VGG16网络作为主网络，冻结除最后一层卷积层外的所有参数，重新添加全连接层，使用17flowers数据进行fine-turn， 训练模型保存在logs1中。
net_train.py 下载logs1中的模型，冻结参数，重新添加全连接层，使用标注的2flowers数据集进行训练，模型保存logs中。
net_test.py 使用logs2中的模型进行预测。
