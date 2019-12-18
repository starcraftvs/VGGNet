import torch
import torchvision
import time
import os
#用torchvison里的Compose类建立几个实例，这几个实例代表了几个不同的图片转换方式,通过对图像进行一定的处理从而达到数据增强的效果
#变换一：
transform=torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),#随机翻转图片
        torchvision.transforms.RandomGrayscale(),#随机调整图片整体greyscale（即图片亮度）
        torchvision.transforms.ToTensor(),#默认图片格式为numpy，需转换为Tensor（numpy代表的是每个维度的值为（0-255），转换为tensor后变为（0-1））
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #将图片归一化 （前一个矩阵代表的是mean值，后一个矩阵代表的是std值）
        #变化公式为image=(image-mean)/std，即将image的范围变为了（-1到1）
#归一化变换：
transform1=transform.Compose(
    [
        torchvision.transforms.Totensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)#从CIFAR10(pytorch提供的一个数据库）里下载训练数据，记为trainset
trainloader=torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=True)#torch.utils.data主要有三类，.Dataset,.Dataloader,.sampler.Sampler
#（注释该类里面的属性，trainset:训练集的数据接口，必须保证为torch.utils.data.Dataset类或其他函数里它的子类，batch_size：每次送入训练的样本数量，shuffle：
#每个epoch里会不会打乱数据集，默认为False，用于加载数据的子进程数，默认为0，表示在主进程里运行（一般别改）
#因为python如果要用到并行计算多进程必须在主程序中，需要if name == ‘main’:来运行主程序，具体可参见知乎的一片文章https://zhuanlan.zhihu.com/p/39542342，
#collate_fn:可调用的一个function，用于合并样本列表形成mini_batch，pin_memory:默认为False，如果为True,将会把张量复制到CUDA固定内存中，然后返回它，drop_last:默认为False，如果改为True，将丢掉
#最后一部分不足以构成batch_size的多余素材）
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform1)#同上，获得测试集，只是train改为False了
testloader=torch.utils.data.DataLoader(test,batchsize=2,shuffle=True)#同上，加载测试集
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck') #定义标签

#此处采用VGGNet的改版作为神经网络
#定义神经网络
class Net(torch.nn.Moudle):
    def __init__(self):
        #nn.Module是一个特殊的类，它的子类必须在构造函数中执行父类的构造函数（具体原因要以后看）
        super(Net,self).__init__()#等价于nn.Module.__init__(self)，但推荐使用第一种写法
        #接下里定义神经网络，使用函数torch.nn.Conv2d(此时输入的都是规范的32*32*3的图片)    
        self.conv1=torch.nn.Conv2d(3,64,3,padding=1)      
        #torch.nn.Conv2d(in_channels:输入特征矩阵的通道数Cin，out_channels:输出特征矩阵的通道数Cout,kernal_size：卷积核的大小，padding:边缘的扩充，默认
        #值为0，代表使用0进行扩充，dilation:内核间的距离，默认为1，groups：组数，默认为1，bias:要不要加偏差，默认为True
        self.conv2=torch.nn.Conv2d(64,64,3,padding=1)
        self.pool1=torch.nn.MaxPool2d(2,2)#池化层的矩阵大小
        #torch.nn.MaxPool2d(kernal_size:窗口大小（可以是int或者tuple）,stride:(int/tuple)，步长，默认为kernal_size，padding:输入的每一条边补充0的层数？
        #真的？，dilation:内核间的距离，也有叫控制步幅的参数，return_indices：如果True，会返回输出最大值的序号，默认False，ceil_mode:True的话计算输出信号
        #大小的时候向上取整，默认为False，向下取整)
        self.bn1=torch.nn.BatchNorm2d(64) #用于标准化输出参数，比较复杂，以后慢慢看，目前姑且认为输入为feature_map的数量
        self.relu1=nn.ReLU()#激活函数为ReLU函数，具体参数以后看
        #往下按照VGG-16Net继续往下搭
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU()

        self.conv5 = torch.nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = torch.nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = torch.nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = torch.nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()

        self.conv8 = torch.nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = torch.nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()

        self.conv11 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = torch.nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = torch.nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.relu5 = torch.nn.ReLU()

        #对得出的矩阵做线性变换 明天再细看
        self.fc14=torch.nn.Linear(512*4*4,1024)#真的吗？那图片应该不是32*32的把？
        self.drop1=torch.nn.Dropout2D()#效果上来讲，使得得到的矩阵一部分行随机为0。另一个角度，即使得部分特征随机被消除，
        #也即抑制了部分神经节点，使得结果不容易过拟合
        self.fcl5=torch.nn.Linear(1024,1024)#再次线性转换
        self.drop2=torch.nn.Dropout2D()#同上
        self.fc16==torch.nn.Linear(1024,10)#因为结果有10分类，最后得到的是对于10种分类的不同概率，下面取最大概率即为预测结果
    def forward(self,x):#定义前向传播，即把上面定义的神经网络全部跑一遍
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

    def train_sgd(self,device):#定义训练
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) #定义优化器：种类为Adam，学习率lr=0.0001
        #介绍optim.Adam:
        #params (iterable) – 待优化参数的iterable或者是定义了参数组的dict(j即可以为待优化的模型参数或自己定义的dict)
	    #lr (float, 可选) – 学习率（默认：1e-3）(控制权重的更新比率)
	    #betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）？？
	    #eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）？？
	    #weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0） ？？

        path = 'weights.tar'   #用一个变量规定模型参数的存储文件名及路径
        initepoch = 0           #初始化epoch参数，从0开始

        if os.path.exists(path) is not True:   #若没有已保存的现有模型参数，则新建
            loss = torch.nn.CrossEntropyLoss()       #新建loss参数，选择为交叉熵类
            # optimizer = torch.optim.SGD(self.parameters(),lr=0.01)

        else:
            checkpoint = torch.load(path)      #若已有，则导入模型，存在checkpoint变量里，该变量为一个dict
            self.load_state_dict(checkpoint['model_state_dict'])     #load模型结构参数
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    #load模型状态参数
            initepoch = checkpoint['epoch']        #load现在已经运行了多少个epoch
            loss = checkpoint['loss']              #load现在的loss（看上面，是一个类，而不是一个数值）




        for epoch in range(initepoch,100):  #跑100次epoches
            timestart = time.time()#记录时间

            running_loss = 0.0 #用于记录running_loss的数值，用以判断模型精度
            total = 0          #记录统计数据
            correct = 0        #记录统计数据
            for i, data in enumerate(trainloader, 0): #跑一遍epoch，跑i次batches
                #得到输入的数据
                inputs, labels = data   #data是从trainloader中返回的一个btach，包括输入以及标签
                inputs, labels = inputs.to(device),labels.to(device) #用.to()函数，将结果输入device（device是自己定义的
                #可以看出这个函数是要从外部输入device参数的）

                #每一次batch要将梯度清0，否则每一个batch的数据不一样，上一个batch的梯度对这个batch是没有意义的
                torch.optimizer.zero_grad()

                # 前向传播，反向传播以及优化
                outputs = self(inputs)  #调用类自身就相当于函数，可以跑出outputs?具体需要看nn.Module
                l = loss(outputs, labels)#用定义为交叉熵类的loss来计算loss的具体值，根据输出和自己定义的标签来确定差值
                #自带反向传播函数
                l.backward()
                #用optimzer.step()后模型会更新，每个mini_batch之后最好都要调用，不然这个batch就是白跑了
                torch.optimizer.step()

                # 输出统计数据
                running_loss += l.item() #计算统计数据running_loss，其为500个batches后的loss和，下面利用其计算均值
                # print("i ",i)
                if i % 500 == 499:  # print every 500 mini-batches #当跑了500次以后
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i, running_loss / 500))   #打印出500次均值的值
                    running_loss = 0.0                      #归0到下一个循环，继续累计
                    _, predicted = torch.max(outputs.data, 1)#此神经网络输出每个图片对属于10种分类分别的概率，取最大值
                    #即为预测结果，torch.max()作用为取最大值，outputs:取最大值的矩阵，dim:维度，1代表每行的最大值，2代表
                    #每列的最大值，以此类推
                    total += labels.size(0)  #取labels矩阵的行，用于统计各类所有的数据
                    correct += (predicted == labels).sum().item()  #预测得到的和labels相同的即为正确
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                            100.0 * correct / total))         #输出正确率
                    total = 0                                 #置0
                    correct = 0                               #置0
                    torch.save({'epoch':epoch,
                                'model_state_dict':net.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':loss
                                },path)                 #保存模型参数，到path

            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))    #跑了多少epoches及跑一个epoch需要的时间

        print('Finished Training')        #跑完100遍epoches，输出完成训练

    def test(self,device):                #定义测试网络效果
        correct = 0                       #正确的数量为0
        total = 0                         #总数量为0
        with torch.no_grad():                #with语句相当于try-finally
            for data in testloader:          #同上，跑一遍测试
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))                    #输出测试结果
#设定device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#设定一个VGGNet实例
net = VGGNet()
#将其输入device
net = net.to(device)
#训练
net.train_sgd(device)
#测试
net.test(device)



                                                                                          