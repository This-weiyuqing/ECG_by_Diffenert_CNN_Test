#   ECG_by_Diffenert_CNN_Test
# 本科毕业设计：基于 CNN 的心电异常分类算法的设计与实现
Created by:      魏遇卿

Created on:     2023.3

（2024.4.22——CSDN会偷项目，这个项目是只发布在GitHub上的，地址如下）

(沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN沟槽的CSDN)

github: [This-weiyuqing/ECG_by_Diffenert_CNN_Test]( https://github.com/This-weiyuqing/ECG_by_Diffenert_CNN_Test)
# 1 毕业设计的内容和意义
## **1.1    毕业设计的内容**

本课题将以python语言、MATLAB语言为基础，使用CNN（卷积神经网络）技术，针对MIT-BIT ECG（麻省理工学院关于心律失常的数据库）数据集进行模型训练，进而探求在不同的卷积层和池化层数加持下，卷积神经网络可能的具体表现如何，从而为日后的更深层次的研究建立相对应的基础。

本课题主要分为两个部分，第一部分ECG数据集预处理与第二部分CNN搭建、训练、验证。

在第一部分中，在MATLAB中对数据集进行降噪、去基线漂移操作后识别各波段。在第二部分中，在python环境下使用CNN工具针对数据集进行模型训练，统计模型检验结果，以此检验模型性能。

## **1.2    毕业设计的意义**

健康问题向来不容小觑，伴随着疫情发展，我国国策也走向开放共存。在身边的老年人群体中，个体因为新冠去世、感染老年病的个例层出不穷，已经发展为普遍现象。在这种情况下，老年人的心态发生了较大转变，表现出的忧郁、焦虑等心理的对外表现、器官衰竭等导致的行为不畅或部分功能缺失，会像一座大山压在子女身心。针对心脏问题上，心脏的问题具有强实时性，部分问题在不犯病时无法及时检测出以预防，而在这方面上，专业设备的价格和专业性限制了使用环境。开发可穿戴式、适用于居家的专业仪器是一个新的发展方向，国际公司诸如苹果、华为等已经发布了具备ECG功能的智能穿戴设备，通过获取单导联心电图实现简易心脏类症状的识别，但受限于导联数与穿戴设备稳定性，能够识别的症状不多且无法提供诊断上的辅助。

本课题脱胎于以上实际问题，在专业领域中，技术层出不穷，CNN只是最基本、最简单的技术之一。一方面，本课题可探究CNN在ECG这种多导联、互相之间具备强相关性的数据上进行识别分类的大致性能，另一方面本课题更大的意义在于为研究者提供日后深入研究的原始技术积累，并了解算法开发的大致步骤。针对MIT-BIT这一ECG数据集的数据处理过程也有助于理解不同的数据集以及数据集的本质。

# 2 文献综述

## 2.1  研究背景

伴随着本次全球化的健康危机，即使是我国，死亡人数相较往年也有较大增长，这一结论来自于实际生活，具备真实性。随着毒株毒性下降，社会逐渐恢复开放的问题在前文已经提及。老年人的健康因此普遍面临着新一轮的挑战。而据《中国心血管健康与疾病报告2021》的数据研究表明，中国心血管疾病患者数逐年增长，且暂时无法到达下降转折点[1]，且呈现出由中老年向年轻人扩散的趋势。文中亦指出，中国心血管病死亡占城乡居民总死亡原因的首位，农村为 46. 74%，城市为 44. 26%。目前，血压仪已经在中老年人家庭中基本普及，而ECG类仪器则因为需要专业能力以读图，故而在各式各样的家庭构成中普及率普遍并不高。而各式各样的智能穿戴设备都是单导联ECG，有论文指出其在严格受控条件下，成图与专业ECG机器对应导联的成图具备相同的可靠性[2]。但由于其导联数过于单一，且使用受使用环境影响过大致使实际使用时噪声过多，于业内其最终成图并不被做临床诊断使用，只能作为日常参考使用。

而随着计算机科学与人工智能领域的发展，各种各样的交叉学科逐渐出现，人工智能\*医学在国内外也开始有越来越多的学者涉足。通过数字图像处理相关技术，结合CNN，Res-Net等神经网络，针对ECG进行模型训练，通过模型判断进而识别不同的症状提供给医护，从而提高医护诊断的正确率并减轻医护工作量。或直接或间接提高患者存活率。而在机器学习领域，可选择支持向量机（support vector machine，SVM）、贝叶斯分类等算法进行分类[3]，但训练模型性能有限，若通过增加集成学习，进行整合，则能够收获更好的性能。这一思路也被应用在通过ECG进行生物识别上[4]。随着深度学习的发展，具备更好的泛化性的深度学习相关算法逐渐被应用在了这一领域，相对于传统机器学习算法，深度学习算法具备更好的泛化性，其训练出的模型也具备更好的性能，但同时，因为其训练机制的特殊性也引发了过拟合问题。针对过拟合问题可通过具体应用环境设置激活函数进行修正。

## 2.2  国内外研究现状

ECG实质上是一种微弱的生物电信号，也因此，其对于噪声的表现会更加明显，ECG的主要干扰为以下三种：频率小于5Hz的基线漂移；频率固定为50Hz的工频干扰；频率范围较广的肌电干扰。在ECG的降噪方面，有通过小波变换进行降噪、心拍识别的处理方式[5]。这一方式也相较更为成熟。

在ECG识别上，李伟康等通过附加SE模块[6]的1D CNN、SE-CNN、CNN-GRU进行综合集成学习，进一步提升了ECG分类的准确率至99.12%[7]。也有学者利用深度学习，通过SSWPT算法、RDBR算法、ＭＭＤ算法提取PPG信号中的心脏、呼吸信号并与ECG信号进行时域同步处理，将ECG与PPG（光电容积脉搏波描记法photoplethysmonraph）结合[8]，于血压检测精度问题上取得了一定进展。有学者提出了CNN-FWS算法，将CNN与基于特征权重的递归特征消除算法（FW-RFE）进行结合[9]。作为较新的人工智能技术，RNN（递归神经网络）中的长短期记忆网络（Long short-term memory，LSTM）算法也在被引用进该类问题。Petmezas, Georgios等人使用CNN与LSTM结合，提出了CNN-LSTM算法[10]，该算法CNN部分使用池化层选用了最大池化算法，以CNN的输出作为LSTM的输入。LSTM部分由一层LSTM层、一层扁平层、一个全连接层、一个dropout层和一个输出层构成。该算法灵敏度在各算法中较优，但其特异性达到了最优。

## 2.3  实际应用方面的价值

本课题面向ECG诊断问题，聚焦于ECG与基础深度学习算法。通过针对经典ECG数据集MIT-BIT数据集的模型训练，测试CNN在一维数据上的性能。通过调整CNN网络结构，测试不同层数结构下的CNN网络在MIB-BIT数据集上的性能，从而选择出在针对这一问题上所适宜的CNN网络层数。而本课题所验证的算法将作为更深一步研究的基础研究。为针对该问题进行进一步算法结构设计提供原始CNN网络结构支持。同样，由于ECG数据多导联的强时域相关性、保存格式的特殊性，本课题处理数据集时所使用的思想也可作为日后针对其他神经网络对ECG数据处理的参考经验。

# 3：相关技术与研发基础

## 3.1  相关技术

在本课题中，将使用matplotlib等python数学计算包通过小波变换针对原始数据集进行处理，基于tensorflow或pytorch平台进行CNN网络搭建。综上，有如下几个部分：小波变换、CNN、tensorflow、pytorch、MATLAB。

1.  小波变换是一种基于小波的变换分析方法，由于小波的时域性，相较傅里叶变换，小波变换具备更好的时域分析能力，同时，其窗口大小更为灵活，可提供随频率变换的“时间-频率”窗口。由于其在信号时域分析处理上的优势，小波变换很适合在ECG分析中作为降噪、信号提取的工具[5]。
2.  CNN，卷积神经网络，作为深度学习代表算法之一，主要结构为卷积层、池化层、全连接层三层结构[11]。于卷积层通过卷积核进行特征提取，随后传入池化层，经由池化函数进行特征选择、信息过滤等操作，最后传入全连接层向后传递进行后续操作。CNN具有过拟合问题，需要通过添加激励函数解决，常用激励函数有ReLU、sigmoid函数等。激励函数通常加在卷积层与池化层之间，也有算法如LeNet-5将激励函数添加在了池化层之后。
3.  Tensorflow是由Google Brain开发，提供了多种语言支持的开源机器学习平台，封装了大量机器学习通用功能。因此可通过tensorflow构建各式各样的人工智能算法，现行版本为tensorflow 2 。tensorflow 1与tensorflow 2的兼容性并不乐观。
4.  Pytorch是基于torch的开源python机器学习库，也提供了多种语言支持，现行版本为1.13。相较tensorflow具备更好的版本兼容性。本课题将在tensorflow、pytorch中选择一个作为开发环境支持。
5.  MATLAB作为首屈一指的数学处理平台，提供了多种语言的支持，python语言上包名为matplotlib。其专业性使得在人工智能领域得到了广泛应用。

## 3.2  研发基础

1.  开发者通过学习，积攒了一定的机器学习、数字图像处理基础理论知识储备。
2.  开发者曾接触过人工智能类项目，对于基础函数、网络设计具备一定的了解与编程能力。
3.  开发者个人PC配置为i7-11800H，RTX3070Laptop，32G DDR4，具备针对本课题的算法运行能力。
4.  开发者可通过浙江图书馆、Google学术等平台，获取到相应的文献以针对本课题进行进一步的知识储备。

## 3.3  毕业论文参考文献

    [1]	《中国心血管健康与疾病报告2021》要点解读 [J]. 中国心血管杂志, 2022, 27(04): 305-18.
    [2]	SAMOL A, BISCHOF K, LUANI B, et al. Single-Lead ECG Recordings Including Einthoven and Wilson Leads by a Smartwatch: A New Era of Patient Directed Early ECG Differential Diagnosis of Cardiac Diseases? [J]. Sensors, 2019, 19(20): 4377.
    [3]	刘奇. 基于集成学习算法的ECG身份识别 [D]; 吉林大学, 2019.
    [4]	欧阳波. 基于小波分析的ECG信号处理技术研究 [D]; 湖南大学, 2014.
    [5]	LOGESH R, SUBRAMANIYASWAMY V, MALATHI D, et al. Enhancing recommendation stability of collaborative filtering recommender system through bio-inspired clustering ensemble method [J]. Neural Computing and Applications, 2020, 32(7): 2141-64.
    [6]	李伟康. 基于混合深度学习算法的ECG心电信号分类研究 [D]; 江苏科技大学, 2022.
    [7]	胡军锋, 郑彬. 基于深度学习的ECG/PPG血压测量方法 [J]. 生物医学工程研究, 2022, 41(1): 46-54.
    [8]	ZHU J, LV J, KONG D. CNN-FWS: A Model for the Diagnosis of Normal and Abnormal ECG with Feature Adaptive [J]. Entropy, 2022, 24(4): 471-.
    [9]	PETMEZAS G, HARIS K, STEFANOPOULOS L, et al. Automated Atrial Fibrillation Detection using a Hybrid CNN-LSTM Network on Imbalanced ECG Datasets [J]. Biomedical Signal Processing and Control, 2021, 63: 102194.
    [10]	GOLDBERGER A, AMARAL, L., GLASS, L., HAUSDORFF, J., IVANOV, P.C., MARK, R., MIETUS, J.E., MOODY, G.B., PENG, C.K. AND STANLEY, H.E., 2000. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. [J]. Circulation [Online], 2000, (101 (23)): e215–e20.
    [11]	贺京生. 基于小波变换的十二导联心电信号分析研究 [D]; 南昌大学, 2016.
    [12]	HUBEL D H, WIESEL T N. Receptive fields, binocular interaction and functional architecture in the cat's visual cortex [J]. The Journal of Physiology, 1962, 160(1): 106-54.
    [13]	FUKUSHIMA K. Cognitron: A self-organizing multilayered neural network [J]. Biological Cybernetics, 1975, 20(3-4): 121-36.
    [14]	FUKUSHIMA K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position [J]. Biological Cybernetics, 1980, 36(4): 193-202.
    [15]	LECUN Y, BOTTOU L, BENGIO Y, et al. Gradient-based learning applied to document recognition [J]. Proceedings of the IEEE, 1998, 86(11): 2278-324.
    [16]	TOMPOLLARD. MIT-LCP/wfdb-python:Native Python WFDB package [Z]. 
    [17]	HOCHREITER S, SCHMIDHUBER J. Long short-term memory [J]. Neural Comput, 1997, 9(8): 1735-80.
    [18]	BUBECK S E, CHANDRASEKARAN V, ELDAN R, et al. Sparks of Artificial General Intelligence: Early experiments with GPT-4 [J]. arXiv pre-print server, 2023.

# 4 代码文件说明
##  4.1  _main_tensorflow_oneLayerCNN.py

本文件为CNN模型搭建文件，代码部分包括：
### 4.1.1  PATH声明
    
    #from ECG_read_weiyuqing_without_wfdb import ECGDATAPATH
    #PATH of this test
    Project_PATH = "../Number-Of-CNN-Layers/One/"
    #PICTUREPATH= Project_PATH + "picture/"
    log_dir = Project_PATH + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_Path_one = Project_PATH  + "model/ecg_model_one_layer.h5"
    #model_Path_one = Project_PATH + "ecg_model_one_layer.h5"

### 4.1.2   CNN参数说明

    RATIO = 0.3
    RANDOM_SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
### 4.1.3   CNN模型构建
    def CNN_model_level_one():
        leavlOneModel = tf.keras.models.Sequential([    
            #take test for one CNN layers model
    
            tf.keras.layers.InputLayer(input_shape=(300, 1)),
    
            # ECG data cant add 0 in edge, will broken data ,take false answer. So we need padding same data in edge.
    
            # take four 21*1 conv excitation function used RElu
            tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
            # take four 3*1 pool,take strids 2.
    
            tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
    
            # tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
            # tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
    
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        return leavlOneModel
### 4.1.4   顶层设计
    def main():
        # get traxin test
        X_train, X_test, y_train, y_test = loadData(RATIO, RANDOM_SEED)
    
        # get model or create model
        if os.path.exists(model_Path_one):
            print(' get model in h5 file')
            model = tf.keras.models.load_model(filepath=model_Path_one)
        else:
            # create new model(if model unexists)
            model = CNN_model_level_one()
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.summary()
            # TB make
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                                  histogram_freq=1,
                                                                  )
            # Training and validation
            history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(X_test, y_test),
                                callbacks=[tensorboard_callback])
            # validation_split=RATIO,
            model.save(filepath=model_Path_one)
            plot_history_tf(history)
    
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        plot_heat_map(y_test, y_pred)


# 5 文件存储体系结构说明 

## 5.1  文件夹结构说明
![img.png](pictureOfReadme%2Fimg.png)
### 5.2 Number-Of-CNN-Layers
表示本文件夹存储信息为：不同CNN模型的层数，并进行相应存储

#### 5.2.1  one:表示本文件夹存储的为一层CNN网络（包含一个卷积层和一个池化层、一个全连接层）

##### 5.2.2 logs：对应的训练日志文件夹，存储训练日志

##### 5.2.3 model：对应的CNN模型文件夹，存储CNN模型

##### 5.2.4 picture：图片文件夹，存储生成的图片
5.2.4.1 模型精度曲线（accuracy.png）

![accuracy.png](Number-Of-CNN-Layers%2FOne%2Fpicture%2Faccuracy.png)

5.2.4.2 模型损失函数曲线（loss.png）

![loss.png](Number-Of-CNN-Layers%2FOne%2Fpicture%2Floss.png)

5.2.4.3 模型最终分类效果矩阵（confusion_matrix.png）

![confusion_matrix.png](Number-Of-CNN-Layers%2FOne%2Fpicture%2Fconfusion_matrix.png)

### 5.3 Number-Of-Conv

设置卷积层的不同卷积核进行测试，内部文件夹结构同上。

### 5.4 Number-Of-Pool: 设置池化层的不同大小池化核

设置池化层的不同大小池化核进行测试，内部文件夹结构同上。

#   6   一些可能可行的改观

##  6.1 数据集

##  6.2 数据集处理方式

##  6.3 网络结构的构建

本部分不做展开描述。您有想法可发邮件至如下邮箱探讨。
        
    this_weiyuqing@foxmail.com 

#   7   其他

其他文件请查看毕业论文。
