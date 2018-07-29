# KB4Rec
This is the data for KB4Rec
Project address：[https://github.com/RUCDM/KB4Rec/](https://github.com/RUCDM/KB4Rec/)

## 目录
* [Datasets](#Datasets)
* [Models](#Models)
* [Papers](#Papers)
* [Authors](#Authors)

## <div id="Datasets"></div>Datasets
   KB4Rec is a Knowledge-aware Recommender dataset. Now the datasets consists of 3 domains
 法律合同分析平台是中国人民大学信息学院大数据分析与智能实验室研制推出的一套基于自然语言处理算法和神经网络的分析系统，采用多种不同的算法模型对合同类别进行分类，同时能够自动抽取出合同的基本信息（例如甲乙双方的姓名，借款金额，借款时间，借款原因）并将相关条款提取出来。具有如下特点:
1. 速度快，提交纯文本合同，到返回分类结果所需时间仅１s，目前还在不断地优化中
2. 分类准确率高，在我们的测试数据集上具有80％以上的准确率，目前还在不断改善中
3. 功能丰富，除了可以对纯文本进行分析，目前我们还支持文件上传功能，对`pdf`和`word docx`均予以支持。
4. 信息关键词放入配置文件中，可以进行自动扩展，很方便实现对其他领域的信息抽取功能。

## <div id="Models"></div>Models
* KSR [1]
* SVM [3]
* CRF [4]


## <div id="Papers"></div>Papers
* [1]Jin Huang, Wayne Xin Zhao, Hong-Jian Dou, Ji-Rong Wen, Edward Y. Chang. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. SIGIR 2018: 505-514.

## <div id="Authors"></div>Authors
Wayne Xin Zhao, Gaole He, Hongjian Dou, Jin Huang
