# KB4Rec
This project is a description for dataset KB4Rec, a knowledge-aware recommder linkage dataset.

## Directory
* [Descriptions](#Descriptions)
* [Datasets](#Datasets)
* [DownLoad and Usage](#Download)
* [Papers Related](#Papers)
* [References](#References)
* [Additional Notes](#Addition)

## <div id="Descriptions"></div>Descriptions
<!--
Nowadays, recommender systems (RS), which aim to match users with interested items, have played an important role in various online applications. Traditional recommendation algorithms mainly focus on learning effective preference models from historical user-item interaction data, e.g. matrix factorization. With the rapid development of Web techniques, various kinds of side information has become available in RSs, called context. In an early stage, such context information is usually unstructured, and its availability is limited to specific data domains or platforms.-->
   
   Recently, more and more efforts have been made by both research and industry communities for structurizing world knowledge or domain facts in a variety of data domains. One of the most typical organization forms is knowledge base (KB), also called knowledge graph. KBs provide a general and unified way to organize and relate information entities, which have been shown to be useful in many applications. Specially, KBs have also been used in recommender systems, called knowledge-aware recommender systems [1].
   
   To address the need for the linked dataset of RS and KBs, we present the first public linked KB dataset for recommender systems, named KB4Rec v1.0.
   
## <div id="Datasets"></div>Datasets
   In our KB4Rec v1.0 dataset, we organized the linkage results by linked ID pairs, which consists of a RS item ID and a KB entity ID. All the IDs are inner values from the original datasets. Once such a linkage has been accomplished, it is able to reuse existing large-scale KB data for RSs.
   
   For example, the movie of <Avatar> from MovieLens dataset has a corresponding entity entry in Freebase, and we are able to obtain its attribute information by reading out all its associated relation triples in KBs.

   We consider three popular RS datasets for linkage, namely MovieLens 20M, LFM-1b and Amazon book, which covers the three domains of movie, music and book respectively. For KB, We adopt the large-scale pubic KB Freebase. 

### Linkage Detail Statistics：
| Dataset                 | Items      |    Linked-Items    |  Linkage-ratio   | 
|:-------------------------:|:-------------:|:------------:|:------------:|
|MovieLens 20M|27,279 |25,982|95.2%|
|LFM-1b|6,479,700 |1,254,923|19.4%|
|Amazon book|2,330,066 |109,671|4.7%|

   For more details of our linkage, please refer to our dataset [paper](www.baidu.com).
<!--
## <div id="Models"></div>Models
* KSR [2]
* [SVDfeature](http://apex.sjtu.edu.cn/projects/33)
-->
## <div id="Download"></div>DownLoad and Usage
<!--To use the datasets, you must read and accept the online agreement-->. By using the datasets, you agree to be bound by the terms of its license. Send email to xxx. The email format should contain following contents:

```
License agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions.
2. That you include a reference to the KB4Rec v1.0 dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China).
I have seen this license, and accept it.
```

If you use the dataset, please cite the paper [2],[3] listed in our reference.

## <div id="Papers"></div>Papers Related
Thre are some papers which use our dataset, you can refer to them. If your paper is not listed, please let us know xxx@.cn.

* Jin Huang, Wayne Xin Zhao, Hong-Jian Dou, Ji-Rong Wen, Edward Y. Chang. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. SIGIR 2018: 505-514. [paper](https://dl.acm.org/citation.cfm?doid=3209978.3210017) [code](https://github.com/BetsyHJ/KSR)

## <div id="References"></div>References
* [1] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, Wei-Ying Ma. Collaborative Knowledge Base Embedding for Recommender Systems. KDD 2016: 353-362. [paper](https://dl.acm.org/citation.cfm?doid=2939672.2939673)
* [2] Jin Huang, Wayne Xin Zhao, Hong-Jian Dou, Ji-Rong Wen, Edward Y. Chang. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. SIGIR 2018: 505-514. [paper](https://dl.acm.org/citation.cfm?doid=3209978.3210017) [code](https://github.com/BetsyHJ/KSR)
* [3] Our work.

   You can cite this dataset as below.

```
@inproceedings{DBLP:conf/sigir/HuangZDWC18,
  author    = {Jin Huang and
               Wayne Xin Zhao and
               Hong{-}Jian Dou and
               Ji{-}Rong Wen and
               Edward Y. Chang},
  title     = {Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks}
  booktitle = {The 41st International {ACM} {SIGIR} Conference on Research {\&}
               Development in Information Retrieval, {SIGIR} 2018, Ann Arbor, MI,
               USA, July 08-12, 2018}
  pages     = {505--514}
  year      = {2018},
  crossref  = {DBLP:conf/sigir/2018},
  url       = {http://doi.acm.org/10.1145/3209978.3210017},
  doi       = {10.1145/3209978.3210017},
  timestamp = {Mon, 02 Jul 2018 08:24:13 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/sigir/HuangZDWC18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## <div id="Addition"></div>Additional Notes
* The following people contributed to the the construction of the KB4Rec v1.0 dataset: Wayne Xin Zhao, Gaole He, Hongjian Dou, Jin Huang, Siqi Ouyang and Ji-Rong Wen. This project is lead by Wayne Xin Zhao, School of Information, Renmin University of China.
* If you have any questions or suggestions with this dataset, please kindly let us know. Our goal is to make the dataset reliable and useful for the community.
