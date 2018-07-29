# KB4Rec
This is the data for KB4Rec
Project address：[https://github.com/RUCDM/KB4Rec/](https://github.com/RUCDM/KB4Rec/)

## 目录
* [Descriptions](#Descriptions)
* [Datasets](#Datasets)
* [Models](#Models)
* [Papers](#Papers)
* [Authors](#Authors)

## <div id="Descriptions"></div>Descriptions
<!--
Nowadays, recommender systems (RS), which aim to match users with interested items, have played an important role in various online applications. Traditional recommendation algorithms mainly focus on learning effective preference models from historical user-item interaction data, e.g. matrix factorization. With the rapid development of Web techniques, various kinds of side information has become available in RSs, called context. In an early stage, such context information is usually unstructured, and its availability is limited to specific data domains or platforms.-->
   
   Recently, more and more efforts have been made by both research and industry communities for structurizing world knowledge or domain facts in a variety of data domains. One of the most typical organization forms is knowledge base (KB), also called knowledge graph. KBs provide a general and unified way to organize and relate information entities, which have been shown to be useful in many applications. Specially, KBs have also been used in recommender systems, called knowledge-aware recommender systems[1].
   
   To address the need for the linked dataset of RS and KBs, we present the first public linked KB dataset for recommender systems, named KB4Rec v1.0.
   
## <div id="Datasets"></div>Datasets
   In our KB4RecSys v1.0 dataset, we organized the linkage results by linked ID pairs, which consists of a RS item ID and a KB entity ID. All the IDs are inner values from the original datasets. Once such a linkage has been accomplished, it is able to reuse existing large-scale KB data for RSs.
   
   For example, the movie of <Avatar> from Movielens dataset has a corresponding entity entry in Freebase, and we are able to obtain its attribute information by reading out all its associated relation triples in KBs.


   We consider three popular RS datasets for linkage, namely MovieLens, LFM-1b and Amazon Book, which covers the three domains of movie, music and book respectively. For KB, We adopt the large-scale pubic KB Freebase. 

### Linkage Detail Statis：
| Dataset                 | Items      |    Linked-Items    |  Linkage-ratio   | 
|:-------------------------:|:-------------:|:------------:|:------------:|
|Movielens-20m|27,279 |25,982|95.2%|
|music|6,479,700 |1,254,923|19.4%|
|book|2,330,066 |109,671|4.7%|

## <div id="Models"></div>Models
* KSR [2]
* [SVDfeature](http://apex.sjtu.edu.cn/projects/33)

## <div id="Download"></div>DownLoad and Usage
To use the datasets, you must read and accept the online agreement. By using the datasets, you agree to be bound by the terms of its license. Send email to xxx. The email format should contain following contents:

If you use the dataset, please cite the paper [2],[3] listed in our reference.
## <div id="Papers"></div>Papers
* [1] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, Wei-Ying Ma. Collaborative Knowledge Base Embedding for Recommender Systems. KDD 2016: 353-362. [paper](https://dl.acm.org/citation.cfm?doid=2939672.2939673)
* [2] Jin Huang, Wayne Xin Zhao, Hong-Jian Dou, Ji-Rong Wen, Edward Y. Chang. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. SIGIR 2018: 505-514. [paper](https://dl.acm.org/citation.cfm?doid=3209978.3210017) [code](https://github.com/BetsyHJ/KSR)
* [3] Our work.

## <div id="Authors"></div>Authors
Wayne Xin Zhao, Gaole He, Hongjian Dou, Jin Huang
