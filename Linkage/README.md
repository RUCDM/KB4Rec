## Description
We have three separate files for the three different domains. The three files are:

* ml2fb.txt: MovieLens 20M to Freebase;

* lfb2fb.txt:  LFM-1b to Freebase;

* ab2fb.txt:  Amazon book to Freebase;

In our KB4Rec v1.0 dataset, we organized the linkage results by linked ID pairs, which consists of a RS item ID and a KB entity ID. All the IDs are inner values from the original datasets. Here, we present a sample snippet of our linkage results for MovieLens 20M, in which we pair a MovieLens item ID with a Freebase entity ID.

```   
                                           25991	m.09pglcq
                                           25993	m.0cjwhb
                                           25994	m.0k443
                                           25995	m.0b7kj8
```

## <div id="Licence"></div>Licence

By using the datasets, you must agree to be bound by the terms of the following license.

```
License agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions. 
2. That you include a reference to the KB4Rec v1.0 dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the dataset website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China).
```

## References
Please cite our papers if you have used the datasets in research. 

You can cite this dataset as below.
```
@inproceedings{huang-SIGIR-2018,
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
  url       = {http://doi.acm.org/10.1145/3209978.3210017},
  doi       = {10.1145/3209978.3210017},
}

@inproceedings{Zhao-arxiv-2018,
  author    = {Wayne Xin Zhao and
               Gaole He and
               Hong{-}Jian Dou and
               Jin Huang and 
               Siqi Ouyang and
               Ji{-}Rong Wen and},
  title     = {KB4Rec: A Dataset for Linking Knowledge Bases with Recommender Systems},
  year      = {2018},
  eprint = {arXiv:\embh{cond-mat}/1807.11141},
}
```
