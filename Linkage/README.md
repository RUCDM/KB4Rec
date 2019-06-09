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
