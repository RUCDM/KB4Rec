We have three separate files for the three different domains. The three files are:

(1) ml2fb.txt: MovieLens 20M to Freebase;

(2) lfb2fb.txt:  LFM-1b to Freebase;

(3) ab2fb.txt:  Amazon book to Freebase;

Each file takes the following data format by lines:

RS_item_ID[\tab]FB_item_ID

where a RS_item_ID denotes an item ID from a recommender system dataset, while a FB_item_ID denotes an entity ID from Freebase.

Please cite our papers if you have used the datasets in research. 

You can cite this dataset as below.

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
