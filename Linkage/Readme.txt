We have three separate files for the three different domains. 

Every directory named by RS dataset name, and contains linkage to freebase as xx2fb.txt, linkage to yago as xx2yago.txt.
Besides, we collect the overlap of linkage on three domains as {domain}_link_overlap.txt.

Freebase linkage:

(1) ml2fb.txt: MovieLens 20M to Freebase;

(2) lfb2fb.txt:  LFM-1b to Freebase;

(3) ab2fb.txt:  Amazon book to Freebase;

YAGO linkage:

(4) ml2yago.txt: MovieLens 20M to YAGO;

(5) lfb2yago.txt:  LFM-1b to YAGO;

(6) ab2yago.txt:  Amazon book to YAGO;

Overlap of linkage:

(7) movie_link_overlap.txt: MovieLens 20M linkage overlap of Freebase and YAGO;

(8) music_link_overlap.txt:  LFM-1b linkage overlap of Freebase and YAGO;

(9) book_link_overlap.txt:  Amazon book linkage overlap of Freebase and YAGO;

Each file takes the following data format by lines:

(1) - (3): RS_item_ID[\tab]FB_item_ID

(4) - (6): RS_item_ID[\tab]YAGO_item_ID

(7) - (9): RS_item_ID[\tab]FB_item_ID[\tab]YAGO_item_ID[\tab]RS_item_title

where a RS_item_ID denotes an item ID from a recommender system dataset, RS_item_title denotes an item title from a recommender system dataset.
while a FB_item_ID denotes an entity ID from Freebase, YAGO_item_ID denotes an entity ID from YAGO.

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

@article{DBLP:journals/corr/abs-1807-11141,
  author    = {Wayne Xin Zhao and
               Gaole He and
               Hong{-}Jian Dou and
               Jin Huang and
               Siqi Ouyang and
               Ji{-}Rong Wen},
  title     = {KB4Rec: {A} Dataset for Linking Knowledge Bases with Recommender Systems},
  journal   = {CoRR},
  volume    = {abs/1807.11141},
  year      = {2018},
  url       = {http://arxiv.org/abs/1807.11141},
  archivePrefix = {arXiv},
  eprint    = {1807.11141},
  timestamp = {Mon, 13 Aug 2018 16:48:44 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1807-11141},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

