nohup python scan_triple.py ~/KB4Rec/ml2fb.txt ../freebase-rdf-latest.gz? 1step_graph/graph_movie_1step.txt > tr_movie_ &
nohup python scan_triple.py ~/KB4Rec/ab2fb.txt ../freebase-rdf-latest.gz? 1step_graph/graph_book_1step.txt > tr_amazon_ &
nohup python scan_triple.py ~/KB4Rec/lfb2fb.txt ../freebase-rdf-latest.gz? 1step_graph/graph_music_exp.txt > tr_music_ &
