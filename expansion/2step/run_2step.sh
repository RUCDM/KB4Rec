#python filter_ent.py 2step_graph/ent_1step_all_domain.txt
nohup python scan_triple_2step.py 2step_graph/ent_1step_all_domain.txt ../freebase-rdf-latest.gz? 2step_graph/graph_kb4rec_2step.txt > tr_all_2step &
