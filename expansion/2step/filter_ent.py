import sys
import os
def filter_ent(output):
    files = ["movie", "music", "book"]
    root_path = "./1step_graph/"
    entities = set()
    for file in files:
        filename = "graph_%s_1step.txt"%(file)
        filename = os.path.join(root_path, filename)
        f = open(filename)
        for line in f:
            line = line.strip().split("\t")
            if "http://rdf.freebase.com/ns/" in line[0]:
                entities.add(line[0][28:-1])
            if "http://rdf.freebase.com/ns/" in line[2]:
                entities.add(line[2][28:-1])
        f.close()
    f1 = open(output, "w")
    for ent in entities:
        f1.write("%s\n"%(ent))
    f1.close()
filter_ent(sys.argv[1])