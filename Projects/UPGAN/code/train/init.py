import numpy as np
from Model import UGAT, UGAT_mlp, generator
from Model import DistMult, dot_2layer, concat_2layer, concat_1layer, generator_concat
from functools import reduce


def init_model(args,
               user_total,
               item_total,
               entity_total,
               relation_total,
               logger,
               i_map=None,
               e_map=None,
               new_map=None,
               share_total=0):
    logger.info("Building {}.".format("Discriminator"))
    if args.model_name == "UGAT":
        model = UGAT.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.model_name == "UGAT_mlp":
        model = UGAT_mlp.build_model(args, user_total, item_total, entity_total, relation_total)
    else:
        raise NotImplementedError
    logger.info("Architecture: {}".format(model))

    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.info("Total params: {}".format(total_params))

    logger.info("Building {}.".format("Generator"))
    if args.G_name == "generator":
        G = generator.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "generator_concat":
        G = generator_concat.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "dot_2layer":
        G = dot_2layer.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "concat_2layer":
        G = concat_2layer.build_model(args, user_total, item_total, entity_total, relation_total)
    elif args.G_name == "concat_1layer":
        G = concat_1layer.build_model(args, user_total, item_total, entity_total, relation_total)
    else:
        raise NotImplementedError
    logger.info("Architecture: {}".format(G))
    # total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
    #                     for w in G.parameters()])
    # logger.info("Total params: {}".format(total_params))

    # D = DistMult.build_model(args, user_total, item_total, entity_total, relation_total)

    return model, G