import numpy as np
from Model import UGAT, UGAT_mlp
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

    return model