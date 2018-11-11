
def get_embedding_dim(use_glove: bool):
    if USE_GLOVE:
        # 200 if using glove
        return 100
    else:
        # 256 if without pretrained embedding
        return 256
