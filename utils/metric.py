import torch

def euclidean_distance(x1, x2) :

    return torch.norm(x1-x2).item()


def cosine_similarity(x1,x2) :
    dot_prod = torch.dot(x1,x2)
    norm_x1 = torch.norm(x1)
    norm_x2 = torch.norm(x2)

    return (dot_prod / (norm_x1*norm_x2)).item()


def mean_absolute_difference(original , generated) :
    abs_diff = torch.abs(original-generated)

    mad = torch.mean(abs_diff)

    return mad

def anomaly_score(embedding_score, difference_score , ratio = 0.5) :
    ano_score = (ratio * embedding_score) + (1-ratio) * (difference_score)

    return ano_score

"""
위 파이썬 파일을 디렉토리에 넣고
from metric import euclidean_distance , cosine_similarity

euclidean_distance(original_embedding , generated_embedding)
cosine_similarity(original_embedding , generated_embedding)


generated = gen_img_accum / mask_iter
mean_absolute_difference(input , generated)

anomaly_score(임베딩 벡터간 구한 거리 , mean_absolute_difference)

"""