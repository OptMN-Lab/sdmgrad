import numpy as np
from collections import defaultdict
"""
The CSV file shoule be like:

Method	mIoU	Pix Acc	Abs Err	Rel Err
LS	75.18	93.49	0.0155	46.77
SI	70.95	91.73	0.0161	33.83
RLW	74.57	93.41	0.0158	47.79
DWA	75.24	93.52	0.016	44.37
UW	72.02	92.85	0.014	30.13
MGDA	68.84	91.54	0.0309	33.5
PCGrad	75.13	93.48	0.0154	42.07
GradDrop	75.27	93.53	0.0157	47.54
CAGrad	75.16	93.48	0.0141	37.6
IMTL-G	75.33	93.49	0.0135	38.41
MoCo	75.42	93.55	0.0149	34.19
MoDo	74.55	93.32	0.0159	41.51
NashMTL	75.41	93.66	0.0129	35.02
FAMO	74.54	93.29	0.0145	32.59
SDMGrad	74.53	93.52	0.0137	34.01
"""


def read_file(fname):
    stats = defaultdict(list)
    keys = ["Method", "mIoU", "Pix Acc", "Abs Err", "Rel Err"]
    with open(fname, "r") as f:
        next(f)
        for line in f:
            if not line.strip("\n"):
                continue
            line = line.strip("\n").split("\t")
            assert len(keys) == len(line)
            for k, item in zip(keys, line):
                stats[k].append(item)
    return stats


reverse_metric_dict = {
    "mIoU": True,
    "Pix Acc": True,
    "Abs Err": False,
    "Rel Err": False
}


def get_rank(stats):
    ranks = defaultdict(dict)
    keys = ["mIoU", "Pix Acc", "Abs Err", "Rel Err"]
    for k in keys:
        values = [(m, v) for m, v in zip(stats["Method"], stats[k])]
        rank_dict = {}
        # if there are duplicated values
        candidates = list(set([v for m, v in values]))
        # if there are no duplicated values
        # candidates = list([v for m, v in values])
        candidates.sort(reverse=reverse_metric_dict[k])
        for i, candidate in enumerate(candidates):
            rank_dict[candidate] = i + 1
        for m, v in values:
            ranks[k][m] = rank_dict[v]
    return ranks


def get_mean_rank(ranks, methods):
    for m in methods:
        sum = 0
        total = len(ranks.keys())
        for k in ranks.keys():
            sum += ranks[k][m]
        r = sum / total
        print(f"Method: {m}\tRank: {r}")

if __name__ == "__main__":
    fname = "cityscapes.csv"
    stats = read_file(fname)
    ranks = get_rank(stats)
    get_mean_rank(ranks, stats["Method"])
