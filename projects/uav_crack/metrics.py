"""Small dependency-free binary segmentation metrics for protocol checks."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    confusion: tuple[tuple[int, int], tuple[int, int]]  # rows=true, cols=pred
    iou_background: float
    iou_crack: float
    miou: float
    precision_crack: float
    recall_crack: float
    f1_crack: float
    pixel_accuracy: float


def metrics_from_confusion(confusion) -> BinaryMetrics:
    """Compute metrics from a 2x2 confusion matrix; zero-union IoU is 0."""
    c=[[int(v) for v in row] for row in confusion]
    if len(c)!=2 or any(len(row)!=2 for row in c) or any(v<0 for row in c for v in row):
        raise ValueError('confusion must be a non-negative 2x2 matrix')
    total=sum(map(sum,c))
    if total==0: raise ValueError('confusion matrix must contain at least one pixel')
    ious=[]
    for k in (0,1):
        union=sum(c[k])+sum(c[r][k] for r in (0,1))-c[k][k]
        ious.append(c[k][k]/union if union else 0.0)
    tp,fp,fn=c[1][1],c[0][1],c[1][0]
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return BinaryMetrics(tuple(tuple(row) for row in c),ious[0],ious[1],sum(ious)/2,
                         precision,recall,f1,(c[0][0]+c[1][1])/total)


def binary_metrics(truth, prediction) -> BinaryMetrics:
    """Compute pixel-aggregated metrics. Inputs must contain only class IDs 0 and 1."""
    y,p=list(truth),list(prediction)
    if len(y)!=len(p) or not y:
        raise ValueError('truth and prediction must have the same non-zero length')
    if any(v not in (0,1) for v in y) or any(v not in (0,1) for v in p):
        raise ValueError('truth and prediction must contain only 0 and 1')
    c=[[0,0],[0,0]]
    for t,q in zip(y,p): c[t][q]+=1
    return metrics_from_confusion(c)
