

def accuracy(x, y):

    compare = (x.float() == y.float())
    acc = compare.sum().item()

    return acc/len(x.flatten())


def iou_fnc(pred, target, n_classes=12):
    import numpy as np

    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection  # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count
