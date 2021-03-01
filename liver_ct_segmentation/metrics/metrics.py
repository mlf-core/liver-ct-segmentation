
def accuracy(x, y):

    compare = (x.float() == y.float())
    acc = compare.sum().item()

    ##########################
    # import numpy as np

    # x = x.numpy().astype(np.float16)
    # y = y.numpy().astype(np.float16)

    # compare = np.equal(x, y)
    # acc = np.sum(compare)

    return acc/len(x.flatten())


def iou_fnc(pred, target, n_classes = 12):
    import numpy as np

    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(n_classes)

    # Ignore IoU for background class ("0")
    for cls in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        #intersection = (pred_inds[target_inds]).long().sum().item() #data.cpu()[0]  # Cast to long to prevent overflows
        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0) #ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            count[cls] += 1 
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count
