def metrics_report_func(x):
    if x is not None:
        if x[0].dim() == 0:
            loss, accuracy = x
            return 'loss: {:.4f} - acc: {:.4f}'.format(loss.item(), accuracy)
