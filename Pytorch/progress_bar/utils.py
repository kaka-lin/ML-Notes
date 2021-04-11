def metrics_report_func(x):
    if x is not None:
        loss, accuracy = x
        return 'loss: {:.4f} - acc: {:.4f}'.format(loss.item(), accuracy)
