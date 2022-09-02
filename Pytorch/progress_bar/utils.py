# declare global variables for storing previous loss, previous accuracy
g_loss, g_accuracy = None, None

def metrics_report_func(x):
    # using global variables for storing loss and accuracy
    global g_loss
    global g_accuracy

    if x is not None:
        if x[0].dim() == 0:
            loss, accuracy = x
            g_loss, g_accuracy = loss, accuracy # store loss, accuracy into global variables
            return f'loss: {loss.item():.4f} - accuracy: {accuracy:.4f}'
        else:
            if g_loss is not None:
                return f'loss: {g_loss.item():.4f} - accuracy: {g_accuracy:.4f}'
