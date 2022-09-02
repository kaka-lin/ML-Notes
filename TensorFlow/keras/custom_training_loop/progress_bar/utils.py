import tensorflow as tf

# declare global variables for storing previous loss, previous accuracy
g_loss, g_accuracy = None, None

def metrics_report_func(x):
    # using global variables for storing loss and accuracy
    global g_loss
    global g_accuracy

    if x is not None:
        if tf.size(x[0]).numpy() == 1:
            loss, accuracy = x
            g_loss, g_accuracy = loss, accuracy # store loss, accuracy into global variables
            return f'loss: {loss.numpy():.4f} - accuracy: {accuracy.numpy():.4f}'
        else:
            if g_loss is not None:
                return f'loss: {g_loss.numpy():.4f} - accuracy: {g_accuracy.numpy():.4f}'
