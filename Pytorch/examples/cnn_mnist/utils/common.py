import os
import errno

import torch
import numpy as np


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False


def save_model(model, path='./', mode='train',
               model_name='model', **kwargs):
    if mode == 'checkpoint':
        path = path + 'models/pre_trains/{}_ckpt.pth'.format(model_name)
    else:
        path = path + 'models/pre_trains/{}.pth'.format(model_name)

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if mode == 'inference':
        torch.save(model.state_dict(), path)
    elif mode == 'checkpoint':
        torch.save({
            'model_state_dict': model.state_dict(),
            **kwargs
        }, path)
    else:
        torch.save(model, path)
