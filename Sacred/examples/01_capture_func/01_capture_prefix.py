# If you have some function that only needs to access some sub-dictionary
# of your configuration you can use the prefix parameter of @ex.capture:

from sacred import Experiment

ex = Experiment('prefix_demo')

@ex.config
def my_config1():
    dataset = {
        'filename': 'foo.txt',
        'path': '/tmp/'
    }

@ex.capture(prefix='dataset')
def print_me(filename, path):  # direct access to entries of the dataset dict
    print("filename =", filename)
    print("path =", path)

# the automain function needs to be at the end of the file.
# Otherwise everything below it is not defined yet when the experiment is run.
@ex.main
def my_main():
    print_me()            # {'filename': 'foo.txt', 'path': '/tmp/'}
    print_me('test.txt')  # {'filename': 'test.txt', 'path': '/tmp/'}


if __name__ == '__main__':
    ex.run_commandline()
