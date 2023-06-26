# The method of usage: python3 02_configuration.py help

from sacred import Experiment

ex = Experiment('config_demo')

# Config Scopes
@ex.config
def my_config():
    """This is my demo configuration"""

    a = 10  # some integer

    # a dictionary
    foo = {
        'a_squared': a**2,
        'bar': 'my_string%d' % a
    }
    if a > 8:
        # cool: a dynamic entry
        e = a/2

# the automain function needs to be at the end of the file.
# Otherwise everything below it is not defined yet when the experiment is run.
@ex.main
def my_main():
    pass


if __name__ == '__main__':
    ex.run_commandline()
