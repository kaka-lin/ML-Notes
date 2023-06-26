from sacred import Experiment

ex = Experiment('capture_config')

# Config Scopes
@ex.config
def my_config():
    foo = 42
    bar = 'baz'

# To use a configuration value all
# you have to do is capture a function
# and accept the configuration value as a parameter.
# Whenever you now call that function Sacred
# will try to fill in missing parameters from the configuration.
# To see how that works we need to capture some function:
@ex.capture
def some_function(a, foo, bar=10):
    print(a, foo, bar)

# the automain function needs to be at the end of the file.
# Otherwise everything below it is not defined yet when the experiment is run.
@ex.main
def my_main():
    some_function(1, 2, 3)     #  1  2   3
    some_function(1)           #  1  42  'baz'
    some_function(1, bar=12)   #  1  42  12
    some_function()            #  TypeError: missing value for 'a'


if __name__ == '__main__':
    ex.run_commandline()
