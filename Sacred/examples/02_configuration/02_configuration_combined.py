# This example show how to `Combining Configurations`
# Show config: python3 02_configuration_combined.py print_config
# Update config from command-line and print_config
#  python3 02_configuration_combined.py print_config with a=20

from sacred import Experiment

ex = Experiment('multiple_configs_demo')

@ex.config
def my_config():
    a = 10
    b = 'test'

@ex.config
def my_config2(a):  # notice the parameter a here
    c = a * 2       # we can use a because we declared it
    a = -1          # we can also change the value of a
    #d = b + '2'    # error: no access to b

ex.add_config({'e': 'from_dict'})
# out here: {'a': -1, 'b': 'test', 'c': 20, 'e': 'from_dict'}.
# could also add a config file here

@ex.main
def my_main():
    pass


if __name__ == '__main__':
    ex.run_commandline()
