# This example show how to `Named Configurations`
# default config: python3 02_configuration_named.py print_config
#   {'a':10, 'b':30, 'c':"foo"}
# named config, variant1: python3 02_configuration_named.py print_config with variant1
#   {'a':100, 'b':300, 'c':"bar"}
# named config with json file: python3 02_configuration_named.py print_config with my_variant.json
#   {'a':50, 'b':150, 'c':"kaka"}

from sacred import Experiment

ex = Experiment('named_configs_demo')

@ex.config
def cfg():
    a = 10
    b = 3 * a
    c = "foo"

@ex.named_config
def variant1():
    a = 100
    c = "bar"

@ex.main
def my_main():
    pass


if __name__ == '__main__':
    ex.run_commandline()
