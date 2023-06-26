from sacred import Experiment

ex = Experiment('hello_config')

# Config Scopes
@ex.config
def my_config():
    recipient = "world"
    message = f"Hello {recipient}"

# the automain function needs to be at the end of the file.
# Otherwise everything below it is not defined yet when the experiment is run.
@ex.automain
def my_main(message):
    print(message)
