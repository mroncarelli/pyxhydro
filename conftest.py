import tests.randomutils

def pytest_addoption(parser):
    parser.addoption("--eventlist",
                     action="store",
                     default="standard",
                     choices=["standard", "complete"],
                     type=str,
                     help="Available options: standard, complete"
                     )
    parser.addoption("--seed",
                     action="store",
                     default=None,
                     help="Seed for random number generator")


# The initial random seed is stored as a global variable in the tests.randomutils module
def pytest_configure(config):
    seed_ = config.getoption("seed")
    tests.randomutils.globalRandomSeed = seed_ if seed_ is None else int(seed_)
