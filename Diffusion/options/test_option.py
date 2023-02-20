from .base_option import BaseOption

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        
        self.parser.add_argument("--epoch", type=int, default=100)

