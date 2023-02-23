from .base_option import BaseOption

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument("--is_train", type=bool, default=True)


        self.parser.add_argument("--patch_size", type=int, default=256,
            help="size of random patch for training")

        self.parser.add_argument("--loss_type", type=str, default="l1",
            help="loss function for model training")
        self.parser.add_argument("--metric_type", type=str, default="l2",
            help="metric function for modeltraining monitoring")

        self.parser.add_argument("--batch_size", type=int, default=1,
            help="batch size")
        self.parser.add_argument("--num_workers", type=int, default=8,
            help="# of process for dataloader")

        self.parser.add_argument("--lr", type=float, default=0.0002,
            help="initial learning rate")
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument("--weight_decay", type=float, default=0.001,
            help="l2 regularization")

        self.parser.add_argument("--nb_epochs", type=int, default=100,
            help="# of epochs with initial learning rate")
        self.parser.add_argument("--nb_epochs_decay", type=int, default=100,
            help="# of epochs with linearly decaying learning rate")

        self.parser.add_argument("--report_freq", type=int, default=1000,
            help="report frequency in iterations")

