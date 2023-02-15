from .base_option import BaseOption

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument("--patch_size", type=int, default=256,
            help="size of random patch for training")

        self.parser.add_argument("--noise_loc", type=float, default=0,
            help="mode value of Gaussian noise")
        self.parser.add_argument("--noise_scale", type=float, default=10,
            help="standard deviation value of Gaussian noise")
        self.parser.add_argument("--scale_range", type=float, default=3,
            help="deviation range noise scale")

        self.parser.add_argument("--steps", type=int, default=200,
            help="# of adding diffusion steps")
        self.parser.add_argument("--beta_start", type=float, default=0.0001,
            help="Diffusion beta start value")
        self.parser.add_argument("--beta_end", type=float, default=0.02,
            help="Diffusion beta end value")

        self.parser.add_argument("--loss_type", type=str, default="l1",
            help="loss function for model training")

        self.parser.add_argument("--metric_type", type=str, default="l2",
            help="metric function for modeltraining monitoring")

        self.parser.add_argument("--batch_size", type=int, default=4,
            help="batch size")
        self.parser.add_argument("--num_workers", type=int, default=16,
            help="# of process for dataloader")

        self.parser.add_argument("--lr", type=float, default=0.0002,
            help="initial learning rate")
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument("--weight_decay", type=float, default=0.001,
            help="l2 regularization")

        self.parser.add_argument("--nb_epochs", type=int, default=10,
            help="# of epochs with initial learning rate")
        self.parser.add_argument("--nb_epochs_decay", type=int, default=10,
            help="# of epochs with linearly decaying learning rate")

        self.parser.add_argument("--report_freq", type=int, default=100,
            help="report frequency in iterations")