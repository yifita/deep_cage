import argparse
import datetime
import os
import io
from glob import glob
import itertools

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--name", help="experiment name")
        parser.add_argument("--is_poly", action="store_true", help="source target contains polygon mesh")
        parser.add_argument("--cage_deg", type=int, help="number of vertices on cage", default=20)
        parser.add_argument("--dataset", type=str, help="dataset name", default="COSEG",
                            choices=["COSEG", "FAUST", "SHAPENET", "SHAPENETV2", "SURREAL", "MNIST_SINGLE", "MNIST_MIXED"])
        parser.add_argument("--isV2", action="store_true", help="using Shapenetv2 for shapenetseg dataset")
        parser.add_argument("--no-preprocessed", dest="use_preprocessed", action="store_false", help="using preprocessed")
        parser.add_argument("--num_point", type=int, help="number of input points", default=2048)
        parser.add_argument("--regular_sampling", action="store_true", help="sample considering face area")
        parser.add_argument("--template", type=str, help="cage template", default="data/sphere_V42_F80.off")
        parser.add_argument("--source_model", type=str, nargs="*", help="source model for testing")
        parser.add_argument("--target_model", type=str, nargs="*", help="target model used for testing")
        parser.add_argument("--data_dir", type=str, help="data root", default="/home/mnt/points/data/Coseg_Wang/Coseg_Wang")
        parser.add_argument("--data_max", type=int, help="maximal number of instance to load", default=100)
        parser.add_argument("--data_cat", type=str, help="data category", default="*")
        parser.add_argument("--dim", type=int, help="2D or 3D", default=3)
        parser.add_argument("--log_dir", type=str, help="log directory", default="./log")
        parser.add_argument("--subdir", type=str, help="save to directory name", default="test")
        parser.add_argument("--batch_size", type=int, help="batch size", default=2)
        parser.add_argument("--blend_style", action="store_true", help="use alpha to control local style")
        parser.add_argument("--full_net", action="store_true", help="use network to predict conditioned source cage")
        # regularizations
        parser.add_argument("--eval", action="store_true", help="evaluatate every epoch")
        parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
        parser.add_argument("--loss", type=str, help="main reconstruction loss", choices=["LCD", "MSE", "CD"])
        parser.add_argument("--loss_weight", type=float, help="matching weight", default=1)
        parser.add_argument("--clap_weight", type=float, help="cage laplacian loss weight", default=0)
        parser.add_argument("--slap_weight", type=float, help="shape laplacian loss weight", default=0)
        parser.add_argument("--sedge_weight", type=float, help="shape edge length change loss weight", default=0)
        parser.add_argument("--slap_norm", action="store_true", help="use laplacian norm")
        parser.add_argument("--snormal_weight", type=float, help="shape normal loss weight", default=0)
        parser.add_argument("--p2f_weight", type=float, help="shape normal loss weight", default=0)
        parser.add_argument("--sfnormal_weight", type=float, help="shape face normal loss weight", default=0)
        parser.add_argument("--sstretch_weight", type=float, help="shape stretch loss weight", default=0)
        parser.add_argument("--cinside_weight", type=float, help="cage inside the shape loss weight", default=0)
        parser.add_argument("--cinside_eps", type=float, help="expand shape in normal direction by epsilon", default=0.1)
        parser.add_argument("--mvc_weight", type=float, help="negative weights penalize weight", default=0)
        parser.add_argument("--gravity_weight", type=float, help="center of cage == center of shape", default=0)
        parser.add_argument("--cshape_weight", type=float, help="use chamfer loss to enforce cage and shape align", default=0)
        parser.add_argument("--csmooth_weight", type=float, help="cage smoothness loss weight", default=0)
        parser.add_argument("--cshort_weight", type=float, help="cage short length loss weight", default=0)
        parser.add_argument("--cfangle_weight", type=float, help="cage face dihedral angle loss weight", default=0)
        parser.add_argument("--sym_weight", type=float, help="cage symmetry loss", default=0)
        parser.add_argument("--ground_weight", type=float, help="staying on the ground loss", default=0)
        parser.add_argument("--beta", type=float, help="weight controlling hausdorff", default=0)
        parser.add_argument("--gamma", type=float, help="weight controlling reverse chamfer distance", default=1.0)
        parser.add_argument("--delta", type=float, help="weight controlling scaled reverse chamfer distance", default=0)
        # training setup
        parser.add_argument("--nepochs", type=int, help="total number of epochs", default=50)
        parser.add_argument("--warmup_epochs", type=float, help="train deformer only before update cage", default=10)
        parser.add_argument("--phase", type=str, choices=["test", "train", "svr_test"], default="train")
        parser.add_argument("--ckpt", type=str, help="test model")
        parser.add_argument("--epoch", type=int, help="resume training from this epoch")
        parser.add_argument("--alternate_cd", action="store_true", help="altenately udpate net_c and net_d")
        parser.add_argument("--d_step", type=int, help="deformer epochs in alternating mode")
        parser.add_argument("--c_step", type=int, help="cage epochs in alternating mode")
        # network options
        parser.add_argument("--bottleneck_size", type=int, help="bottleneck size", default=512)
        parser.add_argument("--use_correspondence", action="store_true", help="use the correspondent xyz in shape decoder")
        parser.add_argument("--c_global", action="store_true", help="use the global code of the encoder")
        parser.add_argument("--optimize_template", action="store_true", help="network2 version, optimize cage parameters on the sphere")
        parser.add_argument("--deform_template", action="store_true", help="network2 version, deform cage parameters with a network")
        parser.add_argument("--pointnet2", action="store_true", help="use pointnet encoder")
        parser.add_argument("--concat_prim", action="store_true", help="concatenate template coordinate in every layer of decoder")
        parser.add_argument("--n_fold", type=int, help="3DN decoder (fold multiple times)", default=1)

        parser.add_argument("--disable_c_residual", dest="c_residual", action="store_false")
        parser.add_argument("--disable_d_residual", dest="d_residual", action="store_false")
        parser.add_argument("--use_init_cage", action="store_true", help="use pre-generated simplified mesh as cage")
        parser.add_argument("--normalization", type=str, choices=["batch", "instance", "none"], default="none")
        parser.add_argument("--disable_enc_code", dest="use_enc_code", action="store_false", help="concatenate encoder's code in decoder")
        parser.add_argument("--mlp", dest="atlas", action="store_false", help="use mlp type of network")
        parser.add_argument("--use_pretrained", action="store_true", help="use pretrained AtlasNet encoder for nc and nd")
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='cage deformation')
            parser = self.initialize(parser)

            # save and return the parser
            self.parser = parser
            # get the basic options
            opt, _ = self.parser.parse_known_args()

        return self.parser.parse_args()

    def print_options(self, opt, output_file=None):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.log_dir)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        if isinstance(output_file, str):
            with open(output_file, "a") as f:
                f.write(message)
                f.write('\n')
        elif isinstance(output_file, io.IOBase):
            output_file.write(message)
            output_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        if opt.phase == "test":
            assert(opt.ckpt is not None)
        opt.batch_size if opt.phase=="train" else 1
        if opt.normalization == "none":
            opt.normalization = None
        self.opt = opt
        if isinstance(opt.source_model, str):
            opt.source_model = [opt.source_model]
        if isinstance(opt.target_model, str):
            opt.target_model = [opt.target_model]

        if opt.source_model is not None and opt.target_model is not None:
            source_model = []
            target_model = []
            for source, target in itertools.product(opt.source_model, opt.target_model):
                source_model.append(source)
                target_model.append(target)
            opt.target_model = target_model
            opt.source_model = source_model
        # if source or target is given create permutations
        elif opt.source_model is not None:
            source_model = []
            target_model = []
            for source, target in itertools.product(opt.source_model, repeat=2):
                source_model.append(source)
                target_model.append(target)
            opt.target_model = target_model
            opt.source_model = source_model
        elif opt.target_model is not None:
            source_model = []
            target_model = []
            for source, target in itertools.product(opt.target_model, repeat=2):
                source_model.append(source)
                target_model.append(target)
            opt.target_model = target_model
            opt.source_model = source_model
        return self.opt


class DeformationOptions(BaseOptions):
    """
    This class defines options used for deformer_3d.
    """
    def parse(self):
        self.opt = self.gather_options()
        self.parser.set_defaults(source_model=None, template=None)
        # parser again with new defaults
        self.opt, _ = self.parser.parse_known_args()
        if self.opt.phase == "test":
            assert(self.opt.ckpt is not None)
        self.opt.batch_size if self.opt.phase=="train" else 1
        if self.opt.normalization == "none":
            self.opt.normalization = None
        # self.opt = super().parse()
        if self.opt.source_model is not None and not isinstance(self.opt.source_model, str):
            self.opt.source_model = self.opt.source_model[0]
        if isinstance(self.opt.target_model, str):
            self.opt.target_model = [self.opt.target_model]

        return self.opt
