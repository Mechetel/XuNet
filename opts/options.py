"""This module provides method to enter various input to the model training."""
import argparse


def arguments() -> str:
    """This function returns arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cover_path",
        default="/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/train"
        # default="~/data/GBRASNET/BOSSbase-1.01-div/cover/train"
    )
    parser.add_argument(
        "--stego_path",
        default="/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/train"
        # default="~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/train"
    )
    parser.add_argument(
        "--valid_cover_path",
        default="/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/val"
        # default="~/data/GBRASNET/BOSSbase-1.01-div/cover/val"
    )
    parser.add_argument(
        "--valid_stego_path",
        default="/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"
        # default="~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"
    )
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)

    opt = parser.parse_args()
    return opt
