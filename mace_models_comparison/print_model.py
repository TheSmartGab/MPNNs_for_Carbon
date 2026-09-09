from argparse import ArgumentParser
import torch


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inputfile", "-i")

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    model = torch.load(args.inputfile, weights_only = False)

    print(model)

    return 0

if __name__ == "__main__":
    main()
