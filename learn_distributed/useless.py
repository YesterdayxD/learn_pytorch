import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('echo')  # add_argument()指定程序可以接受的命令行选项
parser.add_argument('--rank', '-r', required=True, type=int)
parser.add_argument('--local_rank', '-loc_r', required=True, type=int)
parser.add_argument('--init_method', '-im', required=True, type=str)
parser.add_argument('--epoch', '-e', required=False, default=10, type=str)
args = parser.parse_args()  # parse_args()从指定的选项中返回一些数据
print(args)
print(args.rank)
print(args.epoch)
