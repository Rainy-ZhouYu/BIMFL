import argparse

def get_args():
    parser = argparse.ArgumentParser()
    ## federated learning
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")#选择客户端的比例
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")#本地的更新次数
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")#本地批量大小
    parser.add_argument('--bs', type=int, default=128, help="test batch size")#测试的批量大小
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")#学习率
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")#SGD动量
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')#模型
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')#每种内核的数量
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')#内核大小
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_false', default=True, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")  # 图像通道数
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')  # 提前结束轮次
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    ## deep reinforcement learning
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--observation_dim', type=int, default=10)
    parser.add_argument('--data_size', type=str, default=[25, 9, 32, 6, 45, 20, 1, 12, 6, 11], help='The data quantities of clients')
    parser.add_argument('--energy_consumption', type=str, default=[0.4797846374733109, 0.10318370487752862, 0.24691822115565296, 0.1867959886387748,
                         0.34249845996578077, 0.168317423981345, 0.06423717394565366, 0.21075237584492076,
                         0.2642483615848909, 0.31922299207716565], help='The energy_consumption of clients')
    parser.add_argument('--agent_num', type=int, default=10, help='the number of agents')
    parser.add_argument('--action_dim', type=int, default=1, help='the demision of agents')
    parser.add_argument('--psi', type=int, default=0.5, help='the Hyper-parameters')
    # Hyper-parameters
    return args