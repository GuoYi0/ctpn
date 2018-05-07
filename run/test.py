from network.test_network import get_test_network
from lib.load_config import load_config
from ctpn.test_net import TestClass
import tensorflow as tf
from run.evaluation import evaluate

if __name__ == "__main__":
    # 加载配置文件
    cfg = load_config()
    # pprint.pprint(cfg)
    with tf.Graph().as_default() as g:
        # 获取测试网络， 一个空网络
        with g.device('/gpu:0'):
            network = get_test_network(cfg)
            testclass = TestClass(cfg, network)
            testclass.test_net(g)
    evaluate()

