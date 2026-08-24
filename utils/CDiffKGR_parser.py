# -*- coding: utf-8 -*-
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CDiffKGR with conditional Triple Diffusion Model (TDM)")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon-book",
                        help="Choose a dataset:[last-fm,alibaba-ifashion,yelp2018,mind-f,amazon-book,MIND]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=300, help='number of recommendation epochs')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--flag_step', type=int, default=3, help='early-stopping patience')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size (recommendation)')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='drop threshold')
    parser.add_argument('--lr_dc_step', type=float, default=100, help='drop threshold')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='drop threshold')
    parser.add_argument('--max_iter', type=float, default=2, help='iteration times')
    parser.add_argument("--inverse_r", type=bool, default=False, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=1, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=5, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 10]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}')

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--num_neg_sample', type=int, default=1, help='the number of negative sample')
    parser.add_argument('--margin', type=float, default=0.2, help='the margin of contrastive_loss')
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, contrastive_loss]")
    parser.add_argument('--denoiser_reg_weight', type=float, default=0.001,
                        help='weight of generated-KG denoiser regularization loss')
    parser.add_argument('--relation_diversity_weight', type=float, default=0.1,
                        help='weight of relation diversity loss')

    # ===== TDM diffusion model ===== #
    parser.add_argument('--use_tdm', type=bool, default=True, help='use TDM diffusion model')
    parser.add_argument('--tdm_epochs', type=int, default=250, help='TDM training epochs')
    parser.add_argument('--tdm_steps', type=int, default=1000, help='diffusion steps')
    parser.add_argument('--tdm_rebuild_k', type=int, default=10000, help='max triplets per item')
    parser.add_argument('--tdm_max_triplets', type=int, default=50000000, help='max total generated triplets')
    parser.add_argument('--tdm_margin', type=float, default=10.0, help='margin for TDM loss')
    parser.add_argument('--original_relation_num', type=int, default=9,
                        help='number of original KG relations before adding the interaction relation')
    parser.add_argument('--relation_id_offset', type=int, default=1,
                        help='relation index offset after reserving relation 0 for user-item interaction')
    parser.add_argument('--k_per_rel', type=int, default=4,
                        help='nearest tail entities retained per item-relation pair before filtering')
    parser.add_argument('--global_distance_threshold', type=float, default=None,
                        help='fixed global distance threshold; None keeps percentile filtering')
    parser.add_argument('--global_distance_percentile', type=float, default=30,
                        help='distance percentile used when global_distance_threshold is None')

    # ===== Generated KG generation frequency ===== #
    parser.add_argument('--kg_gen_freq', type=int, default=1,
                        help='frequency (epochs) to regenerate the generated KG')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    return parser.parse_args()