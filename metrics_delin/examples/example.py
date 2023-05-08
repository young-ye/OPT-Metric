# ================================================================
#   Filename     : example.py
#
# ================================================================
import os
import sys
import imageio
import numpy as np
import glob
import metrics_delin as md
import time
import os

start = time.time()
# num_false_set = [3]
num_false_set = [1, 3, 5, 7, 9]
for num_false in num_false_set:
    save_path = rf'F:\u\result\metric12\opt\mix\mix_{num_false}_result.txt'  # 存放记录的文件
    gt_dir = r'F:\u\data_newmetric\opt\new_59\gt_resample'
    out_dir = rf'F:\u\data_newmetric\opt\new_59\mix_resample\{num_false}'  # 金标准

    for G_gt_path in glob.glob(os.path.join(gt_dir, '*.swc')):  # 读所有的swc文件
        num = os.path.splitext(os.path.split(G_gt_path)[-1])[0]
        out_swc_name = num + ".swc"
        G_pred_path = os.path.join(out_dir, out_swc_name)
        print('fname的值为：{}'.format(num))

        start = time.time()

        ##r4096 = (-4096, -4096)
        r4096 = (-4096, -4096, -4096)
        SHIFTS = {
            "toronto": r4096,
            "la": r4096,
            "new york": r4096,
            ##"boston": (4096, -4096),
            "boston": (4096, -4096, -4096),
            ##"chicago": (-4096, -8192),
            "chicago": (-4096, -8192, -4096),
            "amsterdam": r4096,
            "denver": r4096,
            "kansas city": r4096,
            "montreal": r4096,
            "paris": r4096,
            "pittsburgh": r4096,
            "saltlakecity": r4096,
            "san diego": r4096,
            "tokyo": r4096,
            "vancouver": r4096,
            ##"columbus": (-4096, -8192),
            "columbus": (-4096, -8192, -4096),
            ##"minneapolis": (-4096, -8192),
            "minneapolis": (-4096, -8192, -4096),
            ##"nashville": (-4096, -8192)}
            "nashville": (-4096, -8192, -4096)}

        print("loading the graphs")  # 加载图片
        G_gt = md.load_graph_txt(G_gt_path)
        G_pred = md.load_graph_txt(G_pred_path)
        city = "vancouver"

        # --------------------------------------------------opt_j
        f1, precision, recall, \
        tp, pp, ap, \
        matches_g, matches_hg, \
        g_gt_snap, g_pred_snap = md.opt_j(G_gt,
                                          G_pred,
                                          th_existing=1,  # 在捕获过程中，只有当该边的所有端点都不在th_existing范围内时，才会将一个附加节点插入到该边中
                                          th_snap=25,  # 如果一个点到最近的边的距离小于th_snap，那么它就被折入图中
                                          alpha=100)  # 鼓励匹配具有相似顺序的两个节点
        with open(save_path, 'a') as f:  # 设置文件对象
            print(
                "OPT-J:          filename:{} precision={:0.3f} recall={:0.3f} f1={:0.3f}\n".format(num, precision,
                                                                                                   recall,
                                                                                                   f1), file=f)
        print(
            "OPT-J:          filename:{} precision={:0.3f} recall={:0.3f} f1={:0.3f}\n".format(num, precision,
                                                                                               recall, f1))
        # --------------------------------------------------
        n_conn_precis, n_conn_recall, \
        n_inter_precis, n_inter_recall, \
        con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)
        with open(save_path, 'a') as f:  # 设置文件对象
            print(
                "OPT-P:          filename:{} con_prob_precis={:0.3f} con_prob_recall={:0.3f} con_prob_f1={:0.3f}\n".format(
                    num,
                    con_prob_precis,
                    con_prob_recall,
                    con_prob_f1), file=f)
        # print(
        #     "OPT-P:          filename:{} con_prob_precis={:0.3f} con_prob_recall={:0.3f} con_prob_f1={:0.3f}\n".format(num,
        #                                                                                                                con_prob_precis,
        #                                                                                                                con_prob_recall,
        #                                                                                                                con_prob_f1))

        # --------------------------------------------------
        # f1, spurious, missings, \
        # n_preds_sum, n_gts_sum, \
        # n_spurious_marbless_sum, \
        # n_empty_holess_sum = md.holes_marbles(G_gt, G_pred,
        #                                       spacing=10,
        #                                       dist_limit=300,
        #                                       dist_matching=25,
        #                                       N=50,  # to speed up this script
        #                                       verbose=False)
        # print("Hole-Marbles:   spurious={:0.3f} missings={:0.3f} f1={:0.3f}\n".format(spurious, missings, f1))

        # --------------------------------------------------
        f1, spurious, missings, \
        n_preds_sum, n_gts_sum, \
        n_spurious_marbless_sum, \
        n_empty_holess_sum = md.opt_g(G_gt, G_pred,
                                      spacing=10,
                                      dist_limit=300,
                                      dist_matching=25,
                                      N=50,  # to speed up this script
                                      verbose=False)
        with open(save_path, 'a') as f:  # 设置文件对象
            print("OPT-G:          filename:{} spurious={:0.3f} missings={:0.3f} f1={:0.3f}\n".format(num, spurious,
                                                                                                      missings,
                                                                                                      f1), file=f)
        # print("OPT-G:          filename:{} spurious={:0.3f} missings={:0.3f} f1={:0.3f}\n".format(num, spurious, missings,
        #                                                                                           f1))

        # --------------------------------------------------
        end = time.time()
        print("循环运行时间:%f秒" % (end - start))
        localtime = time.asctime(time.localtime(time.time()))
        print("本地时间为：{}\n".format(localtime))
