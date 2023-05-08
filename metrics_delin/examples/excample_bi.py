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
# num_false_set = [1, 5, 10, 15, 20, 25, 30, 35, 40]
num_false_set = [40]
# num_false_set = [6, 9, 12, 15]
#, 18, 21, 24, 27, 30]
for num_false in num_false_set:
    save_path = rf'G:\result\metric9\10000\opt\bi\bi_{num_false}_result.txt'  # 存放记录的文件
    pre_dir = r'G:\data_newmetric\opt\metric9\10000\gt'
    gt_dir = rf'G:\data_newmetric\opt\metric9\10000\bi\{num_false}'  # 金标准

    for G_gt_path in glob.glob(os.path.join(gt_dir, '*.swc')):  # 读所有的swc文件
        num = os.path.splitext(os.path.split(G_gt_path)[-1])[0]
        print('fname的值为：{}'.format(num))
        out_swc_name = f'{num}.swc'
        print(out_swc_name)
        G_pred_path = os.path.join(pre_dir, out_swc_name)


    # tmp_list = glob.glob(os.path.join(pre_dir, '*.swc'))
    # for G_pred_path in tmp_list:  # 读所有的swc文件
    #     num = os.path.splitext(os.path.split(G_pred_path)[-1])[0]
    #     print('fname的值为：{}'.format(num))
    #     out_swc_name = f'{num}.swc'
    #     G_gt_path = os.path.join(gt_dir, out_swc_name)

        # for G_gt_path in glob.glob(os.path.join(gt_dir, '*.swc')):  # 读所有的swc文件
        #     num = os.path.splitext(os.path.split(G_gt_path)[-1])[0]
        #     out_swc_name = num + ".swc"
        #     G_pred_path = os.path.join(out_dir, out_swc_name)
        #     print('fname的值为：{}'.format(num))

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
        # scale = 1/4
        # segments = np.array([[G_gt.nodes[s]['pos'], G_gt.nodes[t]['pos']] for s,t in G_gt.edges()])
        # gt_s = md.render_segments(segments*scale,
        #                           filename=None,
        #                           height=h*scale,
        #                           width=w*scale,
        #                           thickness=1)
        #
        # segments = np.array([[G_pred.nodes[s]['pos'], G_pred.nodes[t]['pos']] for s,t in G_pred.edges()])
        # pred_s = md.render_segments(segments*scale,
        #                             filename=None,
        #                             height=h*scale,
        #                             width=w*scale,
        #                             thickness=1)
        #
        # corr, comp, qual, TP_g, TP_p, FN, FP = md.corr_comp_qual(gt_s,
        #                                                          pred_s,
        #                                                          slack=8*scale)
        # print("Corr-Comp-Qual: corr={:0.3f} comp={:0.3f} qual={:0.3f}\n".format(corr, comp, qual))
        #
        # # --------------------------------------------------
        # correct, too_long, too_short, infeasible = md.toolong_tooshort(G_gt, G_pred,
        #                                                                n_paths=50, # to speed up this script
        #                                                                max_node_dist=25)
        # print("2Long-2Short:   correct={:0.3f} 2l+2s={:0.3f} inf={:0.3f}\n".format(correct, too_long+too_short, infeasible))

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
