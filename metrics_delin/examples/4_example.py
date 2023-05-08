'''改多进程让p跑起来'''
import metrics_delin as md
import glob, os
import time
import multiprocessing


def main():
    save_path = rf"F:\u\result\metric12\opt\mix\mix_result.txt"
    pre_path = rf'F:\u\data_newmetric\opt\new_59\pre_opt_topu\120'
    gt_path = rf'F:\u\data_newmetric\opt\new_59\gt_opt_topu\120'

    cores = int(multiprocessing.cpu_count() * 0.8)  # multiprocessing.cpu_count()
    print(cores)
    pool = multiprocessing.Pool(processes=cores)
    res = []

    for G_pred_path in glob.glob(os.path.join(pre_path, '*.swc')):
        print(G_pred_path)
        num = G_pred_path.split('\\')[-1]
        pre_brain = 11
        G_gt_path = os.path.join(gt_path, num)
        print(G_gt_path)

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

        print('optp')

        # n_conn_precis, n_conn_recall, \
        # n_inter_precis, n_inter_recall, \
        # con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)

        res.append(pool.apply_async(md.opt_p, args=(pre_brain, num, save_path, G_gt, G_pred)))

        # with open(save_path, 'a') as f:  # 设置文件对象
        #     print(
        #         f"{pre_brain}_OPT-P:filename_{num}_con_prob_precis={con_prob_precis:0.3f} con_prob_recall={con_prob_recall:0.3f} con_prob_f1_{con_prob_f1:0.3f}",
        #         file=f)

        end = time.time()
        print("循环运行时间:%f秒" % (end - start))
        localtime = time.asctime(time.localtime(time.time()))
        print("本地时间为：{}\n".format(localtime))

    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用


if __name__ == '__main__':
    main()
