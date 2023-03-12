
RES_LEVEL = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FPS_LEVEL = [30, 25, 20, 15, 10, 5, 2, 1]
QP_LEVEL = [20, 24, 28, 32, 36, 40, 44, 48]

def gen_new_config(res, fps, qp):
    new_res = RES_LEVEL.index(res)
    new_fps = FPS_LEVEL.index(fps)
    new_qp = QP_LEVEL.index(qp)

    return new_res, new_fps, new_qp