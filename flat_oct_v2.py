#            0
#           f0
#    1    2      3      4
#      f2    f4      f6
#   f1    f3     f5
# 5    6     7     8
#        f7
#         9
import math
import numpy as np
from graph import *

# total 24 edges


# 0 ~ 2 ** len(edges)

# begin = 0  # 2 ** (len(edges) - 2)
# end = 2 ** len(edges) - 1


def worker(begin, end, edges, faces):
    part_size_vs_prob = np.zeros(shape=(len(faces)+1), dtype=np.float64)

    progress = 0.0

    for bits in permutation(begin=begin, end=end):
        progress += 1
        if int(progress) % 100000 == 0:
            print(100.0 * progress / (end - begin))

        for ibt, bit in enumerate(bits):
            edges[ibt].is_break = (bit == 1)

        complete_faces = {}
        for face_id, face in faces.items():
            if face.is_complete():
                complete_faces[face_id] = face
        parts = bfs(complete_faces)
        prob = 1.0
        for e in edges:
            prob *= e.prob()
        # print(bits, prob, complete_faces.keys(), parts)
        for p in parts:
            part_size = len(p)
            part_size_vs_prob[part_size] += prob
    return part_size_vs_prob


P = 0.01


def flat_oct():
    edges = [Edge(ei, P) for ei in range(24)]

    faces = {
        0: Face(0, [edges[0], edges[1], edges[3]], [4]),
        1: Face(1, [edges[6], edges[7], edges[18]], [2]),
        2: Face(2, [edges[2], edges[8], edges[9]], [1, 3]),
        3: Face(3, [edges[10], edges[11], edges[19]], [2, 4, 7]),
        4: Face(4, [edges[4], edges[12], edges[13]], [0, 3, 5]),
        5: Face(5, [edges[14], edges[15], edges[21]], [4, 6]),
        6: Face(6, [edges[5], edges[16], edges[17]], [5]),
        7: Face(7, [edges[20], edges[22], edges[23]], [3])
    }
    return edges, faces


def oct():
    edges = [Edge(ei, 0.01) for ei in range(12)]

    faces = {
        0: Face(0, [edges[0], edges[1], edges[4]], [1, 3, 4]),
        1: Face(1, [edges[1], edges[2], edges[5]], [0, 2, 5]),
        2: Face(2, [edges[2], edges[3], edges[6]], [1, 3, 6]),
        3: Face(3, [edges[0], edges[3], edges[7]], [0, 2, 7]),
        4: Face(4, [edges[4], edges[8], edges[9]], [0, 5, 7]),
        5: Face(5, [edges[5], edges[9], edges[10]], [1, 4, 6]),
        6: Face(6, [edges[6], edges[10], edges[11]], [2, 5, 7]),
        7: Face(7, [edges[7], edges[8], edges[11]], [3, 4, 6])
    }
    return edges, faces


def cube():
    edges = [Edge(ei, 0.01) for ei in range(12)]

    faces = {
        0: Face(0, [edges[0], edges[1], edges[2], edges[3]], [1, 2, 3, 4]),
        1: Face(1, [edges[1], edges[4], edges[5], edges[9]], [0, 2, 4, 5]),
        2: Face(2, [edges[2], edges[5], edges[6], edges[10]], [0, 1, 3, 5]),
        3: Face(3, [edges[3], edges[6], edges[7], edges[11]], [0, 2, 4, 5]),
        4: Face(4, [edges[0], edges[4], edges[7], edges[8]], [0, 1, 3, 5]),
        5: Face(5, [edges[8], edges[9], edges[10], edges[11]], [1, 2, 3, 4]),
    }
    return edges, faces


if __name__ == '__main__':
    from multiprocessing import Pool, freeze_support

    freeze_support()
    args = []

    N = 2
    n_processes = 2 ** N

    edges, faces = flat_oct()
    n_edges = len(edges)
    begin = 0  # 2 ** (len(edges) - 2)
    end = 2 ** n_edges
    step = 2 ** (n_edges - N)

    for i in range(n_processes):
        args.append((i * step, (i + 1) * step, edges, faces))

    pool = Pool(n_processes)
    all_result = []
    for i in range(n_processes):
        all_result.append(pool.apply_async(worker, args=args[i]))

    result_sum = np.zeros(shape=(len(faces) + 1, ), dtype=np.float64)
    for i in all_result:
        result_sum += i.get()

    print(result_sum)
