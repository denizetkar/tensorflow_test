import multiprocessing as mp
from functools import reduce


def _sum_iterable(iterable):
    return reduce(lambda x, y: x + y, iterable)


def parallel_sum(pool, iterable):
    branch_num = mp.cpu_count()
    total_size = len(iterable)
    if total_size <= branch_num**2//2:
        return reduce(lambda x, y: x + y, iterable)

    chunk_size = (total_size - 1)//branch_num + 1
    chunk_indexes = list(range(0, total_size, chunk_size))
    chunk_indexes.append(total_size)
    num_of_threads = len(chunk_indexes) - 1

    chunks = []
    for i in range(num_of_threads):
        chunks.append(iterable[chunk_indexes[i]:chunk_indexes[i+1]])

    results = pool.map(_sum_iterable, chunks)
    return reduce(lambda x, y: x + y, results)
