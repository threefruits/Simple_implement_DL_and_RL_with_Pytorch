import multiprocessing as mp

def job(b):
    res=0
    for i in range(b):
        res+=i
    return res




if __name__ == '__main__':
    pool = mp.Pool(processes=4)
    res = pool.map(job, range(10))
    print(res)