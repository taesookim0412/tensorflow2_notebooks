import multiprocessing
import sys
from functools import partial
from time import sleep, time

def workerFunction(msg, delay):
    sleep(delay)
    print(f"{msg} {delay} seconds")
    sys.stdout.flush()


def createJobs():
    st = time()

    delays = [10, 20, 30, 40]
    func = partial(workerFunction, "I waited ")
    with multiprocessing.Pool(4) as p:
        start = 2
        for i, _ in enumerate(p.imap_unordered(func, delays[:start])):
            last = p.apply_async(func=workerFunction, args=("I waited", delays[start+i]))
            print(time() - st)
        while not last.ready():
            last.wait(1)
        print(f"Done! {time() - st}s passed")

def createJobs2():
    st = time()

    delays = [10, 30, 10, 20]
    func = partial(workerFunction, "I waited ")
    with multiprocessing.Pool(4) as p:
        start = 2
        for i, _ in enumerate(p.imap_unordered(func, delays[:start])):
            last = p.apply_async(func=workerFunction, args=("I waited", delays[start+i]))
            print(time() - st)
        while not last.ready():
            last.wait(1)
        print(f"Done! {time() - st}s passed")

def createJobs3():
    st = time()

    delays = [10, 9, 20, 5]
    func = partial(workerFunction, "I waited ")
    with multiprocessing.Pool(4) as p:
        start = 2
        for i, _ in enumerate(p.imap_unordered(func, delays[:start])):
            last = p.apply_async(func=workerFunction, args=("I waited", delays[start+i]))
            print(time() - st)
        while not last.ready():
            last.wait(1)
        print(f"Done! {time() - st}s passed")


def createJobs4():
    st = time()

    delays = [10, 9, 20, 5]
    func = partial(workerFunction, "I waited ")
    with multiprocessing.Pool(4) as p:
        start = 2
        createdJobs = []
        for i, _ in enumerate(p.imap_unordered(func, delays[:start])):
            createdJobs.append(p.apply_async(func=workerFunction, args=("I waited", delays[start+i])))
            print(time() - st)
        for job in createdJobs:
            while not job.ready():
                job.wait(1)
        print(f"Done! {time() - st}s passed")




