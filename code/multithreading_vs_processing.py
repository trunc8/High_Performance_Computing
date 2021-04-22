#!/usr/bin/env python3
# trunc8 did this
# Reference: https://engineering.contentsquare.com/2018/multithreading-vs-multiprocessing-in-python/

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
import time

def runCPUHeavyTask(x):
  print(f'Running task: {x}')
  start = time.time()
  count = 0
  for i in range(10**8):
    count += i
  print(f'Done: {x}')
  end = time.time()
  return end-start


def multithreading(func, args, workers):
  with ThreadPoolExecutor(workers) as ex:
    res = ex.map(func, args)
  return list(res)


def multiprocessing(func, args, workers):
  with ProcessPoolExecutor(workers) as ex:
    res = ex.map(func, args)
  return list(res)

def visualize(result, title):
  plt.figure()
  plt.bar(np.arange(1,workers+1), result)
  plt.xticks(np.arange(1,workers+1))
  plt.ylabel("Time taken (s)")
  plt.title(title)

workers = 4

mt_result = multithreading(runCPUHeavyTask, range(4), 4)

mp_result = multiprocessing(runCPUHeavyTask, range(workers), workers)

visualize(mt_result, "Multithreading")
visualize(mp_result, "Multiprocessing")

plt.show()