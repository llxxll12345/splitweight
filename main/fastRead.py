import os.path
from multiprocessing import Pool
import sys
import time
import csv

data = {}

def process_file(name):
    linecount=0
    wordcount=0
    print("reading")
    with open(name, 'r') as inp:
        for line in inp:
            linecount+=1
            items = line.split(',')
            wordcount+=len(items)
            x = [float(item) for item in items]
            data.update({name, x})
    print("{0}, {1}, {2}".format(name, linecount, wordcount))
    return name, linecount, wordcount

def process_files_parallel(names):
    pool=Pool()
    results=pool.map(process_file, names)

def process_files(arg, dirname, names):
    results=map(process_file, [os.path.join(dirname, name) for name in names])

if __name__ == '__main__':
    #start=time.time()
    #os.path.walk('input/', process_files, None)
    #print("process_files()", time.time()-start)

    start=time.time()
    names = []
    with open('csv_output/format.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            input_name = 'csv_output/' + name + '.csv'
            print(input_name)
            names.append(input_name)

    process_files_parallel(names)

    print("process_files_parallel()", time.time()-start)
    for (k, v) in data.items():
        print(k, len(v))