import os
import time 
import csv
import threading
from concurrent.futures import ThreadPoolExecutor

data = {}

def get_chunks(file_size):
    chunk_start = 0
    chunk_size = 0x20000
    while chunk_start + chunk_size < file_size:
        yield(chunk_start, chunk_size)
        chunk_start += chunk_size
    
    final_size = file_size - chunk_start
    yield(chunk_start, final_size)


def read_file_chunked(file_path):
    with open(file_path) as f:
        file_size = os.path.getsize(file_path)
        print('File size: {}'.format(file_size))
        progress = 0
        res = ""
        for chunk_start, chunk_size in get_chunks(file_size):
            file_chunk = f.read(chunk_size)
            res += file_chunk
            progress += len(file_chunk)
            '''
            print('{0} of {1} bytes read ({2}%)'.format(
                progress, file_size, int(progress / file_size * 100)
            ))
            '''
        return res

def process(sublist):
    return [float(x) for x in sublist]

def do_read():
    names = []
    with open('csv_output/format.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start=time.time()
            name = row['name']
            input_name = 'csv_output/' + name + '.csv'
            print(input_name)
            res = read_file_chunked(input_name)
            '''
            l = len(res)
            splitted = res.split(',')
            mylist = []
            cur_sz = 0
            inc = 2000
            futures = []
            while cur_sz + inc < l:
                mylist.append(splitted[cur_sz: cur_sz + inc])
                cur_sz += inc
            mylist.append(splitted[cur_sz:])
            executor = ThreadPoolExecutor(max_workers=100)
            for lists in mylist:
                futures.append(executor.submit(process, lists))
            print("Appended.")
            mylist = []
            for future in futures:
                mylist.extend(future.result())
            print("Len: ", len(mylist))
            data.update({name: mylist})
            '''
            print("time used", time.time()-start)

    for (k, v) in data.items():
        print(k, len(v))

if __name__ == '__main__':
    do_read()