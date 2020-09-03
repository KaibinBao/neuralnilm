from __future__ import print_function
from threading import Thread, Event

import sys
if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue

from queue import Queue, Empty


class DataThread(Thread):
    def __init__(self, data_pipeline, max_queue_size=8, **get_batch_kwargs):
        super(DataThread, self).__init__(name='neuralnilm-data-process')
        self._stopevent = Event()
        self._queue = Queue(maxsize=max_queue_size)
        self.data_pipeline = data_pipeline
        self._get_batch_kwargs = get_batch_kwargs

    def run(self):
        while not self._stopevent.is_set():
            batch = self.data_pipeline.get_batch(**self._get_batch_kwargs)
            self._queue.put(batch)

    def get_batch(self, timeout=30):
        if self.is_alive():
            return self._queue.get(timeout=timeout)
        else:
            raise RuntimeError("Process is not running!")

    def stop(self):
        self._stopevent.set()
        try:
            self._queue.get(block=False)
        except Empty:
            pass
        self.join()

class DataThread2(Thread):
    def __init__(self, data_pipelines, max_queue_size=8):
        super(DataThread2, self).__init__(name='neuralnilm-data-process')
        self._stopevent = Event()
        self._queue = Queue(maxsize=max_queue_size)
        self.data_pipelines = data_pipelines
        self.counter = 0

    def run(self):
        while not self._stopevent.is_set():
            batch = self.data_pipelines[self.counter].get_batch()
            self._queue.put(batch)
            self.counter = (self.counter+1) % len(self.data_pipelines)

    def get_batch(self, timeout=30):
        if self.is_alive():
            return self._queue.get(timeout=timeout)
        else:
            raise RuntimeError("Process is not running!")

    def stop(self):
        self._stopevent.set()
        try:
            self._queue.get(block=False)
        except Empty:
            pass
        self.join()