package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

import uk.ac.imperial.lsds.crossbow.utils.BaseObjectPoolImpl;
import uk.ac.imperial.lsds.crossbow.utils.IObjectPool;

public class TestTaskQueue {
	
	static class Monitor implements Runnable {
		
		private ITaskQueue<Example> queue;
		private Worker [] workers;
		
		public Monitor (ITaskQueue<Example> queue, Worker [] workers) {
			
			this.queue = queue;
			this.workers = workers;
		}
		
		public void run () {
			
			int _queue_;
			
			long time, _time = 0L;
			long delta;
			
			long tasks, _tasks = 0;
			
			double Dt, tps;
			
			while (true) {
				
				try {
					Thread.sleep(1000L);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				
				time = System.currentTimeMillis();
				
				delta = time - _time;
				
				/* Find tasks processed */
				tasks = 0;
				for (int i = 0; i < workers.length; ++i)
					tasks += workers[i].getTasksProcessed();
				
				// System.out.println(String.format("%13d tasks processed", tasks));
				
				tps = 0;
				
				if (_tasks > 0) {

					Dt  = (delta / 1000.0);
					tps = (double) (tasks - _tasks) / Dt;
				}
				
				_tasks = tasks;
				_time  = time;
				
				_queue_ = queue.size();
			
				System.out.println(String.format("[MON] q %13d %13.3f tasks/s", _queue_, tps));
			}
		}
	}
	
	static class Worker implements Runnable {
		
		private int id;
		
		private long duration;
		
		private ITaskQueue<Example> queue;
		private CountDownLatch counter;
		
		private volatile boolean stop = false;
		
		private CountDownLatch signal = null;
		
		private AtomicLong count = new AtomicLong(0);
		
		public Worker (int id, long duration, ITaskQueue<Example> queue, CountDownLatch counter) {
			
			this.id = id;
			this.duration = duration;
			this.queue = queue;
			this.counter = counter;
		}
		
		public long getTasksProcessed() {
			return count.get();
		}

		public void run () {
			
			Example task = null;
			
			retry: while (! stop) {
				
				while ((task = queue.poll()) == null) {
					LockSupport.parkNanos(1L);
					if (stop)
						continue retry;
				}
				
				/* Do something */
				if (duration > 0) {
					try {
						Thread.sleep(duration);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
				
				count.incrementAndGet();
				
				counter.countDown();
				
				task.free();
			}
			
			System.out.println(String.format("Worker %2d shuts down", id));
			
			signal.countDown();
			return;
		}
		
		public void shutdown (CountDownLatch signal) {
			
			this.signal = signal;
			stop = true;
		}
	}
	
	public static void main (String [] args) throws InterruptedException {
		
		int N = 8;
		int M = 1000000000; /* 1 billion tasks */
		
		long duration = 0L;
		
		int type = 3;
		
		ITaskQueue<Example> queue;
		
		if (1 == type)
		{
			queue = new LockBasedTaskQueueImpl<Example>(0);
		}
		else 
		if (2 == type)
		{	
			queue = new LazyLockBasedTaskQueueImpl<Example>();
		}
		else 
		if (3 == type)
		{
			queue = new LockFreeTaskQueueImpl<Example>();
		}
		else
		{
			queue = new BaseTaskQueueImpl<Example>();
		}
		
		Executor executor = Executors.newCachedThreadPool();
		
		Worker [] workers = new Worker [N];
		
		ExampleFactory factory = new ExampleFactory();
		
		IObjectPool<Example> pool = new BaseObjectPoolImpl<Example>(factory);
		
		CountDownLatch counter = new CountDownLatch (M - 1);
		
		for (int i = 0; i < N; ++i) {
			
			workers[i] = new Worker (i, duration, queue, counter);
			
			executor.execute(workers[i]);
		}
		
		Thread monitor = new Thread (new Monitor (queue, workers));
		monitor.start();
		
		long start = System.nanoTime();
		
		/* Now, populate the task queue */
		
		Example example;
		
		for (int id = 1; id < M; ++id) {
			
			example = pool.getInstance();
			example.setId(id);
			example.setPool(pool);
			
			/*
			while (queue.size() > 1000)
				;
			*/
			if (! queue.add(example))
				throw new IllegalStateException("error: failed to add task");
			
			/* LockSupport.parkNanos(1L); */
		}
		
		System.out.println("Waiting for counter...");
		
		counter.await();
		
		long dt = System.nanoTime() - start;
		System.out.println(String.format("dt = %.5f msecs\n", (double) dt / 1000000D));
		
		System.out.println("Shutting down...");
		
		CountDownLatch signal = new CountDownLatch (N);
		
		for (int i = 0; i < N; i++)
			workers[i].shutdown(signal);
		
		signal.await();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
