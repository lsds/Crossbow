package uk.ac.imperial.lsds.crossbow;

import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;
import uk.ac.imperial.lsds.crossbow.dispatcher.ITaskDispatcher;
import uk.ac.imperial.lsds.crossbow.task.TaskFactory;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class PerformanceMonitor implements Runnable {

	private final static Logger log = LogManager.getLogger (PerformanceMonitor.class);
	
	private ExecutionContext context;
	
	private AtomicBoolean exit;
	
	private long time, _time = 0L;
	private long dt;

	private Dataflow dataflow;
	private int size;

	private Measurement [] measurements;

	public PerformanceMonitor (ExecutionContext context) {
		
		this.context = context;
		
		exit = new AtomicBoolean (false);

		dataflow = context.getDataflow(Phase.TRAIN);
		size = dataflow.numberOfSubGraphs();
		
		measurements = new Measurement [size];
		for (int i = 0; i < size; ++i)
			measurements[i] = null;
		
		SubGraph next = dataflow.getSubGraph();
		while (next != null) {
			log.info(String.format("Monitor %s", next.getName()));
			measurements[next.getId()] = new Measurement (next.getId(), (next.isMostUpstream()) ? dataflow.getTaskDispatcher() : next.getTaskDispatcher());
			next = next.getNext();
		}
	}
	
	public double getCurrentThroughput (int id) {
		
		return measurements[id].window.getLastThroughput ();
	}
	
	public void stop () {
		exit.set(true);
	}
	
	public void run () {
		
		long interval = SystemConf.getInstance().getPerformanceMonitorInterval ();

		while (! exit.get()) {

			try { 
				Thread.sleep(interval); 
			} catch (Exception e) {

				e.printStackTrace();
				System.exit(1);
			}

			time = System.currentTimeMillis();

			StringBuilder b = new StringBuilder("[MON]");

			dt = time - _time;

			for (int i = 0; i < size; i++)
				if (measurements[i] != null)
					b.append(measurements[i].info(dt));

			int queuesize = context.getExecutorQueueSize();
			b.append(String.format(" q %6d", queuesize));

			/* Append factory sizes */
			b.append(String.format(" t %6d", TaskFactory.count.get()));
			b.append(String.format(" w %6d", BatchFactory.count.get()));
			/*
			 * b.append(String.format(" b %6d", DataBufferFactory.count.get()));
			 * 
			 * TODO
			 * 
			 * Now that data buffers come from individual pools, monitoring 
			 * allocations would require traversing the dataflow graph.
			 */

			System.out.println(b);

			_time = time;
		}
		
		log.info("Performance monitor shuts down");
	}
	
	class Measurement {
		
		int id;
		
		ITaskDispatcher dispatcher;
		VirtualCircularDataBuffer [] buffer;
		
		long bytes, _bytes = 0;
		double Dt, MBps;
		double MB, _1MB_ = 1048576.0;
		
		double examplespersecond;
		long tasks, _tasks = 0;
		
		int batchsize;
		
		private PerformanceMonitorQueue window;

		public Measurement (int id, ITaskDispatcher dispatcher) {

			this.id = id;
			this.dispatcher = dispatcher;
			this.buffer = dispatcher.getBuffers();
			
			this.window = new PerformanceMonitorQueue (10, true);
			
			this.batchsize = ModelConf.getInstance().getBatchSize();
		}
		
		@Override
		public String toString () {
			return null;
		}

		public String info (long delta) {

			String s = "";

			if (buffer == null)
				return "";
			
			bytes = 0;
			tasks = 0;
			
			for (int i = 0; i < buffer.length; ++i)
				bytes += buffer[i].getBytesProcessed();
			
			tasks += buffer[0].getTasksProcessed();

			if (_bytes > 0) {

				Dt = (delta / 1000.0);
				MB = (bytes - _bytes) / _1MB_;
				MBps = MB / Dt;
				examplespersecond = (tasks - _tasks) * batchsize / Dt; 

				s = String.format(" S%03d %10.3f examples/s %10.3f MB/s", id, examplespersecond, MBps);
				
				/* Append measurement to queue (for auto-tuning) */
				window.add(MBps);
			}
			
			_bytes = bytes;
			_tasks = tasks;

			return s;
		}
	}
}
