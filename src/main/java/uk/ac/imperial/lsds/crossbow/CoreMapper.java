package uk.ac.imperial.lsds.crossbow;

import java.util.Arrays;

import uk.ac.imperial.lsds.crossbow.types.HandlerType;

public class CoreMapper {
	
	private int [] offset;
	
	private boolean planned;
	
	public CoreMapper () {
		offset = new int [3];
		Arrays.fill(offset, -1);
		planned = false;
	}
	
	public CoreMapper plan () {
		
		int pivot;
		
		if (SystemConf.getInstance().getGPU()) {
			
			/* Core #0 is reserved for the task dispatcher and core #1 is reserved for the GPU worker thread.
			 * Subsequent core are reserved for CPU worker threads, if any.
			 */
			pivot = SystemConf.getInstance().numberOfWorkerThreads() + 1;
			
			/* Configure offset for task handlers */
			if (SystemConf.getInstance().numberOfGPUTaskHandlers() > 0) {
				offset[0] = pivot;
				pivot += SystemConf.getInstance().numberOfGPUTaskHandlers();
			}
			
			/* Configure offset for callback handlers */
			// offset[1] = pivot;
			// pivot += SystemConf.getInstance().numberOfGPUCallbackHandlers();
			
			/* Configure offset for dataset handlers */
			// offset[2] = pivot;
			
			/* Configure offset for callback handlers */
			offset[2] = pivot;
			pivot += SystemConf.getInstance().numberOfFileHandlers();
			
			/* Configure offset for dataset handlers */
			offset[1] = pivot;
		}
		
		return this;
	}
	
	public int getOffset (HandlerType handler) {
		if (! planned) {
			plan ();
			planned = true;
		}
		return offset[handler.getId()];
	}
	
	public int     getTaskHandlerOffset () { return getOffset (HandlerType.TASK     ); }
	public int getCallbackHandlerOffset () { return getOffset (HandlerType.CALLBACK ); }
	public int  getDatasetHandlerOffset () { return getOffset (HandlerType.DATASET  ); }
	
	public void dump () {
		
		return;
	}
}
