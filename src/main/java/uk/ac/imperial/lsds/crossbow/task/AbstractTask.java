package uk.ac.imperial.lsds.crossbow.task;

import java.util.concurrent.atomic.AtomicMarkableReference;

import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public abstract class AbstractTask implements ITask {
	
	protected long p1, p2, p3, p4, p5, p6, p7;
	
	public int taskId;
	public int graphId;
	
	public AtomicMarkableReference<AbstractTask> next;
	
	protected boolean GPU = false;
	
	protected boolean validate = false;
	
	protected boolean usesRecords = false;
	
	public int lowerBound;
	
	public ModelAccess access;
	
	public Integer replicaId;
	
	public abstract int run ();
	
	public abstract void free ();
	
	public void setGPU (boolean GPU) {
		this.GPU = GPU;
	}
	
	public boolean isGPUTask () {
		return GPU;
	}
	
	public void setValidationTask (boolean validate) {
		this.validate = validate;
	}
	
	public boolean isValidationTask () {
		return validate;
	}
	
	public void setRecordDatasetUse (boolean useRecords) {
		this.usesRecords = useRecords;
	}
	
	public boolean usesRecordDataset () {
		return usesRecords;
	}
	
	public Phase getPhase () {
		return (isValidationTask () ? Phase.CHECK : Phase.TRAIN);
	}
	
	public String toString() {
		return String.format("[%5s task %4d graph %2d replica %s barrier %4d]", 
				(validate) ? "test" : "train", 
				taskId, 
				graphId, 
				(replicaId == null) ? "null" : replicaId.toString(),
				lowerBound);
	}
}
