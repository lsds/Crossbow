package uk.ac.imperial.lsds.crossbow.task;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public interface ITask {

	public void outputBatchResult (Batch batch);

	public boolean isValidationTask ();
	
	public Phase getPhase ();

	public boolean isGPUTask ();

	public Operator getPrevious (Operator operator);
	
	public boolean isMostDownstream (Operator operator);
	
	public boolean isMostUpstream (Operator operator);
}
