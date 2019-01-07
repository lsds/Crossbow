package uk.ac.imperial.lsds.crossbow.dispatcher;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.data.VirtualCircularDataBuffer;

public interface ITaskDispatcher {

	void dispatchNext (int taskid, int bound);
	
	void dispatch (Batch batch, Integer replicaId);

	VirtualCircularDataBuffer [] getBuffers ();
}
