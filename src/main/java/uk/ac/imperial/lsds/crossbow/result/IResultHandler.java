package uk.ac.imperial.lsds.crossbow.result;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

import uk.ac.imperial.lsds.crossbow.model.ModelGradient;

public interface IResultHandler {
	
	public IResultHandler setup ();
	
	public int getNext ();
	public boolean ready (int next);
	
	public void setSlot (int taskid, long [] free, float loss, float accuracy, ModelGradient gradient, boolean GPU);
	public void freeSlot (int next);
	
	public ByteBuffer getResultSlots ();
	public int numberOfSlots ();
	
	public void flush (); /* Flush last measurement to queue (only if measurements are accumulated) */
	
	public MeasurementQueue getMeasurementQueue ();

	public void setTargetLoss (float target, AtomicBoolean finish);
}
