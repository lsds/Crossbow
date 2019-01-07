package uk.ac.imperial.lsds.crossbow.model;

import uk.ac.imperial.lsds.crossbow.data.DataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.utils.Linked;

public class VariableGradient implements Linked<VariableGradient> {
	
	public IDataBuffer buffer;
	public VariableGradient next = null;
	private float multiplier;
	
	public VariableGradient (Variable v) {
		
		buffer = new DataBuffer (0, v.capacity(), v.getType());
		buffer.finalise(v.capacity());
		multiplier = v.getLearningRateMultiplier();
		next = null;
	}

	@Override
	public VariableGradient getNext() {
		return next;
	}

	public IDataBuffer getDataBuffer() {
		return buffer;
	}

	public float getLearningRateMultiplier() {
		return multiplier;
	}
}