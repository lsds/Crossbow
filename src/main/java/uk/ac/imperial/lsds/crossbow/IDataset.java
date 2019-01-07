package uk.ac.imperial.lsds.crossbow;

import java.nio.ByteBuffer;

import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public interface IDataset {
	
	public void setPhase (Phase phase);

	public DatasetMetadata getMetadata ();

	public boolean isInitialised ();

	public void init ();

	public DatasetType getType ();
	
	public ByteBuffer getDatasetSlots ();

	public int numberOfSlots ();
}
