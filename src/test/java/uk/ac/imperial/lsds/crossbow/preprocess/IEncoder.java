package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;

import uk.ac.imperial.lsds.crossbow.DatasetMetadata;

public interface IEncoder {
	
	public DatasetMetadata getMetadata ();
	
	public void encode () throws IOException;
	
	public abstract DataTupleIterator iterator ();
}
