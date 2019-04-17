package uk.ac.imperial.lsds.crossbow;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.HandlerType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

public class RecordDataset implements IDataset {
	
	private final static Logger log = LogManager.getLogger (RecordDataset.class);
	
	private DatasetMetadata meta;
	
	private Phase phase;
	
	private int parts;
	
	private boolean initialised;
	
	private int [] capacity;
	
	public RecordDataset (String metadatafile) throws IOException {
		
		meta = new DatasetMetadata (metadatafile);
		
		meta.load();
		
		phase = null;
		
		parts = meta.numberOfPartitions();
		
		initialised = false;
		
		capacity = new int [2];
	}
	
	public void setPhase (Phase phase) {
		this.phase = phase;
	}
	
	public Phase getPhase () {
		return phase;
	}
	
	public DatasetMetadata getMetadata () {
		return meta;
	}
	
	public int numberOfPartitions () {
		return parts;
	}
	
	public void init () {
		
		String filename;
		
		if (meta.getBatchSize () != ModelConf.getInstance ().getBatchSize ())
			meta.setBatchSize (ModelConf.getInstance().getBatchSize());
		
		/* Are examples and/or labels page-aligned? If yes, do nothing, else set padding anew. */
		meta.realign ();
		meta.refill  ();
		
		capacity [0] = SystemConf.getInstance().getTaskQueueSizeLimit() * (meta.getExampleSize () * meta.getBatchSize () + meta.getExamplesFilePad ());
		capacity [1] = SystemConf.getInstance().getTaskQueueSizeLimit() * (meta.getLabelSize   () * meta.getBatchSize () + meta.getLabelsFilePad   ());
		
		log.info(">>>>>>>>> Initialise dataset... " + SystemConf.getInstance().getCoreMapper().getOffset(HandlerType.DATASET));
		TheGPU.getInstance().recordDatasetInit (
				phase.getId(), 
				SystemConf.getInstance().numberOfFileHandlers(),
				capacity,
				SystemConf.getInstance().getTaskQueueSizeLimit(),
				ModelConf.getInstance().getBatchSize(),
				meta.getPad()
		);
		
		for (int id = 0; id < parts; ++id) {
			
			/* Map examples */
			
			filename = String.format("%s.%d", meta.getExamplesFilePrefix (), (id + 1));
			
			TheGPU.getInstance().recordDatasetRegister(phase.getId(), id, filename);
		}
		
		/* Finalise the dataset memory managers */
		TheGPU.getInstance().recordDatasetFinalise(phase.getId());
		
		initialised = true;
		
		return;
	}
	
	@Override
	public boolean isInitialised () {
		return initialised;
	}

	@Override
	public ByteBuffer getDatasetSlots () {
		throw new UnsupportedOperationException ("error: unsupported dataset method call");
	}

	@Override
	public int numberOfSlots () {
		throw new UnsupportedOperationException ("error: unsupported dataset method call");
	}

	@Override
	public DatasetType getType() {
		return DatasetType.RECORD;
	}

	public long getExamplesCapacity () {
		return (long) capacity[0];
	}

	public long getLabelsCapacity() {
		return (long) capacity[1];
	}
}
