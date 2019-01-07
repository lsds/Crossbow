package uk.ac.imperial.lsds.crossbow;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.dataset.LightWeightDatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.types.DatasetFileType;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.Phase;

/*
 * If the circular data buffer reflects the in-memory data buffer
 * used to copy examples and labels, then how can we track tasks?
 */
public class LightWeightDataset implements IDataset {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (LightWeightDataset.class);
	
	private DatasetMetadata meta;
	
	private Phase phase;
	
	private int parts;
	
	private MappedDataBuffer examples;
	private MappedDataBuffer   labels;
	
	private final static int SLOT_OFFSET = 64;
	
	private ByteBuffer slots;
	
	private boolean initialised;
	
	public LightWeightDataset (String metadatafile) throws IOException {
		
		meta = new DatasetMetadata (metadatafile);
		
		meta.load();
		
		phase = null;
		
		parts = meta.numberOfPartitions();
		
		examples = labels = null;
		
		slots = null;
		
		initialised = false;
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
	
	public long getExamplesCapacity () {
		return examples.capacity();
	}
	
	public long getLabelsCapacity () {
		return labels.capacity();
	}
	
	public MappedDataBuffer getExamples () {
		return examples;
	}
	
	public MappedDataBuffer getLabels () {
		return labels;
	}
	
	public ByteBuffer getDatasetSlots () {
		return slots;
	}
	
	public int numberOfSlots () {
		/* A wrapper over task queue size limit */
		return SystemConf.getInstance().getTaskQueueSizeLimit();
	}
	
	private int getOffset (int slot) {
		return (slot * SLOT_OFFSET);
	}
	
	public void init () {
		
		String filename;
		DatasetFileType type;
		long address;
		int size;
		
		if (meta.getBatchSize () != ModelConf.getInstance ().getBatchSize ())
			meta.setBatchSize (ModelConf.getInstance().getBatchSize());
		
		/* Are examples and/or labels page-aligned? If yes, do nothing, else set padding anew. */
		meta.realign ();
		meta.refill  ();
		
		LightWeightDatasetMemoryManager.getInstance().init (phase.getId(), parts, SystemConf.getInstance().getGPU(), 
			new int [] { 
				meta.getExampleSize () * meta.getBatchSize () + meta.getExamplesFilePad (), 
				meta.getLabelSize ()   * meta.getBatchSize () + meta.getLabelsFilePad ()
			}
		);
		
		LightWeightDatasetMemoryManager.getInstance().setPadding (phase.getId(), meta.getPad());
		
		/* 
		 * Initialise dataset slot buffer 
		 */
		slots = ByteBuffer.allocateDirect (getOffset(numberOfSlots())).order(ByteOrder.LITTLE_ENDIAN);
		
		for (int ndx = 0; ndx < SystemConf.getInstance().getTaskQueueSizeLimit(); ++ndx) {
			
			int pos = getOffset (ndx);
			slots.position(pos);
			
			/* Initialise lock */
			slots.putInt(0);
			
			/* 4 out of 64 bytes written. Fill the rest with 0's. */
			for (int j = 0; j < (SLOT_OFFSET - 4); j++)
				slots.put((byte) 0);
		}
		
		LightWeightDatasetMemoryManager.getInstance().setHandler (phase.getId(), slots, numberOfSlots());
		
		for (int id = 0; id < parts; ++id) {
			
			/* Map examples */
			
			filename = String.format("%s.%d", meta.getExamplesFilePrefix (), (id + 1));
			type = DatasetFileType.EXAMPLES;
			LightWeightDatasetMemoryManager.getInstance().register (phase.getId(), type.getId(), id, filename);
			
			/* Map labels */
			
			filename = String.format("%s.%d", meta.getLabelsFilePrefix (), (id + 1));
			type = DatasetFileType.LABELS;
			LightWeightDatasetMemoryManager.getInstance().register (phase.getId(), type.getId(), id, filename);
		}
		
		/* Configure the dataset memory managers */
		LightWeightDatasetMemoryManager.getInstance().configure (phase.getId(), meta.getBatchSize(), meta.numberOfExamples(), meta.getFill());
		
		/* Finalise the dataset memory managers */
		LightWeightDatasetMemoryManager.getInstance().finalise(phase.getId());
		
		/* Set examples */
		
		address = LightWeightDatasetMemoryManager.getInstance().address  (phase.getId(), DatasetFileType.EXAMPLES.getId());
		size    = LightWeightDatasetMemoryManager.getInstance().capacity (phase.getId(), DatasetFileType.EXAMPLES.getId());
		
		examples = new MappedDataBuffer (phase, DatasetFileType.EXAMPLES, 0, address, size, meta.getExampleType());
		
		examples.order(ByteOrder.LITTLE_ENDIAN);
		
		/* Set labels */
		
		address = LightWeightDatasetMemoryManager.getInstance().address  (phase.getId(), DatasetFileType.LABELS.getId());
		size    = LightWeightDatasetMemoryManager.getInstance().capacity (phase.getId(), DatasetFileType.LABELS.getId());
		
		labels = new MappedDataBuffer (phase, DatasetFileType.LABELS, 0, address, size, meta.getLabelType());
		
		labels.order(ByteOrder.LITTLE_ENDIAN);
		
		initialised = true;
		
		return;
	}
	
	public void translate (long [] p, long [] q, long [] f) {
		/* Nothing to translate. Just prepare task data for execution. */
		LightWeightDatasetMemoryManager.getInstance().reserve (phase.getId(), f);
		return;
	}

	@Override
	public boolean isInitialised () {
		return initialised;
	}

	@Override
	public DatasetType getType() {
	
		return DatasetType.LIGHT;
	}
}
