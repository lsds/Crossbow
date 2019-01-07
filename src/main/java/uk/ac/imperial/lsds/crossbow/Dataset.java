package uk.ac.imperial.lsds.crossbow;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import uk.ac.imperial.lsds.crossbow.data.MappedDataBuffer;
import uk.ac.imperial.lsds.crossbow.device.dataset.DatasetMemoryManager;
import uk.ac.imperial.lsds.crossbow.types.DatasetFileType;
import uk.ac.imperial.lsds.crossbow.types.DatasetType;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.utils.SlottedObjectPool;

public class Dataset implements IDataset {

	private DatasetMetadata meta;
	
	private Phase phase;
	
	private int parts;
	
	private SlottedObjectPool<MappedDataBuffer> examples;
	private SlottedObjectPool<MappedDataBuffer>   labels;
	
	private long [] capacity;
	
	private boolean [] copy;
	
	private boolean initialised;
	
	public Dataset (String metadatafile) throws IOException {
		
		meta = new DatasetMetadata (metadatafile);
		
		meta.load();
		
		phase = null;
		
		parts = meta.numberOfPartitions();
		
		examples = new SlottedObjectPool<MappedDataBuffer> (parts);
		labels   = new SlottedObjectPool<MappedDataBuffer> (parts);
		
		capacity = new long [2];
		Arrays.fill(capacity, 0L);
		
		initialised = false;
	}
	
	public void init () {
		
		String filename;
		DatasetFileType type;
		long address;
		int size;
		MappedDataBuffer buffer;
		
		if (meta.getBatchSize() != ModelConf.getInstance().getBatchSize()) {
			System.err.println(String.format("error: invalid batch size in dataset (found %d, expected %d)", meta.getBatchSize(), ModelConf.getInstance().getBatchSize()));
			System.exit(1);
		}
		/* Are examples and/or labels page-aligned? If yes, do nothing, else set padding anew. */
		copy = meta.realign ();
		
		DatasetMemoryManager.getInstance().init (phase.getId(), parts, SystemConf.getInstance().getGPU(), 
			new int [] { 
				meta.getExampleSize () * meta.getBatchSize () + meta.getExamplesFilePad (), 
				meta.getLabelSize ()   * meta.getBatchSize () + meta.getLabelsFilePad ()
			}
		);
		
		DatasetMemoryManager.getInstance().configure (phase.getId(), meta.getPad());
		DatasetMemoryManager.getInstance().configure (phase.getId(), copy);
		
		for (int id = 0; id < parts; ++id) {
			
			/* Map examples */
			
			filename = String.format("%s.%d", meta.getExamplesFilePrefix (), (id + 1));
			type = DatasetFileType.EXAMPLES;
			
			DatasetMemoryManager.getInstance().register (phase.getId(), type.getId(), id, filename);
			
			address = DatasetMemoryManager.getInstance().address  (phase.getId(), type.getId(), id);
			size    = DatasetMemoryManager.getInstance().capacity (phase.getId(), type.getId(), id);
			
			buffer  = new MappedDataBuffer (phase, type, id, address, size, meta.getExampleType());
			buffer.order(ByteOrder.LITTLE_ENDIAN);
			
			examples.setElementAt(id, buffer);
			
			capacity [0] += size;
			
			/* Map labels */
			
			filename = String.format("%s.%d", meta.getLabelsFilePrefix (), (id + 1));
			type = DatasetFileType.LABELS;
			
			DatasetMemoryManager.getInstance().register (phase.getId(), type.getId(), id, filename);
			
			address = DatasetMemoryManager.getInstance().address  (phase.getId(), type.getId(), id);
			size    = DatasetMemoryManager.getInstance().capacity (phase.getId(), type.getId(), id);
			
			buffer  = new MappedDataBuffer (phase, type, id, address, size, meta.getLabelType());
			buffer.order(ByteOrder.LITTLE_ENDIAN);
			
			labels.setElementAt(id, buffer);
			
			capacity [1] += size;
		}
		
		/* Finalise the dataset memory managers */
		DatasetMemoryManager.getInstance().finalise(phase.getId());
		
		/* Register the first file */
		DatasetMemoryManager.getInstance().slideIn(phase.getId(), 0);
		
		/* (Re-) Assign address for the first file containing examples and labels, respectively */
		examples.elementAt(0).setAddress (DatasetMemoryManager.getInstance().address (phase.getId(), DatasetFileType.EXAMPLES.getId(), 0));
		labels.  elementAt(0).setAddress (DatasetMemoryManager.getInstance().address (phase.getId(),   DatasetFileType.LABELS.getId(), 0));
		
		initialised = true;
		
		return;
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
	
	public long [] capacity () {
		return capacity;
	}
	
	public long capacity (int ndx) {
		return capacity [ndx];
	}
	
	public long getExamplesCapacity () {
		return capacity (0);
	}
	
	public long getLabelsCapacity () {
		return capacity (1);
	}
	
	public SlottedObjectPool<MappedDataBuffer> getExamples () {
		return examples;
	}
	
	public MappedDataBuffer getExamples (int idx) {
		return examples.elementAt (idx);
	}
	
	public SlottedObjectPool<MappedDataBuffer> getLabels () {
		return labels;
	}
	
	public MappedDataBuffer getLabels (int idx) {
		return labels.elementAt (idx);
	}

	@Override
	public boolean isInitialised () {
		return initialised;
	}

	@Override
	public ByteBuffer getDatasetSlots() {
		throw new UnsupportedOperationException ("error: unsupported dataset method call");
	}

	@Override
	public int numberOfSlots() {
		throw new UnsupportedOperationException ("error: unsupported dataset method call");
	}

	@Override
	public DatasetType getType() {
		
		return DatasetType.BASIC;
	}
}
