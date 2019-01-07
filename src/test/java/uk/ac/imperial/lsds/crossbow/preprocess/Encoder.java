package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.DatasetMetadata;
import uk.ac.imperial.lsds.crossbow.SystemConf;

public abstract class Encoder implements IEncoder {
	/*
	 * Knowing the example size and the number of examples per batch
	 * we can compute the batch size in bytes (padded to page size).
	 * 
	 * This is encoded in `conf.getBatchInfo ()`
	 * 
	 * Based on the batch size, we can compute how many batches fit
	 * in 1GB (or less), say M:
	 * 
	 * M = conf.getBatchInfo().numberOfBatchesPerFile();
	 * 
	 * This way, we can also rotate output files:
	 * 
	 * `.data.0`, `.data.1`, and so on, up to `.data.N`.
	 * 
	 * Output files are managed by a DatasetFileWriter.
	 * 
	 * The fill of the last batch can be determined at the end.
	 * 
	 *           1   2   3       M
	 * file 0: |---|---|---|...|---|
	 * file 1: |---|---|---|...|---|
	 *  ...
	 * file N: |---|-xx| [END]
	 * 
	 * The last file may contain less that M batches. If this 
	 * is the case, we can truncate it.
	 * 
	 * Some more comments:
	 * 
	 * - Knowing the output type (e.g. float, int, double) will help
	 *   us differentiate between putFloat, putInt, putDouble, etc.
	 *   
	 *   It should be part of the metadata.
	 *   
	 */
	
	private final static Logger log = LogManager.getLogger (Encoder.class);
	
	protected EncoderConf conf;
	
	protected DatasetMetadata meta = null;
	
	public Encoder (EncoderConf conf) {
		
		this.conf = conf;
	}
	
	public void encode () throws IOException {
		
		/*
		 * Set the limit for output files, determined by the maximum number of examples 
		 * (or labels, depending on which one is bigger) that fit in a file partition.
		 */
		conf.configure (SystemConf.getInstance().getFilePartitionSize());
		
		DatasetInfo dataset = conf.getDataset ();
		
		DatasetDescriptor X = dataset.getExamplesDescriptor ();
		DatasetDescriptor Y = dataset.getLabelsDescriptor ();
		
		/*
		 * We write examples and labels separately, so we will
		 * use two writers.
		 */
		DatasetFileWriter [] writer = new DatasetFileWriter [] {
			
			X.getDatasetFileWriter(),
			Y.getDatasetFileWriter()
		};
		
		DataTuplePair pair = conf.getDataTuplePair ();
		
		DataTupleIterator iterator = iterator ();
		
		Iterator<DatasetFile> it;
		int delta;
		
		if (X.sharesInputWith (Y))
		{
			/* Examples and labels share the same input files */
			
			it = X.getDatasetFileReader ().iterator ();
			
			while (it.hasNext ()) {
				
				DatasetFile file = it.next ();
				ByteBuffer buffer = file.getByteBuffer ();
				
				iterator.parseExamplesFileHeader (buffer);
				
				while (buffer.hasRemaining ()) {
					
					delta = iterator.__nextTuple (buffer, pair);
					if (delta < 1) {
						System.err.println ("error: failed to read next tuple");
						System.exit (1);
					}
					
					/*
					 * Examples and their labels are stored in different
					 * output files
					 */
					writer [0].write (pair.getExample ());
					writer [1].write (pair.getLabel ());
					
					/* log.debug(String.format("%4d examples, %4d labels", writer[0].counter(), writer[1].counter())); */
					
					if (writer [0].counter () == conf.getBatchInfo ().elements ()) {
						
						/* Batch complete: align it to page size */
						
						writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
						writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
					}
				}
				
				file.close ();
			}
		} 
		else
		{
			/* Encode examples and labels separately. First, encode examples */
			
			it = X.getDatasetFileReader ().iterator ();
			
			while (it.hasNext ()) {
				
				DatasetFile file = it.next ();
				ByteBuffer buffer = file.getByteBuffer ();
				
				X.getDatasetFileReader().dump();
				writer [0].dump();
				
				iterator.parseExamplesFileHeader (buffer);
				
				while (buffer.hasRemaining ()) {
					
					delta = iterator.__nextExample (buffer, pair.getExample ());
					if (delta < 1) {
						System.err.println ("error: failed to read next example");
						System.exit (1);
					}
					
					writer [0].write (pair.getExample ());
					
					if (writer [0].counter () == conf.getBatchInfo ().elements ()) {
						
						/* Batch complete: align it to page size */
						writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
					}
				}
				
				file.close ();
			}
			
			/* Second, encode labels */
			
			it = Y.getDatasetFileReader ().iterator ();
			
			while (it.hasNext ()) {
				
				DatasetFile file = it.next ();
				ByteBuffer buffer = file.getByteBuffer ();
				
				Y.getDatasetFileReader().dump();
				writer [1].dump();
				
				iterator.parseLabelsFileHeader (buffer);
				
				while (buffer.hasRemaining ()) {
					
					delta = iterator.__nextLabel (buffer, pair.getLabel ());
					if (delta < 1) {
						System.err.println ("error: failed to read next label");
						System.exit (1);
					}
					
					writer [1].write (pair.getLabel ());
					
					if (writer [1].counter () == conf.getBatchInfo ().elements ()) {
						
						/* Batch complete */
						writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor ().getPad ());
					}
				}
				
				file.close ();
			}
		}
		
		/* 
		 * At this point, all input tuples have been written to one 
		 * or more output files. 
		 * 
		 * If the current batch (i.e. the last one) is not complete 
		 * (say, it's missing N tuples), it should be filled with N 
		 * tuples of the first output file.
		 * 
		 * If the current output file is not full, truncate it.
		 */
		
		if (writer [0].counter() != writer [1].counter()) {
			
			System.err.println (String.format("error: inconsistent number of examples and labels written (%d examples, %d labels)", writer[0].counter(), writer[1].counter()));
			System.exit (1);
		}
		
		int missing = 0;
		
		if (writer [0].counter() > 0) { /* Is the last batch still open? */
			
			missing  = conf.getBatchInfo ().elements () - writer [0].counter ();
			if (missing > 0) {
			
				writer [0].repeat (missing, conf.getDataTuplePair().getExample ());
				writer [1].repeat (missing, conf.getDataTuplePair().getLabel   ());
			}
			
			/* Batch complete, align it to page size */
			writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
			writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
		}
		
		writer [0].close ();
		writer [1].close ();
		
		/* Create metadata */
		
		meta = new DatasetMetadata (conf.getMetadata());
		
		meta.setExamplesFilePrefix (X.getDestination ());
		meta.setLabelsFilePrefix   (Y.getDestination ());
		
		meta.setNumberOfPartitions (writer [0].numberOfFilesWritten ());
		
		meta.setNumberOfExamples (writer [0].numberOfTuplesWritten ());
		
		meta.setBatchSize (conf.getBatchInfo().elements ());
		
		meta.setExampleShape (conf.getDataTuplePair ().getExample ().getShape ());
		meta.setLabelShape   (conf.getDataTuplePair ().getLabel   ().getShape ());
		
		meta.setExampleType (conf.getDataTuplePair ().getExample ().getDataType ());
		meta.setLabelType   (conf.getDataTuplePair ().getLabel   ().getDataType ());
		
		meta.setExamplesFilePad (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
		meta.setLabelsFilePad   (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
		
		meta.setFill (missing);
	}
	
	/*
	 * Reshuffle existing dataset, described in `_meta`
	 */
	public void reshuffle (DatasetMetadata _meta) throws IOException {
		
		if (! _meta.isLoaded())
			_meta.load();
		
		if (! _meta.isFillSet ())
			throw new IllegalStateException ();
		
		/*
		 * Set the limit for output files, determined by the maximum number of examples 
		 * (or labels, depending on which one is bigger) that fit in a file partition.
		 */
		conf.configure (SystemConf.getInstance().getFilePartitionSize());
		
		DatasetInfo dataset = conf.getDataset ();
		
		DatasetDescriptor X = dataset.getExamplesDescriptor ();
		DatasetDescriptor Y = dataset.getLabelsDescriptor ();
		
		/*
		 * We write examples and labels separately, so we will
		 * use two writers.
		 */
		DatasetFileStreamWriter [] writer = new DatasetFileStreamWriter [] {
			
			new DatasetFileStreamWriter (X.getDestination()),
			new DatasetFileStreamWriter (Y.getDestination())
		};
		
		/* Since file stream writers are new, we need to initialise them. */
		writer[0].setLimit (X.getDatasetFileWriter ().getLimit ());
		writer[1].setLimit (Y.getDatasetFileWriter ().getLimit ());
		
		/* To skip `fill`, we need to know the input batch size (in bytes) */
		long [] mark = new long [] {
				
				(long) (_meta.getBatchSize() * _meta.getExampleSize () + _meta.getExamplesFilePad ()),
				(long) (_meta.getBatchSize() * _meta.getLabelSize   () + _meta.getLabelsFilePad   ())
		};
		
		DataTuplePair pair = conf.getDataTuplePair ();
		
		String filename;
		boolean last = false;
		int fill = 0;
		
		/* Reshuffle examples and labels separately. First, reshuffle examples */
		
		for (int idx = 1; idx <= _meta.numberOfPartitions(); ++idx) {
			
			last = (idx == _meta.numberOfPartitions());
			
			filename = String.format("%s.%d", _meta.getExamplesFilePrefix (), idx);
			
			log.debug(String.format("Reshuffle %s", filename));
			
			if (conf.copyBeforeReshuffle ())
				filename = DatasetUtils.copy (filename, conf.getTemporaryFile ());
			
			DatasetFileStream file = new DatasetFileStream (filename);
			
			/* writer [0].dump(); */
			
			while (file.hasRemaining ()) {
				
				/* If we are in the last file, the last batch may contains 0 or more 
				 * additional examples to align it to batch size
				 */
				fill = (last && (file.remaining() == mark[0])) ? _meta.getFill() : 0;
				
				/* Process examples in batch */
				for (int n = 0; n < (_meta.getBatchSize() - fill); ++n) {
					
					/* Populate data tuple buffer with the n-th example */
					byte [] example = pair.getExample ().getBuffer ().array ();
					
					file.read (example);
					
					/* Write image to output file */
					writer [0].write (pair.getExample ());
					
					if (writer [0].counter () == conf.getBatchInfo ().elements ()) {
						/* Batch complete: align it to page size */
						writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
					}
				}
				
				/* Skip padding or filler after processing batch, if any */
				file.skip (_meta.getExamplesFilePad() + (fill * _meta.getExampleSize()));
				
				/* log.debug(file); */
			}
			
			file.close ();
			
			/*
			if (conf.copyBeforeReshuffle ())
				DatasetUtils.delete (filename);
			*/
		}
		
		log.debug(String.format("%d tuples written", writer[0].numberOfTuplesWritten()));
		
		/* Second, reshuffle labels */
		
		for (int idx = 1; idx <= _meta.numberOfPartitions(); ++idx) {
			
			last = (idx == _meta.numberOfPartitions());
			
			filename = String.format("%s.%d", _meta.getLabelsFilePrefix (), idx);
			
			log.debug(String.format("Reshuffle %s", filename));
			
			if (conf.copyBeforeReshuffle ())
				filename = DatasetUtils.copy (filename, conf.getTemporaryFile ());
			
			DatasetFileStream file = new DatasetFileStream (filename);
			
			/* writer [1].dump(); */
			
			while (file.hasRemaining()) {
				
				/* If we are in the last file, the last batch may contains 0 or more
				 * additional labels to align it to batch size.
				 */
				fill = (last && (file.remaining() == mark[1])) ? _meta.getFill() : 0;
				
				/* Process labels in batch */
				for (int n = 0; n < (_meta.getBatchSize() - fill); ++n) {
					
					/* Populate data tuple buffer with the n-th label */
					byte [] label = pair.getLabel ().getBuffer ().array ();
					
					file.read(label);
					
					/* Write label to output file */
					writer [1].write (pair.getLabel ());
					
					if (writer [1].counter () == conf.getBatchInfo ().elements ()) {
						/* Batch complete */
						writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor ().getPad ());
					}
				}
				
				/* Skip padding after processing batch, if any */
				file.skip(_meta.getLabelsFilePad() + (fill * _meta.getLabelSize()));
				
				/* log.debug(file); */
			}
			
			file.close ();
			
			/*
			if (conf.copyBeforeReshuffle ())
				DatasetUtils.delete (filename);
			*/
		}
		
		log.debug(String.format("%d tuples written", writer[0].numberOfTuplesWritten()));
		
		/* 
		 * At this point, all input tuples have been written to one 
		 * or more output files. 
		 * 
		 * If the current batch (i.e. the last one) is not complete 
		 * (say, it's missing N tuples), it should be filled with N 
		 * tuples of the first output file.
		 * 
		 * If the current output file is not full, truncate it.
		 */
		
		if (writer [0].counter() != writer [1].counter()) {
			
			System.err.println (String.format("error: inconsistent number of examples and labels written (%d examples, %d labels)", writer[0].counter(), writer[1].counter()));
			System.exit (1);
		}
		
		int missing = 0;
		
		if (writer [0].counter() > 0) { /* Is the last batch still open? */
			
			missing  = conf.getBatchInfo ().elements () - writer [0].counter ();
			if (missing > 0) {
				
				/*
				 * `missing` examples is less than a batch and an input file, in 
				 * case of reshuffling, contains at least one batch.
				 * 
				 * So it is safe to read `missing` examples the first input file.
				 */
				filename = String.format("%s.%d", _meta.getExamplesFilePrefix (), 1);
				log.debug(String.format("Get %d missing examples from %s", missing, filename));
				
				if (conf.copyBeforeReshuffle ())
					filename = DatasetUtils.copy (filename, conf.getTemporaryFile ());
				
				writer [0].repeat (missing, conf.getDataTuplePair().getExample (), new DatasetFileStream (filename));
				
				/* Repeat for labels */
				filename = String.format("%s.%d", _meta.getLabelsFilePrefix (), 1);
				log.debug(String.format("Get %d missing labels from %s", missing, filename));
				
				if (conf.copyBeforeReshuffle ())
					filename = DatasetUtils.copy (filename, conf.getTemporaryFile ());
				
				writer [1].repeat (missing, conf.getDataTuplePair().getLabel (), new DatasetFileStream (filename));
			}
			
			/* Batch complete, align it to page size */
			writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
			writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
		}
		
		writer [0].close ();
		writer [1].close ();
		
		/* Create metadata */
		
		meta = new DatasetMetadata (conf.getMetadata());
		
		meta.setExamplesFilePrefix (X.getDestination ());
		meta.setLabelsFilePrefix   (Y.getDestination ());
		
		meta.setNumberOfPartitions (writer [0].numberOfFilesWritten ());
		
		meta.setNumberOfExamples (writer [0].numberOfTuplesWritten ());
		
		meta.setBatchSize (conf.getBatchInfo().elements ());
		
		meta.setExampleShape (conf.getDataTuplePair ().getExample ().getShape ());
		meta.setLabelShape   (conf.getDataTuplePair ().getLabel   ().getShape ());
		
		meta.setExampleType (conf.getDataTuplePair ().getExample ().getDataType ());
		meta.setLabelType   (conf.getDataTuplePair ().getLabel   ().getDataType ());
		
		meta.setExamplesFilePad (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
		meta.setLabelsFilePad   (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
		
		meta.setFill (missing);
	}
	
	public void generate (int M) throws IOException {
		
		/*
		 * Set the limit for output files, determined by the maximum number of examples 
		 * (or labels, depending on which one is bigger) that fit in a file partition.
		 */
		conf.configure (SystemConf.getInstance().getFilePartitionSize());
		
		DatasetInfo dataset = conf.getDataset ();
		
		DatasetDescriptor X = dataset.getExamplesDescriptor ();
		DatasetDescriptor Y = dataset.getLabelsDescriptor ();
		
		/*
		 * We write examples and labels separately, so we will
		 * use two writers.
		 */
		DatasetFileWriter [] writer = new DatasetFileWriter [] {
			
			X.getDatasetFileWriter(),
			Y.getDatasetFileWriter()
		};
		
		DataTuplePair pair = conf.getDataTuplePair ();
		
		DataTupleIterator iterator = iterator ();
		
		int count = M * conf.getBatchInfo().elements();
		
		log.info(String.format("Randomly generate %d images", count));
		
		for (int i = 0; i < count; i++) {
			
			iterator.__nextTuple ((ByteBuffer) null, pair);
			
			/*
			 * Examples and their labels are stored in different
			 * output files
			 */
			writer [0].write (pair.getExample ());
			writer [1].write (pair.getLabel ());
			
			log.info(String.format("%4d examples, %4d labels", writer[0].counter(), writer[1].counter()));
			
			if (writer [0].counter () == conf.getBatchInfo ().elements ()) {
				
				/* Batch complete: align it to page size */
				
				writer [0].fill (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
				writer [1].fill (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
			}
		}
		
		writer [0].close ();
		writer [1].close ();
		
		/* Create metadata */
		
		meta = new DatasetMetadata (conf.getMetadata());
		
		meta.setExamplesFilePrefix (X.getDestination ());
		meta.setLabelsFilePrefix   (Y.getDestination ());
		
		meta.setNumberOfPartitions (writer [0].numberOfFilesWritten ());
		
		meta.setNumberOfExamples (writer [0].numberOfTuplesWritten ());
		
		meta.setBatchSize (conf.getBatchInfo().elements ());
		
		meta.setExampleShape (conf.getDataTuplePair ().getExample ().getShape ());
		meta.setLabelShape   (conf.getDataTuplePair ().getLabel   ().getShape ());
		
		meta.setExampleType (conf.getDataTuplePair ().getExample ().getDataType ());
		meta.setLabelType   (conf.getDataTuplePair ().getLabel   ().getDataType ());
		
		meta.setExamplesFilePad (conf.getBatchInfo ().getExampleBatchDescriptor ().getPad ());
		meta.setLabelsFilePad   (conf.getBatchInfo ().getLabelBatchDescriptor   ().getPad ());
		
		meta.setFill (0);
	}
	
	public DatasetMetadata getMetadata () {
		
		return meta;
	}
}
