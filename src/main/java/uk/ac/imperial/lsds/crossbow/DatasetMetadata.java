package uk.ac.imperial.lsds.crossbow;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class DatasetMetadata {
	
	private final static Logger log = LogManager.getLogger (DatasetMetadata.class);
	
	String filename;
	
	/* The prefix for the exaple and label files */
	private String [] prefix;
	
	/* Number of files */
	private int partitions;
	
	/* Total number of elements */
	private int elements;
	
	private int batchSize;
	
	private Shape [] shape;
	
	private DataType [] type;
	
	/* Data files are page-alligned so that, when mapped to memory, 
	 * they can be registered with CUDA memory. 
	 */
	private int [] pad; /* in bytes, for `images` (0) and `labels` (1) */
	
	private int fill;
	
	private boolean loaded;
	
	public DatasetMetadata (String filename) {
		
		this.filename = filename;
		
		prefix = new   String [2];
		shape  = new    Shape [2];
		type   = new DataType [2];
		pad    = new      int [2];
		
		for (int i = 0; i < 2; ++i) {
			
			prefix [i] = null;
			shape  [i] = null;
			type   [i] = null;
			pad    [i] = 0;
		}
		
		fill = -1;
		loaded = false;
	}
	
	public void setNumberOfPartitions (int partitions) {
		
		this.partitions = partitions;
	}
	
	public int numberOfPartitions () {
		
		return partitions;
	}
	
	public void setNumberOfExamples (int elements) {
		
		this.elements = elements;
	}

	public int numberOfExamples () {
		
		return elements;
	}
	
	public void setBatchSize (int batchSize) {
		
		this.batchSize = batchSize;
	}
	
	public int getBatchSize () {
		
		return batchSize;
	}
	
	public int numberOfExamplesPerBatch () {
		
		return batchSize;
	}
	
	public void setPad (int ndx, int pad) {
		
		this.pad [ndx] = pad;
	}
	
	public void setExamplesFilePad (int pad) {
		
		setPad (0, pad);
	}
	
	public void setLabelsFilePad (int pad) {
		
		setPad (1, pad);
	}
	
	public int [] getPad () {
		
		return pad;
	}
	
	public int getPad (int ndx) {
		
		return pad [ndx];
	}
	
	public int getExamplesFilePad () {
		
		return getPad (0);
	}
	
	public int getLabelsFilePad () {
		
		return getPad (1);
	}
	
	public void setShape (int ndx, Shape shape) {
		
		this.shape [ndx] = shape;
	}
	
	public void setExampleShape (Shape shape) {
		
		setShape (0, shape);
	}
	
	public void setLabelShape (Shape shape) {
		
		setShape (1, shape);
	}
	
	public Shape getShape (int ndx) {
		
		return shape [ndx];
	}
	
	public Shape getExampleShape () {
		
		return getShape (0);
	}
	
	public Shape getLabelShape () {
		
		return getShape (1);
	}
	
	public void setType (int ndx, DataType type) {
		
		this.type [ndx] = type;
	}
	
	public void setExampleType (DataType type) {
		
		setType (0, type);
	}
	
	public void setLabelType (DataType type) {
		
		setType (1, type);
	}
	
	public DataType getType (int ndx) {
		
		return type [ndx];
	}
	
	public DataType getExampleType () {
		
		return getType (0);
	}
	
	public DataType getLabelType () {
		
		return getType (1);
	}
	
	public void setPrefix (int ndx, String prefix) {
		
		this.prefix [ndx] = prefix;
	}
	
	public void setExamplesFilePrefix (String prefix) {
		
		setPrefix (0, prefix);
	}
	
	public void setLabelsFilePrefix (String prefix) {
		
		setPrefix (1, prefix);
	}
	
	public String getPrefix (int ndx) {
		
		return prefix [ndx];
	}
	
	public String getExamplesFilePrefix () {
		
		return getPrefix (0);
	}
	
	public String getLabelsFilePrefix () {
		
		return getPrefix (1);
	}
	
	public int getExampleSize () {
		
		return (shape [0].countAllElements() * type [0].sizeOf());
	}
	
	public int getLabelSize () {
		
		return (shape [1].countAllElements() * type [1].sizeOf());
	}
	
	public void setFill (int fill) {
		
		this.fill = fill;
	}
	
	public int getFill () {
		
		return fill;
	}
	
	public boolean isFillSet () {
		
		return (fill >= 0);
	}
	
	public boolean isLoaded () {
		
		return loaded;
	}
	
	public void load () throws IOException {
		
		if (isLoaded ())
			return;
		
		File file = new File(filename);
		
		List<String> lines = Files.readAllLines (file.toPath(), StandardCharsets.UTF_8);
		
		int nr = 0;
		for (String line: lines) {
			
			nr++;
			log.debug(String.format("%2d: %s (%d chars)", nr, line, line.length ()));
			
			/* Clear before and after */
			line = line.trim ();
			
			/* Skip empty lines and comments */
			if ((line.length () == 0) || line.startsWith("#"))
				continue;
			
			String [] s = line.split(":");
			if (s.length != 2)
				throw new IllegalStateException (String.format("error: %s, line %d", filename, nr));
			
			String key = s[0].trim(), value = s[1].trim();
			
			if (key.equalsIgnoreCase("examples")) {
				
				prefix [0] = value;
				
			} else if (key.equalsIgnoreCase("labels")) {
				
				prefix [1] = value;
				
			} else if (key.equalsIgnoreCase("partitions")) {
				
				partitions = Integer.parseInt(value);
			
			} else if (key.equalsIgnoreCase("elements")) {
				
				elements = Integer.parseInt(value);
			
			} else if (key.equalsIgnoreCase("batch size")) {
				
				batchSize = Integer.parseInt(value);
			
			} else if (key.equalsIgnoreCase("example pad")) {
				
				pad [0] = Integer.parseInt(value);
			
			} else if (key.equalsIgnoreCase("label pad")) {
				
				pad [1] = Integer.parseInt(value);
				
			} else if (key.equalsIgnoreCase("example shape")) {
				
				shape [0] = new Shape (value);
			
			} else if (key.equalsIgnoreCase("label shape")) {
					
				shape [1] = new Shape (value);
			
			} else if (key.equalsIgnoreCase("example type")) {
				
				type [0] = DataType.fromString (value);
			
			} else if (key.equalsIgnoreCase("label type")) {
					
				type [1] = DataType.fromString (value);
			
			} else if (key.equalsIgnoreCase("fill")) {
				
				fill = Integer.parseInt(value);
				
			} else {
				
				throw new IllegalStateException 
					(String.format("error: invalid key '%s' in %s, line %d", key, filename, nr));
			}
		}
		
		loaded = true;
		
		return;
	}
	
	public void store () throws IOException {
		
		StringBuilder meta = new StringBuilder ();
		
		meta.append("# Metadata file\n");
		meta.append("#\n");
		
		meta.append(String.format("examples     : %s\n", prefix [0]));
		meta.append(String.format("labels       : %s\n", prefix [1]));
		meta.append(String.format("partitions   : %d\n", partitions));
		meta.append(String.format("elements     : %d\n",   elements));
		meta.append(String.format("batch size   : %d\n",  batchSize));
		meta.append(String.format("example shape: %s\n",  shape [0]));
		meta.append(String.format("label shape  : %s\n",  shape [1]));
		meta.append(String.format("example type : %s\n",   type [0]));
		meta.append(String.format("label type   : %s\n",   type [1]));
		meta.append(String.format("example pad  : %d\n",    pad [0]));
		meta.append(String.format("label pad    : %d\n",    pad [1]));
		
		if (isFillSet ())
			meta.append(String.format("fill         : %d\n", fill));
		
		meta.append("# EOF\n");
		
		File f = new File (filename);
		BufferedWriter writer = new BufferedWriter (new FileWriter(f));
		writer.write (meta.toString());
		writer.close();
	}
	
	public void dump () {
		
		StringBuilder s = new StringBuilder (String.format("=== [Metadata: %s] ===\n", filename));
		
		s.append(String.format("%d elements in %d files", elements, partitions)).append("\n");
		s.append(String.format("Examples at \"%s.*\"", prefix [0])).append("\n");
		s.append(String.format("Labels   at \"%s.*\"", prefix [1])).append("\n");
		s.append(String.format("%d x %s %ss per example batch", batchSize, shape [0], type [0])).append("\n");
		s.append(String.format("%d x %s %ss per label batch", batchSize, shape [1], type [1])).append("\n");
		s.append(String.format("Example batches padded by %d bytes", pad [0])).append("\n");
		s.append(String.format("Label batches padded by %d bytes", pad [1])).append("\n");
		if (isFillSet ())
			s.append(String.format("%d items repeated to fill last batch", fill)).append("\n");
		
		s.append("=== [End of metadata dump] ===");
		
		System.out.println(s.toString());
	}

	public boolean [] realign () {
		
		int bytes, padding;
		
		if (! loaded)
			throw new IllegalStateException();
		
		boolean [] copy = new boolean [2];
		Arrays.fill(copy, false);
		
		/* Check examples */
		
		bytes = getExampleSize() * batchSize + getExamplesFilePad ();
		padding = 0;
		
		while ((bytes + padding) % SystemConf.getInstance().getPageSize() != 0)
			padding++;
		
		/* if (padding != getExamplesFilePad ()) { */
		if (padding > 0) {
			/* Update padding for examples */
			log.info(String.format("Padding examples by %d bytes (original padding was %d)", padding, getExamplesFilePad ()));
			setExamplesFilePad (padding);
			copy [0] = true;
		}
		
		/* Check labels */
		
		bytes = getLabelSize() * batchSize + getLabelsFilePad ();
		padding = 0;
		
		while ((bytes + padding) % SystemConf.getInstance().getPageSize() != 0)
			padding++;
		
		/* if (padding != getLabelsFilePad ()) { */
		if (padding > 0) {
			/* Update padding for labels */
			log.info(String.format("Padding labels by %d bytes (original padding was %d)", padding, getLabelsFilePad ()));
			setLabelsFilePad (padding);
			copy [1] = true;
		}
		
		return copy;
	}

	public void refill () {
		
		/* Is the number of examples a multiple of the batch size? */
		if ((elements % batchSize) != 0) {
			
			if (fill > 0) {
				System.err.println("error: number of elements in dataset is not a multiple of the batch size");
				System.exit(1);
			}
			
			fill = 0;
			while (((elements + fill) % batchSize) != 0)
				fill ++;
			elements += fill;
			
			log.info(String.format("Refill dataset with %d elements", fill));
		}
	}
}
