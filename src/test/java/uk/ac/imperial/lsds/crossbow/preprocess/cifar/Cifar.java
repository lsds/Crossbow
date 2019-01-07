package uk.ac.imperial.lsds.crossbow.preprocess.cifar;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetDescriptor;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetInfo;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class Cifar {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (Cifar.class);
	
	public static void main (String [] args) throws IOException {
		
		long startTime, dt;
		
		int batchSize = 256;
		int padding = 4;
		
		Options options = new Options (Cifar.class.getName());
		
		options.addOption ("-i", "Input data directory",  File.class,    "/mnt/nfs/users/piwatcha/16-crossbow/data/cifar-10/original");
		options.addOption ("-o", "Output data directory", File.class,    String.format("/mnt/nfs/users/piwatcha/16-crossbow/data/cifar-10/pre-processed-and-normalised/b-%03d", batchSize));
		options.addOption ("-b", "Micro-batch size",      Integer.class, Integer.toString(batchSize));
		options.addOption ("-p", "Padding",               Integer.class, Integer.toString(padding));
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		/* Encode training data */
		
		System.out.println("Encode training data");
		
		startTime = System.nanoTime ();
		
		DatasetInfo dataset1 = new DatasetInfo ()
		.setExamplesDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"data_batch_1.bin",
						"data_batch_2.bin",
						"data_batch_3.bin",
						"data_batch_4.bin",
						"data_batch_5.bin"
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "cifar-train.examples")
		)
		.setLabelsDescriptor (
			
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"data_batch_1.bin",
						"data_batch_2.bin",
						"data_batch_3.bin",
						"data_batch_4.bin",
						"data_batch_5.bin"
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "cifar-train.labels")
		);
		
		EncoderConf conf1 = new EncoderConf ()
		.setDataset         (dataset1)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape (new int [] { 3, 32 + padding + padding, 32 + padding + padding }), DataType.FLOAT),
							new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "cifar-train.metadata");
		
		CifarEncoder encoder1 = new CifarEncoder (conf1);
		
		encoder1.setComputeMeanImage (true);
		encoder1.setMeanImageFilename (DatasetUtils.buildPath(options.getOption("-o").getStringValue(), "cifar-train.mean", false));
		
		encoder1.setExpectedCount(50000);
		encoder1.setPadding (padding);
			
		encoder1.encode ();
		encoder1.computeAndStoreMeanImage ();
		encoder1.getMetadata ().store ();
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("Done after %10.2f secs", (double)(dt) / 1000000000));
		
		/* Encode test data */
		
		System.out.println("Encode test data");
		
		startTime = System.nanoTime ();
		
		DatasetInfo dataset2 = new DatasetInfo ()
		.setExamplesDescriptor (
				
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "test_batch.bin")
		.setDestination (options.getOption("-o").getStringValue(), "cifar-test.examples")
		)
		.setLabelsDescriptor (
			
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "test_batch.bin")
		.setDestination (options.getOption("-o").getStringValue(), "cifar-test.labels")
		);
		
		EncoderConf conf2 = new EncoderConf ()
		.setDataset         (dataset2)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape (new int [] { 3, 32 + padding + padding, 32 + padding + padding }), DataType.FLOAT),
							new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "cifar-test.metadata");
		
		CifarEncoder encoder2 = new CifarEncoder (conf2);
		
		encoder2.setComputeMeanImage (true);
		encoder2.setMeanImageFilename (DatasetUtils.buildPath(options.getOption("-o").getStringValue(), "cifar-test.mean", false));
		
		encoder2.setExpectedCount(10000);
		encoder2.setPadding (padding);
		
		encoder2.encode ();
		encoder2.computeAndStoreMeanImage ();
		encoder2.getMetadata ().store ();
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("Done after %10.2f secs", (double)(dt) / 1000000000));
		
		System.out.println("Bye.");
	}
}
