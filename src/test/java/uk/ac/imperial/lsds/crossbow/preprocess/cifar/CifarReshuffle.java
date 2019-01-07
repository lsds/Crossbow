package uk.ac.imperial.lsds.crossbow.preprocess.cifar;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.DatasetMetadata;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetDescriptor;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetInfo;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class CifarReshuffle {
	
	private final static Logger log = LogManager.getLogger (CifarReshuffle.class);
	
	public static void main (String [] args) throws IOException {
		
		long startTime, dt;
		
		Options options = new Options (CifarReshuffle.class.getName());
		
		options.addOption ("-i", "Input data directory",  File.class,    String.format("/%s/data/cifar-10/b-128", SystemConf.getInstance().getHomeDirectory()));
		options.addOption ("-o", "Output data directory", File.class,    String.format("/%s/data/cifar-10/b-064", SystemConf.getInstance().getHomeDirectory()));
		options.addOption ("-t", "Temporary data file",   File.class,    String.format("/%s/data/cifar-10/b-064/tmp.data", SystemConf.getInstance().getHomeDirectory()));
		options.addOption ("-b", "Micro-batch size",      Integer.class, "64");
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		log.info("Reshuffle training data");
		
		startTime = System.nanoTime ();
		
		/* Load training metadata */
		DatasetMetadata metadata1 = new DatasetMetadata (DatasetUtils.buildPath(options.getOption("-i").getStringValue(), "cifar-train.metadata", true));
		metadata1.load();
		
		/* Compute fill, if not set */
		if (! metadata1.isFillSet()) {
			
			/* The original cifar-10 dataset contains 50,000 images. */
			metadata1.setFill(metadata1.numberOfExamples() - 50000);
		}
		
		/* Reshuffle training data */
		DatasetInfo dataset1 = new DatasetInfo ()
		.setExamplesDescriptor (

			new DatasetDescriptor ()
				.setDestination (options.getOption("-o").getStringValue(), "cifar-train.examples")
		)
		.setLabelsDescriptor (
			
			new DatasetDescriptor ()
				.setDestination (options.getOption("-o").getStringValue(), "cifar-train.labels")
		);
		
		EncoderConf conf1 = new EncoderConf ()
			.setDataset         (dataset1)
			.setBatchInfo       (options.getOption("-b").getIntValue())
			.setDataTuplePair   (
				new DataTuple(new Shape(new int [] { 3, 32, 32 }), DataType.FLOAT),
				new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
			)
			.setMetadata        (options.getOption("-o").getStringValue(), "cifar-train.metadata");
		
		conf1.setCopyBeforeReshuffle(true).setTemporaryFile(options.getOption("-t").getStringValue());
		
		CifarEncoder encoder1 = new CifarEncoder (conf1);
		
		encoder1.reshuffle (metadata1);
		encoder1.getMetadata ().store();
		
		dt = System.nanoTime() - startTime;
		log.info(String.format("Done after %10.2f secs", (double)(dt) / 1000000000));
		
		log.info("Reshuffle test data");
		
		startTime = System.nanoTime ();
		
		/* Load testing metadata */
		DatasetMetadata metadata2 = new DatasetMetadata (DatasetUtils.buildPath(options.getOption("-i").getStringValue(), "cifar-test.metadata", true));
		metadata2.load();
		
		/* Compute fill, if not set */
		if (! metadata2.isFillSet()) {
			
			/* The original cidar-10 validation dataset contains 10,000 images.
			 */
			metadata2.setFill(metadata2.numberOfExamples() - 10000);
		}
		
		/* Reshuffle test data */
		DatasetInfo dataset2 = new DatasetInfo ()
		.setExamplesDescriptor (

			new DatasetDescriptor ()
				.setDestination (options.getOption("-o").getStringValue(), "cifar-test.examples")
			)
		.setLabelsDescriptor (
				
			new DatasetDescriptor ()
				.setDestination (options.getOption("-o").getStringValue(), "cifar-test.labels")
		);
		
		EncoderConf conf2 = new EncoderConf ()
		.setDataset         (dataset2)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
			new DataTuple (new Shape (new int [] { 3, 32, 32 }), DataType.FLOAT),
			new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
		)
		.setMetadata        (options.getOption("-o").getStringValue(), "cifar-test.metadata");
		
		conf2.setCopyBeforeReshuffle(true).setTemporaryFile(options.getOption("-t").getStringValue());
		
		CifarEncoder encoder2 = new CifarEncoder (conf2);
		
		encoder2.reshuffle (metadata2);
		encoder2.getMetadata ().store();
		
		dt = System.nanoTime() - startTime;
		log.info(String.format("Done after %10.2f secs", (double)(dt) / 1000000000));
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
