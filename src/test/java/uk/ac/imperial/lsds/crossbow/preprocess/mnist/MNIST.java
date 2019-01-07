package uk.ac.imperial.lsds.crossbow.preprocess.mnist;

import java.io.File;
import java.io.IOException;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetDescriptor;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetInfo;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class MNIST {
	
	public static void main (String [] args) throws IOException {
		
		int batchSize = 1024;
		
		Options options = new Options (MNIST.class.getName());
		
		options.addOption ("-i", "Input data directory",  File.class,    "/data/crossbow/mnist/original");
		options.addOption ("-o", "Output data directory", File.class,    "/data/crossbow/mnist/b-" + String.format("%03d", batchSize));
		options.addOption ("-b", "Micro-batch size",      Integer.class, Integer.toString(batchSize));
		options.addOption ("-s", "Scale factor",          Float.class,   "0.00390625F");
		
		/* SystemConf.getInstance().setFilePartitionLimit (1048576); */
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		/* Encode training data */
		
		DatasetInfo dataset1 = new DatasetInfo ()
		.setExamplesDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "train-images-idx3-ubyte")
		.setDestination (options.getOption("-o").getStringValue(), "mnist-train.examples")
		)
		.setLabelsDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "train-labels-idx1-ubyte")
		.setDestination (options.getOption("-o").getStringValue(), "mnist-train.labels")
		);
		
		EncoderConf conf1 = new EncoderConf ()
		.setDataset         (dataset1)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape (new int [] { 1, 28, 28 }), DataType.FLOAT),
							new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "mnist-train.metadata")
		.setScaleFactor     (options.getOption("-s").getFloatValue());
		
		MNISTEncoder encoder1 = new MNISTEncoder (conf1);
		
		encoder1.encode();
		encoder1.getMetadata().store();
		
		/* Encode test data */
		
		DatasetInfo dataset2 = new DatasetInfo ()
		.setExamplesDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "t10k-images-idx3-ubyte")
		.setDestination (options.getOption("-o").getStringValue(), "mnist-test.examples")
		)
		.setLabelsDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "t10k-labels-idx1-ubyte")
		.setDestination (options.getOption("-o").getStringValue(), "mnist-test.labels")
		);
		
		EncoderConf conf2 = new EncoderConf ()
		.setDataset         (dataset2)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape (new int [] { 1, 28, 28 }), DataType.FLOAT),
							new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "mnist-test.metadata")
		.setScaleFactor     (options.getOption("-s").getFloatValue());
		
		MNISTEncoder encoder2 = new MNISTEncoder (conf2);
		
		encoder2.encode();
		encoder2.getMetadata().store();
		
		System.out.println("Bye.");
	}
}
