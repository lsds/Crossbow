package uk.ac.imperial.lsds.crossbow.preprocess.random;

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

public class RandomGenerator {
	
	/*
	 * Datasets required:
	 * 
	 * AlexNet   : 128 x 3 x 224 x 224 (OK)
	 * Overfeat  : 128 x 3 x 231 x 231 (OK)
	 * VGG-A     :  64 x 3 x 224 x 224 (OK)
	 * GoogleNet : 128 x 3 x 224 x 224 (OK)
	 */
	
	public static void main (String [] args) throws IOException {
		
		int M = 100;
		int N =  16;
		int C =   3;
		int H = 231;
		int W = 231;
		
		Options options = new Options (RandomGenerator.class.getName());
		
		options.addOption ("-o", "Output data directory",             File.class,    String.format("/data/crossbow/random/%03d-%03d-%03d-%03d", N, C, H, W));
		options.addOption ("-m", "Number of micro-batches generated", Integer.class, Integer.toString(M));
		options.addOption ("-n", "Micro-batch size",                  Integer.class, Integer.toString(N));
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		/* Encode randomly generated data */
		
		DatasetInfo dataset = new DatasetInfo ()
				.setExamplesDescriptor (new DatasetDescriptor ().setDestination (options.getOption("-o").getStringValue(), "train.examples"))
				.setLabelsDescriptor   (new DatasetDescriptor ().setDestination (options.getOption("-o").getStringValue(), "train.labels"));
		
		EncoderConf conf = new EncoderConf ()
				.setDataset         (dataset)
				.setBatchInfo       (options.getOption("-n").getIntValue())
				.setDataTuplePair   (
									new DataTuple (new Shape (new int [] { C, H, W }), DataType.FLOAT),
									new DataTuple (new Shape (new int [] { 1 }), DataType.INT)
									)
				.setMetadata        (options.getOption("-o").getStringValue(), "train.metadata");
				
		RandomEncoder encoder = new RandomEncoder (conf);
		
		encoder.generate(options.getOption("-m").getIntValue());
		encoder.getMetadata().store();
		
		System.out.println("Bye.");
	}
}
