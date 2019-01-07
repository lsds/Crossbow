package uk.ac.imperial.lsds.crossbow.preprocess.ratings;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.preprocess.DataTuple;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetDescriptor;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetInfo;
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class Ratings {

	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (Ratings.class);
	
	public static void main (String [] args) throws IOException {
		
		Options options = new Options (Ratings.class.getName());
		
		options.addOption ("-i", "Input data directory",  File.class,    String.format("/%s/data/ratings/", SystemConf.getInstance().getHomeDirectory()));
		options.addOption ("-o", "Output data directory", File.class,    String.format("/%s/data/ratings/", SystemConf.getInstance().getHomeDirectory()));
		options.addOption ("-b", "Micro-batch size",      Integer.class, "2048");
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		/* Encode training data */
		
		DatasetInfo dataset = new DatasetInfo ()
		.setExamplesDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "ratings.txt")
		.setDestination (options.getOption("-o").getStringValue(), "ratings.examples")
		)
		.setLabelsDescriptor (
		
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), "ratings.txt")
		.setDestination (options.getOption("-o").getStringValue(), "ratings.labels")
		);
		
		EncoderConf conf = new EncoderConf ()
		.setDataset         (dataset)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape ( new int [] { 2 } ), DataType.INT),
							new DataTuple (new Shape ( new int [] { 1 } ), DataType.FLOAT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "ratings.metadata");
		
		RatingsEncoder encoder = new RatingsEncoder (conf);
		
		encoder.encode();
		encoder.getMetadata().store();
		
		System.out.println("Bye.");
	}
}
