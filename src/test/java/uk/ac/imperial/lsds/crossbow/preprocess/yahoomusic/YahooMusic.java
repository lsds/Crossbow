package uk.ac.imperial.lsds.crossbow.preprocess.yahoomusic;

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
import uk.ac.imperial.lsds.crossbow.preprocess.EncoderConf;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class YahooMusic {
	
	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (YahooMusic.class);
	
	public static void main (String [] args) throws IOException {
		
		Options options = new Options (YahooMusic.class.getName());
		
		options.addOption ("-i", "Input data directory",  File.class,    "/data/datasets/webscope/r2/ydata-ymusic-user-song-ratings-meta-v1_0");
		options.addOption ("-o", "Output data directory", File.class,    "/data/datasets/webscope/r2/ydata-ymusic-user-song-ratings-meta-v1_0/crossbow");
		options.addOption ("-b", "Micro-batch size",      Integer.class, "2048");
		
		CommandLine commandLine = new CommandLine (options);
		commandLine.parse (args);
		
		/* Encode training data */
		
		DatasetInfo dataset1 = new DatasetInfo ()
		.setExamplesDescriptor (
				
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"train_0.txt",
						"train_1.txt",
						"train_2.txt",
						"train_3.txt",
						"train_4.txt",
						"train_5.txt",
						"train_6.txt",
						"train_7.txt",
						"train_8.txt",
						"train_9.txt"
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "yahoomusic-train.examples")
		)
		.setLabelsDescriptor (
			
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"train_0.txt",
						"train_1.txt",
						"train_2.txt",
						"train_3.txt",
						"train_4.txt",
						"train_5.txt",
						"train_6.txt",
						"train_7.txt",
						"train_8.txt",
						"train_9.txt"
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "yahoomusic-train.labels")
		);
		
		EncoderConf conf1 = new EncoderConf ()
		.setDataset         (dataset1)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape ( new int [] { 2 } ), DataType.INT),
							new DataTuple (new Shape ( new int [] { 1 } ), DataType.FLOAT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "yahoomusic-train.metadata");
		
		YahooMusicEncoder encoder1 = new YahooMusicEncoder (conf1);
		
		encoder1.encode();
		encoder1.getMetadata().store();
		
		/* Encode test data */
		
		DatasetInfo dataset2 = new DatasetInfo ()
		.setExamplesDescriptor (
				
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"test_0.txt",
						"test_1.txt",
						"test_2.txt",
						"test_3.txt",
						"test_4.txt",
						"test_5.txt",
						"test_6.txt",
						"test_7.txt",
						"test_8.txt",
						"test_9.txt",
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "yahoomusic-test.examples")
		)
		.setLabelsDescriptor (
			
		new DatasetDescriptor ()
		.setSource      (options.getOption("-i").getStringValue(), new String [] 
						{	
						"test_0.txt",
						"test_1.txt",
						"test_2.txt",
						"test_3.txt",
						"test_4.txt",
						"test_5.txt",
						"test_6.txt",
						"test_7.txt",
						"test_8.txt",
						"test_9.txt",
						}
						)
		.setDestination (options.getOption("-o").getStringValue(), "yahoomusic-test.labels")
		);
		
		EncoderConf conf2 = new EncoderConf ()
		.setDataset         (dataset2)
		.setBatchInfo       (options.getOption("-b").getIntValue())
		.setDataTuplePair   (
							new DataTuple (new Shape ( new int [] { 2 } ), DataType.INT),
							new DataTuple (new Shape ( new int [] { 1 } ), DataType.FLOAT)
							)
		.setMetadata        (options.getOption("-o").getStringValue(), "yahoomusic-test.metadata");
		
		YahooMusicEncoder encoder2 = new YahooMusicEncoder (conf2);
		
		encoder2.encode();
		encoder2.getMetadata().store();
		
		System.out.println("Bye.");
	}
}
