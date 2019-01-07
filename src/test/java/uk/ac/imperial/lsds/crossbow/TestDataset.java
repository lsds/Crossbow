package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class TestDataset {

	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (TestDataset.class);
	
	public static void main (String [] args) throws Exception {
		
		String dir = String.format("/%s/data/mnist/b-001", SystemConf.getInstance().getHomeDirectory());
		
		String f = dir + "mnist-train.metadata";
		
		Dataset dataset = new Dataset (f);
		
		DatasetAddressTranslator translator = new DatasetAddressTranslator (dataset.getExamples());
		
		translator.dump();
	}
}
