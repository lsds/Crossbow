package uk.ac.imperial.lsds.crossbow.microbenchmarks.queues;

import uk.ac.imperial.lsds.crossbow.utils.IObjectFactory;

public class ExampleFactory implements IObjectFactory<Example> {

	public Example newInstance() {
		return new Example();
	}

	public Example newInstance (int ndx) {
		return new Example(ndx);
	}
}
