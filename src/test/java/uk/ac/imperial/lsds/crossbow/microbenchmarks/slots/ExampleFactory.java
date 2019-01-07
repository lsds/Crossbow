package uk.ac.imperial.lsds.crossbow.microbenchmarks.slots;

import uk.ac.imperial.lsds.crossbow.utils.ObjectFactory;

public class ExampleFactory implements ObjectFactory<Example> {

	public Example newInstance() {
		return new Example();
	}

	public Example newInstance(int ndx) {
		return new Example(ndx);
	}
}
