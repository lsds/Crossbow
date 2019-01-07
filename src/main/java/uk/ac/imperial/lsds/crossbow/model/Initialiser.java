package uk.ac.imperial.lsds.crossbow.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.random.RandomGenerator;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;

public class Initialiser {

	private final static Logger log = LogManager.getLogger (Initialiser.class);
	
	InitialiserConf conf;
	
	Variable variable;
	
	public Initialiser (InitialiserConf conf, Variable variable) {
		this.conf = conf;
		this.variable = variable;
	}
	
	public void setConf (InitialiserConf conf) {
		this.conf = conf;
	}
	
	public void initialise () {
		
		InitialiserType type = conf.getType();
		
		switch (type) {
		case CONSTANT: constant (); break;
		case  UNIFORM:  uniform (); break;
		case   XAVIER:   xavier (); break;
		case GAUSSIAN: gaussian (); break;
		case 	 MSRA: 	   msra (); break;
		default:
			throw new IllegalArgumentException ("error: unknown initialiser type");
		}
	}
	
	private void constant () {
		
		log.debug (String.format("Initialising variable %s", variable.getName()));
		
		IDataBuffer buffer = variable.getDataBuffer ();
		
		if (! buffer.isFinalised())
			throw new IllegalStateException("error: cannot initialise a unfinalised variable");
		
		DataType t = variable.getType();
		
		float value = conf.getValue();
		int v = (int) value;
		
		IDataBufferIterator iterator = buffer.getIterator();
		
		while (iterator.hasNext()) {
			
			int offset = iterator.next();
			
			switch (t) {
			case   INT: buffer.putInt   (offset,     v); break;
			case FLOAT: buffer.putFloat (offset, value); break;
			default:
				throw new IllegalStateException("error: invalid variable data type");
			}
		}
	}
	
	private void uniform () {
		throw new UnsupportedOperationException("error: uniform variable initialisation method is not yet implemented");	
	}
	
	private void xavier () {
		
		log.debug (String.format("Initialising variable %s", variable.getName()));
		
		Shape shape = variable.getShape();
		
		int elements = shape.countAllElements();
		int examples = shape.numberOfExamples();
		int channels = shape.numberOfChannels();
		
		float p = (float) elements / (float) examples;
		float q = (float) elements / (float) channels;
		float n;
		switch (conf.getNorm()) {
		case FAN_IN:  n = p; break;
		case FAN_OUT: n = q; break;
		case AVG: 
			n = (p + q) /2F; break;
		default:
			throw new IllegalArgumentException ("error: unknown variable normalisation type");
		}
		
		/* System.out.println("Shape " + shape + " fan_in " + p + " fan_out " + q); */
		
		float scale = (float) Math.sqrt(3. / n);
		
		log.debug(String.format("Xavier filler: n = %5.5f scale %5.5f", n, scale));
		
		float   low = -scale;
		float  high =  scale;
		
		IDataBuffer buffer = variable.getDataBuffer ();
		DataType t = variable.getType();
		if (t.isInt())
			throw new UnsupportedOperationException ("error: xavier initialisation method is not applicable to integer variables"); 
		
		/*
		Random r = new Random ();
		if (SystemConf.getInstance().getRandomSeed() > 0)
			r.setSeed (SystemConf.getInstance().getRandomSeed());
		
		IDataBufferIterator iterator = buffer.getIterator();
		while (iterator.hasNext()) {
			int offset = iterator.next();
			float value = r.nextFloat() * (high - low) + low;
			buffer.putFloat (offset, value);
		}
		*/
		
		RandomGenerator.getInstance().randomUniformFill (buffer, elements, low, high);
	}
	
	private void gaussian () {
		
		log.debug (String.format("Initialising variable %s", variable.getName()));
		
		if (conf.getSparse() >= 0)
			throw new IllegalArgumentException ("error: sparse gaussian variable initialisation method is not yet implemented");
		
		float mean = conf.getMean();
		float std  = conf.getStd ();
		
		if (std <= 0)
			throw new IllegalArgumentException ("error: standard deviation must be greater than 0");
		
		IDataBuffer buffer = variable.getDataBuffer ();
		int elements = variable.getShape().countAllElements();
		
		/*
		Random r = new Random ();
		if (conf.getSeed() > 0)
			r.setSeed(conf.getSeed());
		
		IDataBufferIterator iterator = buffer.getIterator();
		while (iterator.hasNext()) {
			int offset = iterator.next();
			float value = (float) (r.nextGaussian() * std + mean);
			buffer.putFloat (offset, value);
		}
		*/
		
		RandomGenerator.getInstance().randomGaussianFill(buffer, elements, mean, std, (conf.truncate() ? 0 : 1));
	}
	
	private void msra () {
	
		log.debug (String.format("Initialising variable %s", variable.getName()));
		
		if (conf.getSparse() >= 0)
			throw new IllegalArgumentException ("error: sparse gaussian variable initialisation method is not yet implemented");
		
		float mean = conf.getMean();
		int n = variable.getShape().countAllElements() / variable.getShape().numberOfChannels(); // i.e. kW * kH * nOutputPlane (FAN_OUT type)
		float std = (float) Math.sqrt (2.0/n);
		
		IDataBuffer buffer = variable.getDataBuffer ();
		int elements = variable.getShape().countAllElements();
		
		RandomGenerator.getInstance().randomGaussianFill(buffer, elements, mean, std, (conf.truncate() ? 0 : 1));
	}
}
