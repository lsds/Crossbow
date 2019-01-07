package uk.ac.imperial.lsds.crossbow.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.DataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.utils.Linked;

public class Variable implements Linked<Variable>{
	
	private final static Logger log = LogManager.getLogger(Variable.class);
	
	private Shape shape;
	
	/* The term "phantom" is used because when true, the byte
	 * buffer that underlies this variable can disappear or a
	 * new one may appear at any time.
	 * 
	 * Therefore, it is unsafe to make any assumptions about
	 * the byte buffer reference other than its shape. 
	 * 
	 * This is why, for example, a phantom variable is never 
	 * initialised.
	 */
	private boolean phantom;
	
	private IDataBuffer buffer;
	
	private DataType type;
	
	private String name = "Var";
	private int order = -1;
	
	private int capacity;
	
	private Initialiser initialiser;
	
	private float multiplier;
	
	/* We connect variables together to create a model list
	 * per operator. Accessing this list is not thread-safe
	 */ 
	public Variable next = null;
	
	public Variable () {
		
		this (null, null, true, DataType.FLOAT);
	}
	
	public Variable (Shape shape) {
		
		this (null, shape, true, DataType.FLOAT);
	}
	
	public Variable (Shape shape, boolean phantom) {
		
		this (null, shape, phantom, DataType.FLOAT);
	}
	
	public Variable (String name, Shape shape, boolean phantom) {
		
		this (name, shape, phantom, DataType.FLOAT);
	}
	
	public Variable (String name, Shape shape, boolean phantom, DataType type) {
		
		if (name != null)
			this.name = name;
		
		this.shape = shape;
		this.phantom = phantom;
		this.type = type;
		
		capacity = (shape == null) ? 0 : (shape.countAllElements() * type.sizeOf());
		
		multiplier = 1;
		
		if (! isPhantom ()) {
			
			if (capacity <= 0)
				throw new IllegalStateException ("error: capacity of a non-phantom variable must be greater than 0");
			
			/* Allocate buffer */
			buffer = new DataBuffer (capacity, type);
			buffer.finalise(capacity);
		} 
		else {
			buffer = null;
		}
	}
	
	public int capacity () {
		
		if (capacity <= 0)
			throw new IllegalStateException (String.format("error: capacity of variable %s is 0", getName()));
		
		return capacity;
	}
	
	public Shape getShape () {
		return shape;
	}
	
	public boolean isPhantom () {
		return phantom;
	}
	
	public DataType getType () {
		return type;
	}
	
	public IDataBuffer getDataBuffer () {
		return buffer;
	}
	
	public void wrap (IDataBuffer buffer) {
		
		if (! isPhantom())
			throw new IllegalStateException("error: only phantom variables can wrap buffers");
		
		if (capacity <= 0)
			throw new IllegalStateException ("error: cannot limit the capacity of a buffer to 0");
		
		if (capacity > buffer.capacity())
			throw new IllegalStateException ("error: variable buffer overflow");
		
		if (! buffer.isFinalised()) 
			buffer.finalise(capacity);
			
		this.buffer = buffer;
	}
	
	public Variable copy () {
		
		Variable v = new Variable (name, shape.copy(), phantom, type);
		v.setOrder(order);
		
		v.setLearningRateMultiplier (multiplier);
		
		if (! isPhantom()) {
			
			/* Copy data as well */
			log.debug(String.format("Copy %d bytes from %s to clone", buffer.limit(), getName()));
			
			v.getDataBuffer().put(buffer);
		}
		return v;
	}
	
	public String getName () {
		String str = "undefined";
		if (order > 0) {
			str = String.format("%d", order);
			if (order == 1) str += "st";
			else if (order == 2) str += "nd";
			else str += "th";
		}
		String s = String.format("%s (order %s) (shape %s)", name, str, shape);
		return s;
	}
	
	public int getOrder () {
		return order;
	}

	public void setOrder (int order) {
		this.order = order;
	}

	public void initialise (InitialiserConf conf) {
			
		if (isPhantom())
			throw new UnsupportedOperationException("error: cannot initialise a phantom variable");

		if (initialiser != null) {

			log.warn(String.format("Variable %s has already been initialised", getName()));
			initialiser.setConf(conf);
			
		} else {

			initialiser = new Initialiser (conf, this);
		}
		initialiser.initialise();
	}
	
	public float computeChecksum () {
		
		return buffer.computeChecksum ();
	}

	public Variable setLearningRateMultiplier (float multiplier) {
		
		this.multiplier = multiplier;
		return this;
	}
	
	public float getLearningRateMultiplier () {
		
		return multiplier;
	}

	@Override
	public Variable getNext() {
		return next;
	}
}
