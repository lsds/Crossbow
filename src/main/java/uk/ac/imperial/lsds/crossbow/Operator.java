package uk.ac.imperial.lsds.crossbow;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.data.DataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.CustomDataBufferFactory;
import uk.ac.imperial.lsds.crossbow.kernel.IKernel;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.utils.BaseObjectPoolImpl;
import uk.ac.imperial.lsds.crossbow.utils.CrossbowLinkedList;

public class Operator {
	
	private final static Logger log = LogManager.getLogger (Operator.class);
	
	public static int autoincrement = 0;
	
	private int id = -1;
	
	private String name = "Op";
	
	private IKernel kernel = null;
	
	private Operator peer; /* For gradient operators */
	private int peerRefs;
	
	private Shape [] inputShape;
	private Shape outputShape;
	
	private boolean initialised, registered;
	
	/* Output buffer pool */
	private BaseObjectPoolImpl<DataBuffer> pool;
	
	private DataflowNode [] nodes;
	
	public Operator (IKernel kernel) {
		
		this(null, kernel);
	}
	
	public Operator (String name, IKernel kernel) {
		
		id = autoincrement++;
		
		if (name != null)
			this.name = name;
		this.kernel = kernel;
		
		if (this.kernel != null)
			this.kernel.setOperator(this);
		
		peer = null;
		peerRefs = 0;
		
		inputShape  = null;
		outputShape = null;
		
		initialised = false;
		registered = false;
		
		pool = null;
		
		nodes = new DataflowNode [2];
		Arrays.fill(nodes, null);
	}
	
	public IKernel getKernel () {
		return kernel;
	}
	
	public Shape [] getInputShape () {
		
		if (! isInitialised())
			throw new IllegalStateException (String.format("error: operator %s is not initialised", getName()));
		
		return inputShape;
	}
	
	public int numberOfInputs () {
		
		if (! isInitialised())
			throw new IllegalStateException (String.format("error: operator %s is not initialised", getName()));
		
		return inputShape.length;
	}
	
	public Shape getOutputShape () {
		
		if (! isInitialised())
			throw new IllegalStateException (String.format("error: operator %s is not initialised", getName()));
		
		return outputShape;
	}
	
	public int getId () {
		return id;
	}
	
	public boolean isInitialised () {
		return initialised;
	}
	
	public boolean isRegistered () {
		return registered;
	}
	
	public Operator setPeer (Operator op) {
		peer = op;
		op.peerRefs ++;
		return this;
	}
	
	public int getPeerReferences() {
		return peerRefs;
	}
	
	public boolean isGradient () {
		return (peer != null);
	}
	
	public Operator getPeer () {
		
		if (! isGradient())
			throw new IllegalStateException("error: operator in not a gradient");
		
		return peer;
	}
	
	public ModelAccess getModelAccessType () {
		
		if (! isInitialised())
			throw new IllegalStateException ("error: operator is not initialised");
		
		return kernel.getModelAccessType();
	}
	
	public String getName () {
		if (name.equals("Op"))
			return String.format("%s-%d", name, id);
		return name;
	}
	
	public void setDataflowNode (Phase phase, DataflowNode node) {
		nodes [phase.getId()] = node;
	}
	
	public DataflowNode getDataflowNode (Phase phase) {
		return nodes [phase.getId()];
	}
	
	public void init (Shape [] shape, Model model) {
		
		if (isInitialised()) {
			log.debug(String.format("Operator %s is already initialised", getName()));
			
			/*
			 * TODO
			 * 
			 * An operator can be part of two or more dataflows.
			 * Make sure that the current input shape(s) are the
			 * same as new ones. 
			 */
			return;
		}
		
		log.debug(String.format("Init %s", getName()));
		
		/* `shape` has been allocated for this operator */
		inputShape = shape;
		
		kernel.setup (inputShape, model);
		
		/* The output shape points to the kernel's data structure */
		outputShape = kernel.getOutputShape();
		
		if (outputShape != null) {
			
			int      size = kernel.getOutputSize();
			DataType type = kernel.getOutputType();
			
			pool = new BaseObjectPoolImpl<DataBuffer> (new CustomDataBufferFactory (size, type));
		}
		
		initialised = true;
	}
	
	public DataBuffer getOutputDataBufferInstance () {
		DataBuffer buffer = pool.getInstance();
		if (buffer.referenceCountGetAndIncrement () != 0)
			throw new IllegalStateException ();
		buffer.setPool(pool);
		return buffer;
	}
	
	public boolean GPURegister () {
		
		if (! isInitialised())
			throw new IllegalStateException (String.format("error: operator %s is not initialised", getName()));
		
		if (isRegistered())
			return false;
		
		kernel.GPURegister();
		
		return (registered = true);
	}
	
	@Override
	public boolean equals (Object object) {
		
		if (object == null)
			return false;
		
		if (! Operator.class.isAssignableFrom (object.getClass()))
			return false;
		
		final Operator other = (Operator) object;
		
		return (getName().equals(other.getName()) && id == other.id);
	}
	
	public void computeChecksum (CrossbowLinkedList<IDataBuffer> output) {
		
		if (output.isEmpty()) {
			log.debug(String.format("Kernel's %s output list is empty", getName()));
		}
		else {
			
			int size = output.size();
			
			Iterator<IDataBuffer> iterator = output.iterator ();
			StringBuilder s = new StringBuilder ();
			
			int id = 0;
			
			while (iterator.hasNext()) {
				IDataBuffer buffer = iterator.next();
				s.append(String.format("%.5f", buffer.computeChecksum()));
				if (++id < size)
					s.append(", ");
			}
			
			log.debug(String.format("Kernel's %s output checksum%s %s %s", 
					getName(),
					(size > 1) ?   "s" :   "",
					(size > 1) ? "are" : "is",
					s.toString()));
		}
	}
	
	public static int cardinality () {
		return autoincrement;
	}
}
