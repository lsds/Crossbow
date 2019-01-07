package uk.ac.imperial.lsds.crossbow.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.data.IDataBufferIterator;
import uk.ac.imperial.lsds.crossbow.device.TheGPU;
import uk.ac.imperial.lsds.crossbow.device.blas.BLAS;
import uk.ac.imperial.lsds.crossbow.utils.BaseObjectPoolImpl;

public class Model implements Iterable <Variable> {
	
	private final static Logger log = LogManager.getLogger (Model.class);
	
	private Variable [] variables;
	private int size;
	
	private final static int N = 1024000;
	
	private volatile int clock;
	 
	private ModelIterator<Variable> iterator;
	
	private BaseObjectPoolImpl<ModelGradient> pool;
	
	private ModelGradient last;
	
	private int updates;
	
	private ModelLock mutex;
	
	private boolean finalised;
	
	private Model baseModel;
	
	public Model () {
		this(N);
	}
	
	public Model (int nops) {
		
		variables = new Variable [nops];
		for (int i = 0; i < nops; ++i)
			variables[i] = null;
		
		size = 0;
		
		clock = 0;
		
		// f = null;
		iterator = null;
		
		pool = null;
		last = null;
		
		updates = 0;
		
		mutex = new ModelLock();
		
		finalised = false;
		
		baseModel = null;
	}
	
	public void    readLock      () {        mutex.incReaders();    }
	public void    readUnlock    () {        mutex.decReaders();    }
	public void    writeLock     () {        mutex.lock();          }	
	public void    writeUnlock   () {        mutex.unlock();        }
	public boolean tryWriteLock  () { return mutex.tryLock();       }
	public boolean isWriteLocked () { return mutex.isWriteLocked(); }
	
	public void setBaseModel(Model theModel) {
		baseModel = theModel;
	}
	
	public Model getBaseModel () {
		return baseModel;
	}

	/* 
	 * Whenever a kernel requires access to a particular variable
	 * that it has registered, it queries it by index or by name
	 */ 
	public Variable getVariable (int ndx, int order) {
		
		if (ndx < 0 || ndx > variables.length - 1)
			throw new ArrayIndexOutOfBoundsException("error: model variable index out of bounds");
		
		Variable p = variables[ndx];
		if (p == null)
			throw new NullPointerException("error: kernel model variable is null");
		int n = 1;
		while (n != order && p.next != null) {
			p = p.next;
			n ++;
		}
		if (n == order) 
			return p;
		else
			throw new NullPointerException("error: invalid kernel model variable request");
	}
	
	public Variable getVariable (int ndx, String name) {
		
		if (ndx < 0 || ndx > variables.length - 1)
			throw new ArrayIndexOutOfBoundsException("error: model variable index out of bounds");
		
		Variable p = variables[ndx];
		if (p == null)
			throw new NullPointerException("error: kernel model variable is null");
		while (p.getName().equals(name) && p.next != null)
			p = p.next;
		if (p.getName().equals(name))
			return p;
		else
			throw new NullPointerException("error: invalid kernel model variable request");
	}
	
	public int register (int ndx, Variable variable) {
			
		if (ndx < 0 || ndx > variables.length - 1)
			throw new ArrayIndexOutOfBoundsException("error: model variable index out of bounds");
		
		int order = 1;
		Variable p = variables[ndx];
		if (p == null) {
			variables[ndx] = variable;
		} else {
			order = 2;
			while (p.next != null) {
				p = p.next;
				order ++;
			}
			p.next = variable;
			variable.next = null;
		}
		variable.setOrder(order);
		log.debug(String.format("Registered variable %s for operator %d", variable.getName(), ndx));
		size ++;
		return order;
	}
	
	public int getSize () {
		return size;
	}
	
	public Model finalise (int nops) {
		
		if (finalised)
			throw new IllegalStateException ("error: model is already finalised");
		
		/* Trim model to dataflow size */
		if (nops < N) {
			Variable [] v = new Variable [nops];
			for (int i = 0; i < nops; ++i)
				v[i] = variables[i];
			variables = v;
		}
		
		pool = new BaseObjectPoolImpl<ModelGradient> (new CustomModelGradientFactory (this));
		
		/* Initialise iterator */
		iterator = new ModelIterator<Variable> (variables);
		
		finalised = true;
		return this;
	}
	
	public ModelIterator<Variable> iterator () {
		iterator.reset();
		return iterator;
	}
	
	public ModelGradient getGradientInstance (int uBatch) {
		ModelGradient gradient = pool.getInstance();
		/* Initialise */
		gradient.setMicroBatchId(uBatch);
		/* Return object to this pool */
		gradient.setPool(pool);
		return gradient;
	}
	
	/* Create a model replica */
	public Model copy () {
		
		if (! finalised)
			throw new IllegalStateException ("error: cannot copy a model that is not finalised");
		
		log.debug("Creating a copy of the model...");
		
		Model copy = null;
		copy = new Model ();
		for (int ndx = 0; ndx < variables.length; ++ndx) {
			Variable p = variables[ndx];
			if (p != null) {
				log.debug(String.format("Create a copy of model variable %s with checksum %.5f", p.getName(), p.computeChecksum()));
				copy.variables[ndx] = p.copy();
				log.debug(String.format("Copy's checksum is %.5f", copy.variables[ndx].computeChecksum()));
				copy.size ++;
				Variable q = copy.variables[ndx];
				while (p.next != null) {
					p = p.next;
					q.next = p.copy();
					copy.size ++;
					q = q.next;
				}
			}
		}
		copy.finalise (variables.length);
		return copy;
	}
	
	public int getModelClock () {
		return clock;
	}
	
	/* This method should be thread-safe. `clock` is set while the model is write-locked */
	public void setModelClock (int clock) {
		this.clock = clock;
	}
	
	public ModelGradient getLastGradient () {
		return last;
	}
	
	public int numberOfUpdates () {
		return updates;
	}
	
	public int incUpdates () {
		return (++updates);
	}
	
	public void resetUpdates () {
		updates = 0;
	}
	
	public Variable [] getVariables () {
		return variables;
	}
	
	/* Apply gradient `p` to model */
	
	public void apply (ModelGradient gradient) {
		
		ModelIterator<Variable>         m =     this.iterator();
		ModelIterator<VariableGradient> g = gradient.iterator();
		
		IDataBuffer X, Y;
		int count;
		
		while (m.hasNext() && g.hasNext()) {
			
			Y = m.next().getDataBuffer();
			X = g.next().getDataBuffer();
			
			count = Y.limit() / Y.getType().sizeOf();
			
			BLAS.getInstance().saxpby(count, -1, X, 0, X.limit(), /* incX */ 1, 1F, Y, /* incY */ 1);
		}
		
		incUpdates();
		
		/*
		 * Commented out lines below
		 *  
		 * Need to sort out model synchronisation before we deal with momentum 
		 * (which is what we use `last` for)
		 */

//		if (requireLastGradient()) {
//			ModelGradient q = last;
//			last = p;
//
//			if (q != null) {
//				q.setLastGradientStatus(false);
//				tryFreeGradient(q);
//			}
//		}
	}
	
	public void scale (float factor) {
		
		ModelIterator<Variable> m = iterator();
		
		IDataBuffer buffer;
		
		IDataBufferIterator b;
		int offset;
		
		while (m.hasNext()) {
			
			buffer = m.next().getDataBuffer();
			
			b = buffer.getIterator();
			while (b.hasNext()) {
				offset = b.next();
				buffer.putFloat (offset, factor * buffer.getFloat(offset));
			}
		}
	}
	
	public void merge (float factor, Model other) {
		
		ModelIterator<Variable> m1 =  this.iterator();
		ModelIterator<Variable> m2 = other.iterator();
		
		IDataBuffer X, Y;
		int count;
		
		while (m1.hasNext() && m2.hasNext()) {
			
			Y = m1.next().getDataBuffer();
			X = m2.next().getDataBuffer();
			
			count = Y.limit() / Y.getType().sizeOf();
			
			BLAS.getInstance().saxpby(count, factor, X, 0, X.limit(), /* incX */ 1, 1F, Y, /* incY */ 1);
		}
	}
	
	public int capacity () {
		int size = 0;
		if (! finalised)
			return size;
		for (int ndx = 0; ndx < variables.length; ++ndx) {
			Variable p = variables[ndx];
			while (p != null) {
				size += p.capacity();
				p = p.next;
			}
		}
		return size;
	}
	
	public void GPURegister () {
		
		if (! finalised)
			throw new IllegalStateException ("error: model is not finalised");
		
		int bytes = capacity ();
		TheGPU.getInstance().setModel (variables.length, bytes);
		
		for (int ndx = 0; ndx < variables.length; ++ndx) {
			Variable p = variables[ndx];
			while (p != null) {
				TheGPU.getInstance().setModelVariable (ndx, p.getOrder(), p.getShape().array(), p.capacity());
				TheGPU.getInstance().setModelVariableData (ndx, p.getOrder(), p.getDataBuffer());
				TheGPU.getInstance().setModelVariableLearningRateMultiplier(ndx, p.getOrder(), p.getLearningRateMultiplier());
				p = p.next;
			}
		}
		
		// TheGPU.getInstance().overrideModelData (SystemConf.getInstance().getModelDirectory ());
		
		/* Set work per clock */
		TheGPU.getInstance().setModelWorkPerClock (ModelConf.getInstance().getWpc());
		
		/* Set model update type */
		TheGPU.getInstance().setUpdateModelType (ModelConf.getInstance().getUpdateModel().getId());
		
		/* Set solver parameters */
		if (ModelConf.getInstance().getSolverConf() == null)
			throw new IllegalStateException ("error: solver configuration is null");
		
		ModelConf.getInstance().getSolverConf().GPURegister ();
		
		return;
	}
	
	public String toString() {
		return String.format("<clock %3d>", clock);
	}
	
	public void dump () {
		StringBuilder s = new StringBuilder (String.format("=== [Model: %d variables] ===\n", size));
		for (int ndx = 0; ndx < variables.length; ++ndx) {
			s.append(String.format("Op %2d: ", ndx));
			Variable p = variables[ndx];
			while (p != null) {
				s.append(String.format("%s -> ", p.getName()));
				p = p.next;
			}
			s.append("null");
			s.append("\n");
		}
		s.append("=== [End of model dump] ===");
		System.out.println(s.toString());
	}
}
