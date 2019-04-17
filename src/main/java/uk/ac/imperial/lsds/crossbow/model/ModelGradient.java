package uk.ac.imperial.lsds.crossbow.model;

import uk.ac.imperial.lsds.crossbow.utils.IObjectPool;
import uk.ac.imperial.lsds.crossbow.utils.Pooled;
//import uk.ac.imperial.lsds.crossbow.utils.TTASLock;

public class ModelGradient implements Pooled<ModelGradient> {
	
	private int id;
	private Model parent;
	
	private VariableGradient [] gradients = null;
	private ModelIterator<VariableGradient> iterator;
	
	private int uBatch;
	
	private volatile boolean accumulated, retained;
	
//	private TTASLock mutex;
	
	private IObjectPool<ModelGradient> pool;
	
	public ModelGradient (Model parent) {	
		this (0, parent);
	}
	
	public ModelGradient (int id, Model parent) {
		this.id = id;
		this.parent = parent;
		Variable [] variables = parent.getVariables ();
		/* Copy structure */
		gradients = new VariableGradient [variables.length];
		for (int ndx = 0; ndx < variables.length; ++ndx) {
			Variable p = variables [ndx];
			if (p != null) {
				gradients [ndx] = new VariableGradient (p);
				VariableGradient g = gradients [ndx];
				while (p.next != null) {
					p = p.next;
					g.next = new VariableGradient (p);
					g = g.next;
				}
			}
		}
		iterator = new ModelIterator<VariableGradient> (gradients);
		uBatch = -1;
		accumulated = false;
		retained = false;
//		mutex = new TTASLock ();
		pool = null;	
	}
	
	public int getId () {
		return id;
	}
	
	public int getMicroBatchId () {
		return uBatch;
	}
	
	public ModelGradient setMicroBatchId (int uBatch) {
		this.uBatch = uBatch;
		return this;
	}
	
	public Model getParent () {
		return parent;
	}
	
	public boolean isAccumulated () {
		return accumulated;
	}
	
	public void setAccumulated (boolean accumulated) {
		this.accumulated = accumulated;
	}
	
	public boolean isRetained () {
		return retained;
	}
	
	public void retain () {
		retained = true;
	}
	
	public void setPool (IObjectPool<ModelGradient> pool) {
		this.pool = pool;
	}
	
	public void clear () {
		uBatch = -1;
		accumulated = false;
		retained = false;
	}
	
	public void free () {
		if (pool == null)
			throw new IllegalStateException ("error: model gradient pool is not set");
		/* Clean up state */
		clear ();
		pool.free (this);
	}
	
	public ModelIterator<VariableGradient> iterator () {
		iterator.reset();
		return iterator;
	}
	
	public VariableGradient getVariableGradient (int ndx, int order) {
		VariableGradient p = gradients [ndx];
		if (p == null)
			throw new NullPointerException(String.format("error: requested model gradient is null (op=%d, order=%d)", ndx, order));
		int n = 1;
		while (n != order && p.next != null) {
			p = p.next;
			n ++;
		}
		if (n != order)
			throw new NullPointerException(String.format("error: invalid gradient request (op=%d, order=%d)", ndx, order));
		return p;
	}
}
