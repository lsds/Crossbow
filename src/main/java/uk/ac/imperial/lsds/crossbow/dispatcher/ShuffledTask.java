package uk.ac.imperial.lsds.crossbow.dispatcher;

public class ShuffledTask {
	
	private int id;
	
	private long [] p;
	private long [] q;
	
	private boolean scheduled;
	
	public ShuffledTask (int id) {
		this.id = id;
		
		p = new long [2];
		q = new long [2];
		
		scheduled = false;
	}
	
	public int getId () {
		return id;
	}
	
	public void setP (long ... pointers) {
		p[0] = pointers[0];
		p[1] = pointers[1];
	}
	
	public void setQ (long ... pointers) {
		q[0] = pointers[0];
		q[1] = pointers[1];
	}
	
	public long [] getP () {
		return p;
	}
	
	public long [] getQ () {
		return q;
	}
	
	public void setScheduled (boolean scheduled) {
		this.scheduled = scheduled;
	}
	
	public boolean isScheduled () {
		return scheduled;
	}
	
	public String toString () {
		return String.format ("%03d [%9d, %9d) [%9d, %9d)", id, p[0], q[0], p[1], q[1]);
	}
}
