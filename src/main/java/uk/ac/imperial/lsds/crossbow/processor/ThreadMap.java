package uk.ac.imperial.lsds.crossbow.processor;

public class ThreadMap {
	
	private static final ThreadMap instance = new ThreadMap ();
	
	/* Note that the following value must be a power of two (see `hash`). */
	public static final int _SIZE = 64;
	
	private static class ThreadMapNode {
		
		public long key;
		public int value;
		
		public ThreadMapNode next;
		
		public ThreadMapNode (long key, int value, ThreadMapNode next) {
			
			this.key = key;
			this.value = value;
			this.next = next;
		}
	}
	
	ThreadMapNode [] content;
	int size = 0;
	int nextValue = 0;
	
	public static ThreadMap getInstance () { return instance; }
	
	public synchronized int register (long tid) {
		int idx;
		idx = get (tid);
		if (idx < 0)
			idx = put (tid);
		return idx;
	}
	
	public int size () {
		return this.size;
	}
	
	public ThreadMap () {
		content = new ThreadMapNode [_SIZE];
		for (int i = 0; i < content.length; i++)
			content[i] = null;
	}
	
	private int hash (long key) {
		return (int) (key ^ (key >>> 32));
	}
	
	private int put (long key) {
		/* Lookup element hash code in the table.
		 *  Ideally, there is no chaining. 
		 */
		int h = hash(key) & (_SIZE - 1);
		ThreadMapNode q = content[h];
		ThreadMapNode p = new ThreadMapNode(key, nextValue++, null);
		if (q == null) {
			content[h] = p;
			size++;
		} else {
			System.err.println(
				String.format("warning: chaining entry for thread %d in ThreadMap", 
					key));
			while (q.next != null)
				q = q.next;
			q.next = p;
			size++;
		}
		return p.value;
	}
	
	public int get (long key) {
		int h = hash(key) & (_SIZE - 1);
		ThreadMapNode q = content[h];
		if (q == null)
			return -1;
		while (q.key != key && q.next != null)
			q = q.next;
		if (q.key == key)
			return q.value;
		return -1;
	}
}
