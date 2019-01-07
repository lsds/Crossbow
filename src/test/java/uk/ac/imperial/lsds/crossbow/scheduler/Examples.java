package uk.ac.imperial.lsds.crossbow.scheduler;

public class Examples {
	
	static DataflowNode [] getNodes (int N) {
		
		DataflowNode [] nodes = new DataflowNode [N];
		
		for (int i = 0; i < N; ++i)
			nodes[i] = new DataflowNode (i);
		
		return nodes;
	}
	
	static DataflowNode _1 () {
		
		DataflowNode [] n = getNodes (7);
		
		n[0].connectTo(n[1]);
		n[1].connectTo(n[2]);
		n[1].connectTo(n[3]);
		n[2].connectTo(n[4]);
		n[3].connectTo(n[5]);
		n[4].connectTo(n[5]);
		n[5].connectTo(n[6]);
		
		return n[0];
	}
	
	static DataflowNode _2 () {
		
		DataflowNode [] n = getNodes (10);
		
		n[0].connectTo(n[1]);
		n[1].connectTo(n[2]);
		n[1].connectTo(n[3]);
		n[2].connectTo(n[4]);
		n[2].connectTo(n[5]);
		n[3].connectTo(n[6]);
		n[4].connectTo(n[7]);
		n[5].connectTo(n[7]);
		n[6].connectTo(n[8]);
		n[7].connectTo(n[8]);
		n[8].connectTo(n[9]);
		
		return n[0];
	}
	
	static DataflowNode _3 () {
		
		DataflowNode [] n = getNodes (12);
		
		n[ 0].connectTo(n[ 1]);
		n[ 1].connectTo(n[ 2]);
		n[ 1].connectTo(n[ 7]);
		n[ 2].connectTo(n[ 3]);
		n[ 2].connectTo(n[ 6]);
		n[ 3].connectTo(n[ 4]);
		n[ 3].connectTo(n[ 5]);
		n[ 4].connectTo(n[ 8]);
		n[ 5].connectTo(n[ 8]);
		n[ 6].connectTo(n[ 9]);
		n[ 7].connectTo(n[10]);
		n[ 8].connectTo(n[ 9]);
		n[ 9].connectTo(n[10]);
		n[10].connectTo(n[11]);
		
		return n[0];
	}
	
	static DataflowNode _4 () {
		
		DataflowNode [] n = getNodes (14);
		
		n[ 0].connectTo(n[ 1]);
		n[ 1].connectTo(n[ 2]);
		n[ 1].connectTo(n[ 3]);
		n[ 2].connectTo(n[ 4]);
		n[ 2].connectTo(n[ 5]);
		n[ 3].connectTo(n[ 6]);
		n[ 3].connectTo(n[ 7]);
		n[ 4].connectTo(n[ 8]);
		n[ 5].connectTo(n[ 8]);
		n[ 6].connectTo(n[ 9]);
		n[ 7].connectTo(n[ 9]);
		n[ 8].connectTo(n[10]);
		n[ 9].connectTo(n[10]);
		n[10].connectTo(n[11]);
		n[10].connectTo(n[12]);
		n[11].connectTo(n[13]);
		
		return n[0];
	}
}
