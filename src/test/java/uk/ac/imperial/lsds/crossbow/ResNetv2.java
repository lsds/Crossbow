//package uk.ac.imperial.lsds.crossbow;
//
//import java.util.ArrayList;
//import java.util.Iterator;
//import java.util.LinkedList;
//
//import org.apache.logging.log4j.LogManager;
//import org.apache.logging.log4j.Logger;
//
//import uk.ac.imperial.lsds.crossbow.kernel.*;
//import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
//import uk.ac.imperial.lsds.crossbow.model.Shape;
//import uk.ac.imperial.lsds.crossbow.utils.CrossbowLinkedList;
//
//public class ResNetv2 {
//
//	public static final String usage = "usage: ResNetv2";
//
//	private final static Logger log = LogManager.getLogger (ResNetv2.class);
//	
//	public static DataflowNode buildResNetUnit (DataflowNode input, int filter, int [] stride, boolean bottleneck, boolean match, String prefix) {
//		
//		DataflowNode node = input;
//		
//		DataflowNode shortcut = null;
//		if (match) {
//			shortcut = input;
//		}
//		else {
//			shortcut = new DataflowNode (new Operator ("Conv", new Conv (new ConvConf())));
//		}
//		
//		DataflowNode sum = null;
//		
//		if (bottleneck) {
//		
//			/*
//			 * [Input]   -> BatchNorm
//			 * BatchNorm -> ReLU
//			 * ReLU      -> Conv
//			 * 
//			 * Conv      -> BatchNorm
//			 * BatchNorm -> ReLU
//			 * ReLU      -> Conv
//			 * 
//			 * Conv      -> BatchNorm
//			 * BatchNorm -> ReLU
//			 * ReLU      -> Conv
//			 */
//			
//			for (int level = 1; level <= 3; ++level) {
//				
//				DataflowNode b = new DataflowNode (new Operator (String.format("%s-BatchNorm-%d", prefix, level), new BatchNorm (new BatchNormConf())));
//				DataflowNode r = new DataflowNode (new Operator (String.format("%s-ReLU-%d"     , prefix, level), new ReLU      (new ReLUConf())));
//				DataflowNode c = new DataflowNode (new Operator (String.format("%s-Conv-%d"     , prefix, level), new Conv      (new ConvConf())));
//				
//				node = node.connectTo(b).connectTo(r).connectTo(c);
//				
//				if ((! match) && (level == 1))
//					r.connectTo(shortcut);
//			}
//			
//		} else {
//		
//			/*
//			 * [Input]   -> BatchNorm
//			 * BatchNorm -> ReLU
//			 * ReLU      -> Conv
//			 * 
//			 * Conv      -> BatchNorm
//			 * BatchNorm -> ReLU
//			 * ReLU      -> Conv
//			 */
//			
//			for (int level = 1; level <= 2; ++level) {
//				
//				DataflowNode b = new DataflowNode (new Operator (String.format("%s-BatchNorm-%d", prefix, level), new BatchNorm (new BatchNormConf())));
//				DataflowNode r = new DataflowNode (new Operator (String.format("%s-ReLU-%d"     , prefix, level), new ReLU      (new ReLUConf())));
//				DataflowNode c = new DataflowNode (new Operator (String.format("%s-Conv-%d"     , prefix, level), new Conv      (new ConvConf())));
//				
//				node = node.connectTo(b).connectTo(r).connectTo(c);
//				
//				if ((! match) && (level == 1))
//					r.connectTo(shortcut);
//			}
//		}
//		
//		sum = new DataflowNode (new Operator (String.format("%s-sum", prefix), new ElementWiseOp (new ElementWiseOpConf())));
//		
//		shortcut.connectTo(sum);
//		node.connectTo(sum);
//		
//		return sum;
//	}
//	
//	public static DataflowNode buildResNet (Shape shape, int stages, int [] units, int [] filters, boolean bottleneck, int classes) {
//		
//		DataflowNode head, node = null;
//		
//		DataflowNode c, b, r, p, f, s, l;
//		
//		/*
//		 * Should we begin with BatchNorm? MXNet does it. TF seems it does not.
//		 */
//		
//		/*
//		 * if CIFAR (i.e., height <= 32)
//		 * 		
//		 * 		Conv
//		 * 
//		 * else
//		 * 		
//		 * 		Conv      -> BatchNorm
//		 * 		BatchNorm -> ReLU
//		 * 		ReLU      -> Pool
//		 */
//		if (shape.height() <= 32) {
//			
//			c = new DataflowNode (new Operator ("Conv",      new Conv      (new ConvConf())));
//			
//			head = c;
//			node = head;
//			
//		} else {
//			
//			c = new DataflowNode (new Operator ("Conv",      new Conv      (new ConvConf())));
//			b = new DataflowNode (new Operator ("BatchNorm", new BatchNorm (new BatchNormConf())));
//			r = new DataflowNode (new Operator ("ReLU",      new ReLU      (new ReLUConf())));
//			
//			head = c;
//			node = head;
//			node = node.connectTo(b).connectTo(r);
//		}
//		
//		String prefix;
//		int [] stride;
//		boolean match;
//		
//		for (int i = 0; i < stages; ++i) {
//			
//			prefix = String.format("stage-%d-unit-%d", (i + 1), 1);
//			stride = (i == 0) ? new int [] { 1, 1 } : new int [] { 2, 2 };
//			match  = false;
//			
//			node = buildResNetUnit (node, filters [i + 1], stride, bottleneck, match, prefix);
//			
//			log.debug(String.format("After stage %s node is %s", prefix, node.getOperator().getName()));
//			
//			for (int j = 0; j < (units.length - 1); ++j) {
//				
//				prefix = String.format("stage-%d-unit-%d", (i + 1), (j + 2));
//				stride = new int [] { 1, 1 };
//				match  = true;
//				
//				node = buildResNetUnit (node, filters [i + 1], stride, bottleneck, match, prefix);
//				
//				log.debug(String.format("After stage %s node is %s", prefix, node.getOperator().getName()));
//			}
//		}
//		
//		log.debug(String.format("After all stages node is %s", node.getOperator().getName()));
//		
//		/* End */
//		
//		/*
//		 * BatchNorm    -> ReLU 
//		 * ReLU         -> Pool 
//		 * Pool         -> InnerProduct 
//		 * InnerProduct -> SoftMax
//		 * SoftMax      -> SoftMaxLoss
//		 */
//		
//		b = new DataflowNode (new Operator ("BatchNorm",    new BatchNorm    (new BatchNormConf())));
//		r = new DataflowNode (new Operator ("ReLU",         new ReLU         (new ReLUConf())));
//		p = new DataflowNode (new Operator ("Pool",         new Pool         (new PoolConf())));
//		f = new DataflowNode (new Operator ("InnerProduct", new InnerProduct (new InnerProductConf())));
//		s = new DataflowNode (new Operator ("SoftMax",      new SoftMax      (new SoftMaxConf())));
//		l = new DataflowNode (new Operator ("SoftMaxLoss",  new SoftMaxLoss  (new LossConf())));
//		
//		node = node
//		.connectTo(b)
//		.connectTo(r)
//		.connectTo(p)
//		.connectTo(f)
//		.connectTo(s)
//		.connectTo(l);
//		
//		return head;
//	}
//	
//	public static void main (String [] args) throws Exception {
//
//		/* What version of ResNet we want to build? E.g. ResNet-50 */
//		int layers = 50;
//
//		/* 
//		 * Seems that number of stages is always 4,
//		 * which is also the size of `units` array. 
//		 */
//		int stages = 4;
//		
//		/* Number of output classes */
//		int classes = 10;
//		
//		boolean bottleneck = false;
//		int [] filters, units;
//		
//		/* Get channels, height, width of an input image. E.g.: */
//		int channels =   3;
//		int height   =  32;
//		int width    =  32;
//		
//		Shape image = new Shape (new int [] { 1, channels, height, width });
//		
//		if (image.height() <= 28) {
//			/*
//			 * Let's not support this case at the moment.
//			 * It seems to refer to MNIST data, right?
//			 */
//			throw new IllegalStateException ();
//		}
//		
//		if (layers >= 50) {
//			filters = new int [] { 64, 256, 512, 1024, 2048 };
//			bottleneck = true;
//		}
//		else {
//			filters = new int [] { 64, 64, 128, 256, 512 };
//			bottleneck = false;
//		}
//		
//		if      (layers ==  18) { units = new int [] {2,  2,  2, 2}; }
//		else if (layers ==  34) { units = new int [] {3,  4,  6, 3}; }
//		else if (layers ==  50) { units = new int [] {3,  4,  6, 3}; }
//		else if (layers == 101) { units = new int [] {3,  4, 23, 3}; }
//		else if (layers == 152) { units = new int [] {3,  8, 36, 3}; }
//		else if (layers == 200) { units = new int [] {3, 24, 36, 3}; }
//		else if (layers == 269) { units = new int [] {3, 30, 48, 8}; }
//		else {
//			throw new IllegalArgumentException ();
//		}
//		
//		DataflowNode head = buildResNet (image, stages, units, filters, bottleneck, classes);
//		
//		
//		/* ====================== [] ============================ */
//		
//		@SuppressWarnings("unchecked")
//		LinkedList<DataflowNode> [] levels = new LinkedList [200]; /* Assume max depth is 40 */
//		for (int i = 0; i < levels.length; ++i)
//			levels[i] = new LinkedList<DataflowNode>();
//		
//		CrossbowLinkedList<DataflowNode> list = new CrossbowLinkedList<DataflowNode> ();
//		head.visit(list);
//		
//		log.info(list.size() + " nodes in list");
//		
//		head.setLevel(0);
//		
//		Iterator<DataflowNode> iterator = list.iterator();
//		while (iterator.hasNext()) {
//			
//			DataflowNode node = iterator.next();
//			
//			StringBuilder   upstream = new StringBuilder ();
//			StringBuilder downstream = new StringBuilder ();
//			
//			ArrayList<DataflowNode> prev = node.getPreviousList();
//			if (prev == null) {
//				upstream.append("null");
//			} else {
//				for (int i = 0; i < prev.size(); ++i) {
//					upstream.append(prev.get(i).getOperator().getName());
//					if (i < (prev.size() - 1))
//						upstream.append(", ");
//				}
//			}
//			
//			ArrayList<DataflowNode> next = node.getNextList();
//			if (next == null) {
//				downstream.append("null");
//			} else {
//				for (int i = 0; i < next.size(); ++i) {
//					downstream.append(next.get(i).getOperator().getName());
//					if (i < (next.size() - 1))
//						downstream.append(", ");
//				}
//			}
//			
//			System.out.println(String.format("%2d: (%s) -> %s -> (%s)", node.getOperator().getId(), upstream.toString(), node.getOperator().getName(), downstream.toString()));
//		}
//		
//		Iterator<DataflowNode> it1 = list.iterator();
//		while (it1.hasNext()) {
//			DataflowNode node = it1.next();
//			log.debug("Set level for operator " + node.getOperator().getName());
//			ArrayList<DataflowNode> prev = node.getPreviousList();
//			int max = -1;
//			if (prev != null) {
//				for (int i = 0; i < prev.size(); ++i) {
//					if (max < prev.get(i).getLevel()) {
//						max = prev.get(i).getLevel();
//					}
//				}
//			}
//			node.setLevel(max + 1);
//			levels[node.getLevel()].add(node);
//		}
//		
//		for (int i = 0; i < levels.length; i++) {
//			if (! levels[i].isEmpty()) {
//				StringBuilder s = new StringBuilder();
//				int size = levels[i].size();
//				int count = 0;
//				for (DataflowNode n: levels[i]) {
//					s.append(n.getOperator().getName());
//					count ++;
//					if (count < size)
//						s.append(" / ");
//				}
//				System.out.println(String.format("%3d: %s", i, s.toString()));
//			}
//		}
//		
//		System.out.println("Bye.");
//		System.exit(0);
//	}
//}
