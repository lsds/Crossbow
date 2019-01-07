package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.*;
import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;
import uk.ac.imperial.lsds.crossbow.types.UpdateModel;

/*
 * Based on:
 * 
 * https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/inception-v3.py
 */
public class Inceptionv3 {

	public static final String usage = "usage: Inceptionv3";

	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (Inceptionv3.class);

	private static InitialiserType initialiserType = InitialiserType.GAUSSIAN;

	public static DataflowNode [] InceptionUnit (DataflowNode [] input, DataflowNode [] gradient, String prefix, int numberOfOutputs, int [] kernel, int [] stride, int [] padding) {
		
		ConvConf convConf = new ConvConf ();

		convConf.setNumberOfOutputs (numberOfOutputs);
		
		convConf.setKernelSize  (2).setKernelHeight  (kernel  [1]).setKernelWidth  (kernel  [0]);
		convConf.setStrideSize  (2).setStrideHeight  (stride  [1]).setStrideWidth  (stride  [0]);
		convConf.setPaddingSize (2).setPaddingHeight (padding [1]).setPaddingWidth (padding [0]);
		
		convConf.setWeightInitialiser (new InitialiserConf ().setType (initialiserType).setStd (0.1F));
		convConf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT));
		
		convConf.setBias (false);

		BatchNormConf normConf = new BatchNormConf ();
		
		normConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
		
		ReLUConf reluConf = new ReLUConf ();
		
		DataflowNode conv = new DataflowNode (new Operator (String.format("%s-Conv",      prefix), new Conv      (convConf)));
		DataflowNode norm = new DataflowNode (new Operator (String.format("%s-BatchNorm", prefix), new BatchNorm (normConf)));
		DataflowNode relu = new DataflowNode (new Operator (String.format("%s-ReLU",      prefix), new ReLU      (reluConf)));
		
		DataflowNode convgradient = new DataflowNode (new Operator (String.format("%s-ConvGradient",      prefix), new ConvGradient      (convConf)).setPeer(conv.getOperator()));
		DataflowNode normgradient = new DataflowNode (new Operator (String.format("%s-BatchNormgradient", prefix), new BatchNormGradient (normConf)).setPeer(norm.getOperator()));
		DataflowNode relugradient = new DataflowNode (new Operator (String.format("%s-ReLUGradient",      prefix), new ReLUGradient      (reluConf)).setPeer(relu.getOperator()));
		
		/* Create a shallow copy */
		DataflowNode _relu = relu.shallowCopy();
		
		input [0].connectTo (conv).connectTo (norm).connectTo (relu);
		input [1].connectTo (conv.shallowCopy()).connectTo (norm.shallowCopy()).connectTo (_relu);
		
		relugradient.connectTo (normgradient).connectTo (convgradient).connectTo (gradient[0]);

		gradient[0] = relugradient;
		
		/* Set current value for input */
		input [0] =  relu;
		input [1] = _relu;
		
		return input;
	}
	
	public static DataflowNode [] inceptionPool (DataflowNode [] input, DataflowNode [] gradient, String prefix, PoolMethod method, int kernel, int stride, int padding) {
		
		PoolConf conf = new PoolConf().setMethod(method).setKernelSize(kernel).setStrideSize(stride).setPaddingSize(padding);
		
		DataflowNode pool = new DataflowNode (new Operator (String.format("%s-Pool", prefix), new Pool (conf)));
		
		DataflowNode poolgradient = new DataflowNode (new Operator (String.format("%s-PoolGradient", prefix), new PoolGradient (conf)).setPeer(pool.getOperator()));

		/* Create a shallow copy */
		DataflowNode _pool = pool.shallowCopy();
		
		input [0].connectTo (pool);
		input [1].connectTo (_pool);
		
		for (int i = 0; i < gradient.length; i++) {
			if (gradient[i] != null)
				poolgradient.connectTo (gradient[i]);
		}
		
		gradient[0] = poolgradient;
		if (gradient.length > 1)
			for (int i = 1; i < gradient.length; i++)
				gradient[i] = null;
		
		/* Set current value for input */
		input [0] =  pool;
		input [1] = _pool;

		return input;
	}

	public static DataflowNode [] Inception7A (DataflowNode [] input, DataflowNode [] gradient, String prefix,
			/* Tower configurations */
			int t1_a_1x1,
			int t3_a_1x1, int t3_b_3x3, int t3_c_3x3,
			int t2_a_5x5, int t2_b_5x5,
			PoolMethod t4_a_3x3, int t4_b_1x1) {
		
		int N = 4; /* 4 towers */
		
		DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		DataflowNode [][] tower = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			tower[i] = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [][] towergradient = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			towergradient[i] = new DataflowNode [] { merge };
		
		DataflowNode concat, concat_;
		
		/* Tower 0 */
		tower[0] = InceptionUnit (tower[0], towergradient[0], String.format("%s-tower-1(a)-1x1", prefix), t1_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		/* Tower 1 */
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(a)-1x1", prefix), t2_a_5x5, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(b)-5x5", prefix), t2_b_5x5, new int [] {5, 5}, new int [] {1, 1}, new int [] {2, 2});
		/* Tower 2 */
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(a)-1x1", prefix), t3_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(b)-3x3", prefix), t3_b_3x3, new int [] {3, 3}, new int [] {1, 1}, new int [] {1, 1});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(c)-3x3", prefix), t3_c_3x3, new int [] {3, 3}, new int [] {1, 1}, new int [] {1, 1});
		/* Tower 3 */
		tower[3] = inceptionPool (tower[3], towergradient[3], String.format("%s-tower-4(a)-3x3", prefix), t4_a_3x3, 3, 1, 1);
		tower[3] = InceptionUnit (tower[3], towergradient[3], String.format("%s-tower-4(b)-1x1", prefix), t4_b_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		
		concat  = new DataflowNode  (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		concat_ = concat.shallowCopy ();
		
		/* Wire forward nodes */
		for (int i = 0; i < N; i++) {
			tower[i][0].connectTo(concat );
			tower[i][1].connectTo(concat_);
		}
		
		input [0] = concat;
		input [1] = concat_;
		
		/* Wire backward nodes */
		DataflowNode [] concatgradient = new DataflowNode [N];
		for (int i = 0; i < N; i++) {
			concatgradient[i] = new DataflowNode 
					(new Operator (String.format("%s-tower-%d-ConcatGradient",  prefix, (i + 1)), 
							new ConcatGradient (new ConcatConf ().setOffset(i))).setPeer(concat.getOperator()));
			
			concatgradient[i].connectTo(towergradient [i][0]);
			gradient[i] = concatgradient [i];
		}
		/* Set (any) remaining gradients */
		if (N < gradient.length)
			for (int i = N; i < gradient.length; i++)
				gradient[i] = null;
		
		return input;
	}
	
	public static DataflowNode [] Inception7B (DataflowNode [] input, DataflowNode [] gradient, String prefix,
			int t1_a_3x3,
			int t2_a_1x1, int t2_b_3x3, int t2_c_3x3,
			PoolMethod t3_a_3x3) {

		int N = 3; /* 3 towers */
		
		DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		DataflowNode [][] tower = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			tower[i] = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [][] towergradient = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			towergradient[i] = new DataflowNode [] { merge };
		
		DataflowNode concat, concat_;

		/* Tower 0 */
		tower[0] = InceptionUnit (tower[0], towergradient[0], String.format("%s-tower-1(a)-3x3", prefix), t1_a_3x3, new int [] {3, 3}, new int [] {2, 2}, new int [] {0, 0});
		/* Tower 1 */
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(a)-1x1", prefix), t2_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(b)-3x3", prefix), t2_b_3x3, new int [] {3, 3}, new int [] {1, 1}, new int [] {1, 1});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(c)-3x3", prefix), t2_c_3x3, new int [] {3, 3}, new int [] {2, 2}, new int [] {0, 0});
		/* Tower 2 */
		tower[2] = inceptionPool (tower[2], towergradient[2], String.format("%s-tower-3(a)-3x3", prefix), t3_a_3x3, 3, 2, 0);
		
		concat  = new DataflowNode  (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		concat_ = concat.shallowCopy ();
		
		/* Wire forward nodes */
		for (int i = 0; i < N; i++) {
			tower[i][0].connectTo(concat );
			tower[i][1].connectTo(concat_);
		}
		
		input [0] = concat;
		input [1] = concat_;
		
		/* Wire backward nodes */
		DataflowNode [] concatgradient = new DataflowNode [N];
		for (int i = 0; i < N; i++) {
			concatgradient[i] = new DataflowNode 
					(new Operator (String.format("%s-tower-%d-ConcatGradient",  prefix, (i + 1)), 
							new ConcatGradient (new ConcatConf ().setOffset(i))).setPeer(concat.getOperator()));
			
			concatgradient[i].connectTo(towergradient [i][0]);
			gradient[i] = concatgradient [i];
		}
		/* Set (any) remaining gradients */
		if (N < gradient.length)
			for (int i = N; i < gradient.length; i++)
				gradient[i] = null;
		
		return input;
	}

	public static DataflowNode [] Inception7C (DataflowNode [] input, DataflowNode [] gradient, String prefix,
			int t1_a_1x1,
			int t2_a_1x1, int t2_b_1x7, int t2_c_7x1,
			int t3_a_1x1, int t3_b_7x1, int t3_c_1x7, int t3_d_7x1, int t3_e_1x7,
			PoolMethod t4_a_3x3, int t4_b_1x1) {
		
		int N = 4; /* 4 towers */
		
		DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		DataflowNode [][] tower = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			tower[i] = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [][] towergradient = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			towergradient[i] = new DataflowNode [] { merge };
		
		DataflowNode concat, concat_;
		
		/* Tower 0 */
		tower[0] = InceptionUnit (tower[0], towergradient[0], String.format("%s-tower-1(a)-1x1", prefix), t1_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		/* Tower 1 */
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(a)-1x1", prefix), t2_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(b)-1x7", prefix), t2_b_1x7, new int [] {1, 7}, new int [] {1, 1}, new int [] {0, 3});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(c)-7x1", prefix), t2_c_7x1, new int [] {7, 1}, new int [] {1, 1}, new int [] {3, 0});
		/* Tower 2 */
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(a)-1x1", prefix), t3_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(b)-7x1", prefix), t3_b_7x1, new int [] {7, 1}, new int [] {1, 1}, new int [] {3, 0});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(c)-1x7", prefix), t3_c_1x7, new int [] {1, 7}, new int [] {1, 1}, new int [] {0, 3});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(d)-7x1", prefix), t3_d_7x1, new int [] {7, 1}, new int [] {1, 1}, new int [] {3, 0});
		tower[2] = InceptionUnit (tower[2], towergradient[2], String.format("%s-tower-3(e)-1x7", prefix), t3_e_1x7, new int [] {1, 7}, new int [] {1, 1}, new int [] {0, 3});
		/* Tower 3 */
		tower[3] = inceptionPool (tower[3], towergradient[3], String.format("%s-tower-4(a)-3x3", prefix), t4_a_3x3, 3, 1, 1);
		tower[3] = InceptionUnit (tower[3], towergradient[3], String.format("%s-tower-4(b)-1x1", prefix), t4_b_1x1, new int [] {1, 1}, new int[] {1, 1}, new int[] {0, 0});

		concat  = new DataflowNode  (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		concat_ = concat.shallowCopy ();
		
		/* Wire forward nodes */
		for (int i = 0; i < N; i++) {
			tower[i][0].connectTo(concat );
			tower[i][1].connectTo(concat_);
		}
		
		input [0] = concat;
		input [1] = concat_;
		
		/* Wire backward nodes */
		DataflowNode [] concatgradient = new DataflowNode [N];
		for (int i = 0; i < N; i++) {
			concatgradient[i] = new DataflowNode 
					(new Operator (String.format("%s-tower-%d-ConcatGradient",  prefix, (i + 1)), 
							new ConcatGradient (new ConcatConf ().setOffset(i))).setPeer(concat.getOperator()));
			
			concatgradient[i].connectTo(towergradient [i][0]);
			gradient[i] = concatgradient [i];
		}
		/* Set (any) remaining gradients */
		if (N < gradient.length)
			for (int i = N; i < gradient.length; i++)
				gradient[i] = null;
		
		return input;
	}
	
	public static DataflowNode [] Inception7D (DataflowNode [] input, DataflowNode [] gradient, String prefix,
			int t1_a_1x1, int t1_b_3x3,
			int t2_a_1x1, int t1_b_1x7, int t2_c_7x1, int t2_d_3x3,
			PoolMethod t3_a_3x3) {
		
		int N = 3; /* 3 towers */
		
		DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		DataflowNode [][] tower = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			tower[i] = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [][] towergradient = new DataflowNode [N][];
		for (int i = 0; i < N; i++)
			towergradient[i] = new DataflowNode [] { merge };
		
		DataflowNode concat, concat_;
		
		/* Tower 0 */
		tower[0] = InceptionUnit (tower[0], towergradient[0], String.format("%s-tower-1(a)-1x1", prefix), t1_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[0] = InceptionUnit (tower[0], towergradient[0], String.format("%s-tower-1(b)-3x3", prefix), t1_b_3x3, new int [] {3, 3}, new int [] {2, 2}, new int [] {0, 0});
		/* Tower 1 */
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(a)-1x1", prefix), t2_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(b)-1x7", prefix), t1_b_1x7, new int [] {1, 7}, new int [] {1, 1}, new int [] {0, 3});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(c)-7x1", prefix), t2_c_7x1, new int [] {7, 1}, new int [] {1, 1}, new int [] {3, 0});
		tower[1] = InceptionUnit (tower[1], towergradient[1], String.format("%s-tower-2(d)-3x3", prefix), t2_d_3x3, new int [] {3, 3}, new int [] {2, 2}, new int [] {0, 0});
		/* Tower 2 */
		tower[2] = inceptionPool (tower[2], towergradient[2], String.format("%s-tower-3(a)-3x3", prefix), t3_a_3x3, 3, 2, 0);
		
		concat  = new DataflowNode  (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		concat_ = concat.shallowCopy ();
		
		/* Wire forward nodes */
		for (int i = 0; i < N; i++) {
			tower[i][0].connectTo(concat );
			tower[i][1].connectTo(concat_);
		}
		
		input [0] = concat;
		input [1] = concat_;
		
		/* Wire backward nodes */
		DataflowNode [] concatgradient = new DataflowNode [N];
		for (int i = 0; i < N; i++) {
			concatgradient[i] = new DataflowNode 
					(new Operator (String.format("%s-tower-%d-ConcatGradient",  prefix, (i + 1)), 
							new ConcatGradient (new ConcatConf ().setOffset(i))).setPeer(concat.getOperator()));
			
			concatgradient[i].connectTo(towergradient [i][0]);
			gradient[i] = concatgradient [i];
		}
		/* Set (any) remaining gradients */
		if (N < gradient.length)
			for (int i = N; i < gradient.length; i++)
				gradient[i] = null;
		
		return input;
	}

	public static DataflowNode [] Inception7E (DataflowNode [] input, DataflowNode [] gradient, String prefix,
			int t1_a_1x1,
			int t2_a_1x1, int t2_b_1x3, int t2_c_3x1,
			int t3_a_1x1, int t3_b_3x3, int t3_c_1x3, int t3_d_3x1,
			PoolMethod t4_a_3x3, int t4_b_1x1) {
		
		DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		DataflowNode [] tower_1b0 = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [] tower_2b0 = new DataflowNode [] { input[0], input[1] };
		DataflowNode [] tower_2b1 = new DataflowNode [] { null, null };
		DataflowNode [] tower_2b2 = new DataflowNode [] { null, null };
		
		DataflowNode [] tower_3b0 = new DataflowNode [] { input[0], input[1] };
		DataflowNode [] tower_3b1 = new DataflowNode [] { null, null };
		DataflowNode [] tower_3b2 = new DataflowNode [] { null, null };
		
		DataflowNode [] tower_4b0 = new DataflowNode [] { input[0], input[1] };
		
		DataflowNode [] tower_1b0_gradient = new DataflowNode [] { merge };
		
		DataflowNode [] tower_2b0_gradient = new DataflowNode [] { merge };
		
		DataflowNode tower_2_merge = new DataflowNode (new Operator (String.format("%s-tower-2-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		DataflowNode [] tower_2b1_gradient = new DataflowNode [] { tower_2_merge };
		DataflowNode [] tower_2b2_gradient = new DataflowNode [] { tower_2_merge };
		
		DataflowNode [] tower_3b0_gradient = new DataflowNode [] { merge };
		
		DataflowNode tower_3_merge = new DataflowNode (new Operator (String.format("%s-tower-3-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		DataflowNode [] tower_3b1_gradient = new DataflowNode [] { tower_3_merge };
		DataflowNode [] tower_3b2_gradient = new DataflowNode [] { tower_3_merge };
		
		DataflowNode [] tower_4b0_gradient = new DataflowNode [] { merge };
		
		DataflowNode concat, concat_;
		
		/* Tower 1 of 4 */
		tower_1b0 = InceptionUnit (tower_1b0, tower_1b0_gradient, String.format("%s-tower-1(a)-1x1", prefix), t1_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		
		/* Tower 2 of 4 */
		tower_2b0 = InceptionUnit (tower_2b0, tower_2b0_gradient, String.format("%s-tower-2(a)-1x1", prefix), t2_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		
		tower_2_merge.connectTo (tower_2b0_gradient[0]);
		
		tower_2b1[0] = tower_2b0[0];
		tower_2b1[1] = tower_2b0[1];
		
		tower_2b2[0] = tower_2b0[0];
		tower_2b2[1] = tower_2b0[1];
		
		tower_2b1 = InceptionUnit (tower_2b1, tower_2b1_gradient, String.format("%s-Tower-2(b)-1x3", prefix), t2_b_1x3, new int [] {1, 3}, new int [] {1, 1}, new int [] {0, 1});
		tower_2b2 = InceptionUnit (tower_2b2, tower_2b2_gradient, String.format("%s-Tower-2(c)-3x1", prefix), t2_c_3x1, new int [] {3, 1}, new int [] {1, 1}, new int [] {1, 0});
		
		/* Tower 3 of 4 */
		tower_3b0 = InceptionUnit (tower_3b0, tower_3b0_gradient, String.format("%s-Tower-3(a)-1x1", prefix), t3_a_1x1, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		tower_3b0 = InceptionUnit (tower_3b0, tower_3b0_gradient, String.format("%s-Tower-3(b)-3x3", prefix), t3_b_3x3, new int [] {3, 3}, new int [] {1, 1}, new int [] {1, 1});
		
		tower_3_merge.connectTo (tower_3b0_gradient[0]);
		
		tower_3b1[0] = tower_3b0[0];
		tower_3b1[1] = tower_3b0[1];
		
		tower_3b2[0] = tower_3b0[0];
		tower_3b2[1] = tower_3b0[1];
		
		tower_3b1 = InceptionUnit (tower_3b1, tower_3b1_gradient, String.format("%s-tower-3(c)-1x3", prefix), t3_c_1x3, new int [] {1, 3}, new int [] {1, 1}, new int [] {0, 1});
		tower_3b2 = InceptionUnit (tower_3b2, tower_3b2_gradient, String.format("%s-tower-3(d)-3x1", prefix), t3_d_3x1, new int [] {3, 1}, new int [] {1, 1}, new int [] {1, 0});
		
		/* Tower 4 of 4 */
		tower_4b0 = inceptionPool (tower_4b0, tower_4b0_gradient, String.format("%s-Tower-4(a)-3x3", prefix), t4_a_3x3, 3, 1, 1);
		tower_4b0 = InceptionUnit (tower_4b0, tower_4b0_gradient, String.format("%s-Tower-4(b)-1x1", prefix), t4_b_1x1, new int [] {1, 1}, new int []{1, 1}, new int [] {0, 0});

		concat  = new DataflowNode  (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		concat_ = concat.shallowCopy ();
		
		/* Wire forward nodes */
		tower_1b0[0].connectTo(concat );
		tower_1b0[1].connectTo(concat_);
		
		tower_2b1[0].connectTo(concat );
		tower_2b1[1].connectTo(concat_);
		
		tower_2b2[0].connectTo(concat );
		tower_2b2[1].connectTo(concat_);
		
		tower_3b1[0].connectTo(concat );
		tower_3b1[1].connectTo(concat_);
		
		tower_3b2[0].connectTo(concat );
		tower_3b2[1].connectTo(concat_);
		
		tower_4b0[0].connectTo(concat );
		tower_4b0[1].connectTo(concat_);
		
		input [0] = concat;
		input [1] = concat_;
		
		/* Wire backward nodes */
		DataflowNode [] concatgradient = new DataflowNode [6];
		
		concatgradient[0] = new DataflowNode (new Operator (String.format("%s-tower-1(a)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(0))).setPeer(concat.getOperator()));
		concatgradient[1] = new DataflowNode (new Operator (String.format("%s-tower-2(b)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(1))).setPeer(concat.getOperator()));
		concatgradient[2] = new DataflowNode (new Operator (String.format("%s-tower-2(c)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(2))).setPeer(concat.getOperator()));
		concatgradient[3] = new DataflowNode (new Operator (String.format("%s-tower-3(c)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(3))).setPeer(concat.getOperator()));
		concatgradient[4] = new DataflowNode (new Operator (String.format("%s-tower-3(d)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(4))).setPeer(concat.getOperator()));
		concatgradient[5] = new DataflowNode (new Operator (String.format("%s-tower-4(a)-ConcatGradient",  prefix), new ConcatGradient (new ConcatConf ().setOffset(5))).setPeer(concat.getOperator()));
		
		concatgradient[0].connectTo(tower_1b0_gradient [0]);
		concatgradient[1].connectTo(tower_2b1_gradient [0]);
		concatgradient[2].connectTo(tower_2b2_gradient [0]);
		concatgradient[3].connectTo(tower_3b1_gradient [0]);
		concatgradient[4].connectTo(tower_3b2_gradient [0]);
		concatgradient[5].connectTo(tower_4b0_gradient [0]);
		
		for (int i = 0; i < 6; i++)
			gradient[i] = concatgradient [i];

		return input;
	}
	
	public static SubGraph [] buildInceptionImageNet (SolverConf solverConf) {

		int numberOfOutputs = 10; // 1000
		
		SubGraph [] graphs = new SubGraph [] { null, null }; /* Return value */
		
		DataflowNode [] head = new DataflowNode [] { null, null };
		DataflowNode [] node = new DataflowNode [] { null, null };
		
		/* An inception block can produce up to 6 gradients downstream */
		DataflowNode [] gradient = new DataflowNode [6];
		for (int i = 0; i < gradient.length; i++)
			gradient[0] = null;

		DataflowNode datatransform, innerproduct, softmax, softmax_, loss, accuracy, optimiser;

		DataflowNode innerproductgradient, lossgradient;
		
		DataTransformConf datatransformConf;
		InnerProductConf   innerproductConf;
		SoftMaxConf             softmaxConf;
		LossConf                   lossConf;
		AccuracyConf           accuracyConf;
		
		datatransformConf = new DataTransformConf ();
		/* datatransformConf.setCropSize (299); */
		/* datatransformConf.setMeanImageFilename ("/data/crossbow/imagenet10/imagenet-train.mean"); */
		
		/* First operator */
		datatransform = new DataflowNode (new Operator("DataTransform", new DataTransform(datatransformConf)));
		
		/* Last operator */
		optimiser = new DataflowNode(new Operator("Optimiser", new GradientDescentOptimiser(solverConf)));
		
		/* Wire forward nodes */
		head[0] = datatransform;
		node[0] = head[0];
		
		/* Create a shallow copy */
		head[1] = new DataflowNode(datatransform.getOperator());
		node[1] = head[1];
		
		/* Wire backward nodes */
		gradient[0] = optimiser;
		
		/* Stage 1 */
		node = InceptionUnit (node, gradient, "stage-1-unit-1",  32, new int [] {3, 3}, new int [] {2, 2}, new int [] {0, 0});
		node = InceptionUnit (node, gradient, "stage-1-unit-2",  32, new int [] {3, 3}, new int [] {1, 1}, new int [] {0, 0});
		node = InceptionUnit (node, gradient, "stage-1-unit-3",  64, new int [] {3, 3}, new int [] {1, 1}, new int [] {1, 1});
		
		node = inceptionPool (node, gradient, "stage-1", PoolMethod.MAX, 3, 2, 0);
		
		/* Stage 2 */
		node = InceptionUnit (node, gradient, "stage-2-unit-1",  80, new int [] {1, 1}, new int [] {1, 1}, new int [] {0, 0});
		node = InceptionUnit (node, gradient, "stage-2-unit-2", 192, new int [] {3, 3}, new int [] {1, 1}, new int [] {0, 0});
		
		node = inceptionPool (node, gradient, "stage-2", PoolMethod.MAX, 3, 2, 0);
		
		/* Stage 3 */
		node = Inception7A (node, gradient, "stage-3-7A-1",  64, 64, 96, 96, 48, 64, PoolMethod.AVERAGE, 32);
		node = Inception7A (node, gradient, "stage-3-7A-2",  64, 64, 96, 96, 48, 64, PoolMethod.AVERAGE, 64);
		node = Inception7A (node, gradient, "stage-3-7A-3",  64, 64, 96, 96, 48, 64, PoolMethod.AVERAGE, 64);
		node = Inception7B (node, gradient, "stage-3-7B-1", 384, 64, 96, 96, PoolMethod.MAX);
		
		/* Stage 4 */
		node = Inception7C (node, gradient, "stage-4-7C-1", 192, 128, 128, 192, 128, 128, 128, 128, 192, PoolMethod.MAX,     192);
		node = Inception7C (node, gradient, "stage-4-7C-2", 192, 160, 160, 192, 160, 160, 160, 160, 192, PoolMethod.AVERAGE, 192);
		node = Inception7C (node, gradient, "stage-4-7C-3", 192, 160, 160, 192, 160, 160, 160, 160, 192, PoolMethod.AVERAGE, 192);
		node = Inception7C (node, gradient, "stage-4-7C-4", 192, 192, 192, 192, 192, 192, 192, 192, 192, PoolMethod.AVERAGE, 192);
		node = Inception7D (node, gradient, "stage-4-7D-1", 192, 320, 192, 192, 192, 192, PoolMethod.MAX);
		
		/* Stage 5 */
		node = Inception7E (node, gradient, "stage-5-7E-1", 320, 384, 384, 384, 448, 384, 384, 384, PoolMethod.AVERAGE, 192);
		node = Inception7E (node, gradient, "stage-5-7E-2", 320, 384, 384, 384, 448, 384, 384, 384, PoolMethod.MAX, 192);
		
		/* Wrapping up... */
		/*
		DataflowNode merge = new DataflowNode (new Operator (String.format("Wrap-up"), new ElementWiseOp (new ElementWiseOpConf ())));
		
		for (int i = 0; i < gradient.length; i++)
			if (gradient[i] != null)
				merge.connectTo (gradient[i]);
		
		gradient[0] = merge;
		if (gradient.length > 1)
			for (int i = 1; i < gradient.length; i++)
				gradient[i] = null;
		*/
		
		node = inceptionPool (node, gradient, "Wrap-up", PoolMethod.AVERAGE, 8, 1, 0);
		
		innerproductConf = new InnerProductConf ().setNumberOfOutputs (numberOfOutputs);
		
		innerproductConf.setWeightInitialiser (new InitialiserConf().setType(initialiserType).setStd (0.1F));
		innerproductConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT));
		
		softmaxConf = new SoftMaxConf();
		
		lossConf = new LossConf();
		
		accuracyConf = new AccuracyConf ();
		
		innerproduct = new DataflowNode (new Operator("InnerProduct", new InnerProduct (innerproductConf)));
		softmax      = new DataflowNode (new Operator("SoftMax",      new SoftMax           (softmaxConf)));
		loss         = new DataflowNode (new Operator("SoftMaxLoss",  new SoftMaxLoss          (lossConf)));
		accuracy     = new DataflowNode (new Operator("Accuracy",     new Accuracy         (accuracyConf)));
		
		innerproductgradient = new DataflowNode (new Operator ("InnerProductGradient", new InnerProductGradient (innerproductConf)).setPeer(innerproduct.getOperator()));
		lossgradient         = new DataflowNode (new Operator ("SoftMaxLossGradient",  new SoftMaxLossGradient          (lossConf)).setPeer(        loss.getOperator()));
		
		/* Wire forward nodes */
		node[0] = node[0].connectTo(innerproduct).connectTo(softmax).connectTo(loss);
		/* Connect the accuracy node */
		softmax.connectTo (accuracy);
		
		/* Create a shallow copy */
		softmax_ = new DataflowNode(softmax.getOperator());
		
		node[1] = node[1]
				.connectTo(new DataflowNode(innerproduct.getOperator()))
				.connectTo(softmax_)
				.connectTo(new DataflowNode(loss.getOperator()));
		
		softmax_.connectTo(new DataflowNode(accuracy.getOperator()));
		
		/* Wire backward nodes */
		lossgradient.connectTo(innerproductgradient).connectTo(gradient[0]);
		gradient[0] = lossgradient;
		
		/* Complete training dataflow */

		node[0] = node[0].connectTo(gradient[0]);

		graphs[0] = new SubGraph(head[0]);
		graphs[1] = new SubGraph(head[1]);

		/* Return graphs */
		return graphs;
	}
	
	public static void main (String [] args) throws Exception {
		
		long startTime, dt;
		
		Options options = new Options (LeNet.class.getName());
		
		options.addOption ("--N",             "Train for N units", true,  Integer.class,     "2");
		options.addOption ("--training-unit", "Training unit",     true,   String.class, "epochs");
		options.addOption ("--target-loss",   "Target loss",       false,   Float.class,     "0");
		
		
		CommandLine commandLine = new CommandLine (options);
		
		int numreplicas = 1;
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (numreplicas)
			.setNumberOfGPUStreams (numreplicas)
			.setNumberOfGPUCallbackHandlers (8)
			.setDisplayInterval (100)
			.displayAccumulatedLossValue (true)
			.queueMeasurements (true)
			.setGPUDevices(1)
			.setMaxNumberOfMappedPartitions(4)
			.allowMemoryReuse(true)
			.setRandomSeed(123456789);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (32).setWpc (1).setSlack (0).setUpdateModel(UpdateModel.DEFAULT);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.045F)
			.setMomentum (0.9F)
			.setWeightDecay(0.0001F);
			/* .setGamma(0.1F).setStepValues(32000, 48000); */
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		
		
		int N = options.getOption("--N").getIntValue ();
		TrainingUnit unit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		String dir = String.format("/data/crossbow/imagenet10/b-%03d-299/", ModelConf.getInstance().getBatchSize());
		
		Dataset ds1 = new Dataset (dir + "imagenet-train.metadata");
		Dataset ds2 = new Dataset (dir + "imagenet-train.metadata");
		
		startTime = System.nanoTime ();
		
		/* Load dataset(s) */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, ds1).setDataset (Phase.CHECK, ds2);
		
		/* Build dataflows for both training and testing phases */
		SubGraph [] graphs = buildInceptionImageNet (ModelConf.getInstance ().getSolverConf ());

		/* Set dataflow */
		Dataflow [] dataflows = new Dataflow [2];

		dataflows [0] = new Dataflow (graphs [0]).setPhase(Phase.TRAIN);
		dataflows [1] = new Dataflow (graphs [1]).setPhase(Phase.CHECK); 
		
		ExecutionContext context = new ExecutionContext (dataflows);
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();

		try {
			
			context.init();

			context.getDataflow(Phase.TRAIN).dump();
			context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
			/* context.getDataflow(Phase.TRAIN).exportDot("/home/akolious/16-crossbow/inceptionv3.dot"); */
			
			/* context.getDataflow(Phase.CHECK).dump(); */
		
			context.getModel().dump();
		
			if (N > 0){
				context.trainAndTest(N, unit);
			}
		
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));

		if (SystemConf.getInstance().queueMeasurements())
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();

		context.destroy();

		System.out.println("Bye.");
		System.exit(0);
	}
}
