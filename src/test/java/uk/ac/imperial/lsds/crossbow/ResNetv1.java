package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.cli.CommandLine;
import uk.ac.imperial.lsds.crossbow.cli.Options;
import uk.ac.imperial.lsds.crossbow.kernel.*;
import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.*;

public class ResNetv1 {

	public static final String usage = "usage: ResNetv1";

	private final static Logger log = LogManager.getLogger (ResNetv1.class);
	
	/*
	 * TODO 
	 * Remove static variable by passing `int [] numberOfInputs = new int [1]`
	 * to functions
	 */
	private static InitialiserType initialiserType = InitialiserType.XAVIER;
	
	private static int nInputPlane;
	
	/* Batch-normalisation hyper-parameters */
	private static double EPSILON = 0.00001D;
	private static double ALPHA   = 0.9D;
	
	public static DataflowNode [] buildResNetShortcut (DataflowNode [] input, String prefix, boolean match, int numberofoutputs, int stride, DataflowNode [] gradient) {
		
		if (! match) {

			ConvConf convConf = new ConvConf ();
			
			convConf.setNumberOfOutputs (numberofoutputs);

			convConf.setKernelSize  (2).setKernelHeight  (  1   ).setKernelWidth  (  1   );
			convConf.setStrideSize  (2).setStrideHeight  (stride).setStrideWidth  (stride);
			convConf.setPaddingSize (2).setPaddingHeight (  0   ).setPaddingWidth (  0   );
			
			convConf.setWeightInitialiser 
				(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
			
			/* Explicitly state there is no bias term */
			convConf.setBias (false);
			
			BatchNormConf normConf = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
			normConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
			normConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

			DataflowNode conv = new DataflowNode (new Operator (String.format("%s-Conv (shortcut)",      prefix), new Conv      (convConf)));
			DataflowNode norm = new DataflowNode (new Operator (String.format("%s-BatchNorm (shortcut)", prefix), new BatchNorm (normConf)));

			DataflowNode convgradient = new DataflowNode (new Operator (String.format("%s-ConvGradient (shortcut)",      prefix), new ConvGradient      (convConf)).setPeer (conv.getOperator()));
			DataflowNode normgradient = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (shortcut)", prefix), new BatchNormGradient (normConf)).setPeer (norm.getOperator()));
			
			/* Wire forward nodes */
			input [0] = input [0].connectTo(conv).connectTo(norm);
			
			/* Create a shallow copy */
			input [1] = input [1].connectTo(conv.shallowCopy()).connectTo(norm.shallowCopy());
			
			gradient [0] = gradient [0].connectTo(normgradient).connectTo (convgradient);
		}

		return input;
	}

	public static DataflowNode [] buildResNetBlock (int features, int stride, DataflowNode [] input, String prefix, boolean bottleneck, DataflowNode [] gradient) {

		if (bottleneck)
			return buildResNetBottleneckedUnit (features, stride, input, prefix, gradient);
		else
			return buildResNetBasicUnit (features, stride, input, prefix, gradient);
	}

	public static DataflowNode [] buildResNetBasicUnit (int n, int stride, DataflowNode [] input, String prefix, DataflowNode [] gradient) {
		
		DataflowNode sum, sumgradientLeft, sumgradientRight = null;
		
		/* Shallow copies */
		DataflowNode _sum, _relu;

		DataflowNode [] conv = new DataflowNode [2];
		DataflowNode [] norm = new DataflowNode [2];
		DataflowNode [] relu = new DataflowNode [2];

		DataflowNode merge = null;

		DataflowNode [] convgradient = new DataflowNode [2];
		DataflowNode [] normgradient = new DataflowNode [2];
		DataflowNode [] relugradient = new DataflowNode [2];

		ElementWiseOpConf elementWiseOpConf;

		ConvConf      [] convConf = new      ConvConf [2]; 
		BatchNormConf [] normConf = new BatchNormConf [2]; 
		ReLUConf      [] reluConf = new      ReLUConf [2];

		convConf [0] = new ConvConf ();

		convConf [0].setNumberOfOutputs (n);

		convConf [0].setKernelSize  (2).setKernelHeight  (  3   ).setKernelWidth  (  3   );
		convConf [0].setStrideSize  (2).setStrideHeight  (stride).setStrideWidth  (stride);
		convConf [0].setPaddingSize (2).setPaddingHeight (  1   ).setPaddingWidth (  1   );

		convConf [0].setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf [0].setBias (false);

		normConf [0] = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		normConf [0].setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf [0].setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		reluConf [0] = new ReLUConf ();

		conv [0] = new DataflowNode (new Operator (String.format("%s-Conv (a)",      prefix), new Conv      (convConf [0])));
		norm [0] = new DataflowNode (new Operator (String.format("%s-BatchNorm (a)", prefix), new BatchNorm (normConf [0])));
		relu [0] = new DataflowNode (new Operator (String.format("%s-ReLU (a)",      prefix), new ReLU      (reluConf [0])));

		convgradient [0] = new DataflowNode (new Operator (String.format("%s-ConvGradient (a)",      prefix), new ConvGradient      (convConf [0])).setPeer(conv [0].getOperator()));
		normgradient [0] = new DataflowNode (new Operator (String.format("%s-BatchNormgradient (a)", prefix), new BatchNormGradient (normConf [0])).setPeer(norm [0].getOperator()));
		relugradient [0] = new DataflowNode (new Operator (String.format("%s-ReLUGradient (a)",      prefix), new ReLUGradient      (reluConf [0])).setPeer(relu [0].getOperator()));

		convConf [1] = new ConvConf ();

		convConf [1].setNumberOfOutputs (n);

		convConf [1].setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf [1].setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convConf [1].setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);

		convConf [1].setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf [1].setBias (false);

		normConf [1] = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		normConf [1].setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf [1].setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		reluConf [1] = new ReLUConf ();

		conv [1] = new DataflowNode (new Operator (String.format("%s-Conv (b)",      prefix), new Conv      (convConf [1])));
		norm [1] = new DataflowNode (new Operator (String.format("%s-BatchNorm (b)", prefix), new BatchNorm (normConf [1])));
		relu [1] = new DataflowNode (new Operator (String.format("%s-ReLU (b)",      prefix), new ReLU      (reluConf [1])));
		
		/* Create a shallow copy of the last `relu` node */
		_relu = relu [1].shallowCopy();
		
		convgradient [1] = new DataflowNode (new Operator (String.format("%s-ConvGradient (b)",      prefix), new ConvGradient      (convConf [1])).setPeer(conv [1].getOperator()));
		normgradient [1] = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (b)", prefix), new BatchNormGradient (normConf [1])).setPeer(norm [1].getOperator()));
		relugradient [1] = new DataflowNode (new Operator (String.format("%s-ReLUGradient (b)",      prefix), new ReLUGradient      (reluConf [1])).setPeer(relu [1].getOperator()));

		elementWiseOpConf = new ElementWiseOpConf ();

		sum = new DataflowNode (new Operator (String.format("%s-Sum", prefix), new ElementWiseOp (elementWiseOpConf)));
		
		/* Create a shallow copy of `sum` node */
		_sum = sum.shallowCopy();
		
		/* 
		 * Unroll element-wise gradient operator
		 */
		sumgradientLeft  = new DataflowNode (new Operator (String.format("%s-SumGradientLeft",  prefix), new ElementWiseOpGradient (new ElementWiseOpConf())).setPeer(sum.getOperator()));
		sumgradientRight = new DataflowNode (new Operator (String.format("%s-SumGradientRight", prefix), new ElementWiseOpGradient (new ElementWiseOpConf())).setPeer(sum.getOperator()));

		/* Also, create merge operator (for bottleneck) */
		merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));

		/* Wire forward nodes */
		input [0]
				.connectTo (conv [0])
				.connectTo (norm [0])
				.connectTo (relu [0])
				.connectTo (conv [1])
				.connectTo (norm [1])
				.connectTo (sum)
				.connectTo (relu [1]);
		
		/* Create a shallow copy */
		input [1]
				.connectTo (conv [0].shallowCopy())
				.connectTo (norm [0].shallowCopy())
				.connectTo (relu [0].shallowCopy())
				.connectTo (conv [1].shallowCopy())
				.connectTo (norm [1].shallowCopy())
				.connectTo (_sum)
				.connectTo (_relu);
		
		/* Wire backward nodes */
		relugradient[1]
				.connectTo(sumgradientLeft)
				.connectTo(normgradient [1])
				.connectTo(convgradient [1])
				.connectTo(relugradient [0])
				.connectTo(normgradient [0])
				.connectTo(convgradient [0])
				.connectTo(merge)
				.connectTo(gradient[0]);
		
		/* Since we unrolled sumgradient, create another connection */
		relugradient[1].connectTo(sumgradientRight);
		
		/* Configure shortcut */
		boolean match = (nInputPlane == n);
		DataflowNode [] g = new DataflowNode [] { sumgradientRight };
		DataflowNode [] shortcut = buildResNetShortcut (input, prefix, match, n, stride, g);
		
		/* Update nInputPlane */
		nInputPlane = n;
		
		/* Wire forward nodes */
		shortcut [0].connectTo( sum);
		shortcut [1].connectTo(_sum);
		/* Wire backward nodes */
		g [0].connectTo(merge);
		
		/* Set current value for gradient */
		gradient[0] = relugradient [1];
		
		/* Set current value for input */
		input [0] =  relu [1];
		input [1] = _relu;
		
		return input;
	}
	
	public static DataflowNode [] buildResNetBottleneckedUnit (int n, int stride, DataflowNode [] input, String prefix, DataflowNode [] gradient) {
		
		log.debug(String.format("Bottleneck depth %4d input filters %4d stride %d %s", n, nInputPlane, stride, (nInputPlane != (n * 4)) ? "Create shortcut" : ""));

		DataflowNode sum, sumgradientLeft, sumgradientRight = null;
		
		/* Shallow copies */
		DataflowNode _sum, _relu;

		DataflowNode [] conv = new DataflowNode [3];
		DataflowNode [] norm = new DataflowNode [3];
		DataflowNode [] relu = new DataflowNode [3];

		DataflowNode merge = null;

		DataflowNode [] convgradient = new DataflowNode [3];
		DataflowNode [] normgradient = new DataflowNode [3];
		DataflowNode [] relugradient = new DataflowNode [3];

		ElementWiseOpConf elementWiseOpConf;

		ConvConf      [] convConf = new      ConvConf [3]; 
		BatchNormConf [] normConf = new BatchNormConf [3]; 
		ReLUConf      [] reluConf = new      ReLUConf [3];

		convConf [0] = new ConvConf ();

		convConf [0].setNumberOfOutputs (n);

		convConf [0].setKernelSize (2).setKernelHeight (1).setKernelWidth (1);
		convConf [0].setStrideSize (2).setStrideHeight (stride).setStrideWidth (stride);
		
		convConf [0].setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf [0].setBias (false);

		normConf [0] = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		normConf [0].setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf [0].setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		reluConf [0] = new ReLUConf ();

		conv [0] = new DataflowNode (new Operator (String.format("%s-Conv (a)",      prefix), new Conv      (convConf[0])));
		norm [0] = new DataflowNode (new Operator (String.format("%s-BatchNorm (a)", prefix), new BatchNorm (normConf[0])));
		relu [0] = new DataflowNode (new Operator (String.format("%s-ReLU (a)",      prefix), new ReLU      (reluConf[0])));

		convgradient [0] = new DataflowNode (new Operator (String.format("%s-ConvGradient (a)",      prefix), new ConvGradient      (convConf [0])).setPeer(conv [0].getOperator()));
		normgradient [0] = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (a)", prefix), new BatchNormGradient (normConf [0])).setPeer(norm [0].getOperator()));
		relugradient [0] = new DataflowNode (new Operator (String.format("%s-ReLUGradient (a)",      prefix), new ReLUGradient      (reluConf [0])).setPeer(relu [0].getOperator()));

		convConf [1] = new ConvConf ();

		convConf [1].setNumberOfOutputs(n);

		convConf [1].setKernelSize  (2).setKernelHeight  (  3   ).setKernelWidth  (  3   );
		convConf [1].setStrideSize  (2).setStrideHeight  (  1   ).setStrideWidth  (  1   );
        convConf [1].setPaddingSize (2).setPaddingHeight (  1   ).setPaddingWidth (  1   );

		convConf [1].setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf [1].setBias (false);

		normConf [1] = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		normConf [1].setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf [1].setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		reluConf [1] = new ReLUConf ();
		
		conv [1] = new DataflowNode (new Operator (String.format("%s-Conv (b)",      prefix), new Conv      (convConf [1])));
		norm [1] = new DataflowNode (new Operator (String.format("%s-BatchNorm (b)", prefix), new BatchNorm (normConf [1])));
		relu [1] = new DataflowNode (new Operator (String.format("%s-ReLU (b)",      prefix), new ReLU      (reluConf [1])));

		convgradient[1] = new DataflowNode (new Operator (String.format("%s-ConvGradient (b)",      prefix), new ConvGradient      (convConf [1])).setPeer(conv [1].getOperator()));
		normgradient[1] = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (b)", prefix), new BatchNormGradient (normConf [1])).setPeer(norm [1].getOperator()));
		relugradient[1] = new DataflowNode (new Operator (String.format("%s-ReLUGradient (b)",      prefix), new ReLUGradient      (reluConf [1])).setPeer(relu [1].getOperator()));

		convConf [2] = new ConvConf ();

		convConf [2].setNumberOfOutputs(n * 4);

		convConf [2].setKernelSize (2).setKernelHeight (1).setKernelWidth (1);
		convConf [2].setStrideSize (2).setStrideHeight (1).setStrideWidth (1);

		convConf [2].setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf [2].setBias (false);

		normConf [2] = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		normConf [2].setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		normConf [2].setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
		
		reluConf [2] = new ReLUConf ();

		conv [2] = new DataflowNode (new Operator (String.format("%s-Conv (c)",      prefix), new Conv      (convConf [2])));
		norm [2] = new DataflowNode (new Operator (String.format("%s-BatchNorm (c)", prefix), new BatchNorm (normConf [2])));
		relu [2] = new DataflowNode (new Operator (String.format("%s-ReLU (c)",      prefix), new ReLU      (reluConf [2])));
		
		/* Create a shallow copy of the last `relu` node */
		_relu = relu [2].shallowCopy();

		convgradient [2] = new DataflowNode (new Operator (String.format("%s-ConvGradient (c)",      prefix), new ConvGradient      (convConf [2])).setPeer(conv [2].getOperator()));
		normgradient [2] = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (c)", prefix), new BatchNormGradient (normConf [2])).setPeer(norm [2].getOperator()));
		relugradient [2] = new DataflowNode (new Operator (String.format("%s-ReLUGradient (c)",      prefix), new ReLUGradient      (reluConf [2])).setPeer(relu [2].getOperator()));

		elementWiseOpConf = new ElementWiseOpConf ();

		sum = new DataflowNode (new Operator (String.format("%s-Sum", prefix), new ElementWiseOp (elementWiseOpConf)));
		
		/* Create a shallow copy of `sum` node */
		_sum = sum.shallowCopy();

		/* 
		 * Unroll element-wise gradient operator
		 * 
		 * TODO Does it matter that sum is being referenced by 
		 * two nodes instead of only one as the peer operator?
		 */
		sumgradientLeft  = new DataflowNode (new Operator (String.format("%s-SumGradientLeft",  prefix), new ElementWiseOpGradient (new ElementWiseOpConf())).setPeer(sum.getOperator()));
		sumgradientRight = new DataflowNode (new Operator (String.format("%s-SumGradientRight", prefix), new ElementWiseOpGradient (new ElementWiseOpConf())).setPeer(sum.getOperator()));
		

		/* Also, create merge operator (for bottleneck) */
		merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
		
		/* Wire forward nodes */
		input[0]
				.connectTo (conv [0])
				.connectTo (norm [0])
				.connectTo (relu [0])
				.connectTo (conv [1])
				.connectTo (norm [1])
				.connectTo (relu [1])
				.connectTo (conv [2])
				.connectTo (norm [2])
				.connectTo (sum)
				.connectTo (relu [2]);
		
		/* Create a shallow copy */
		input[1]
				.connectTo (conv [0].shallowCopy())
				.connectTo (norm [0].shallowCopy())
				.connectTo (relu [0].shallowCopy())
				.connectTo (conv [1].shallowCopy())
				.connectTo (norm [1].shallowCopy())
				.connectTo (relu [1].shallowCopy())
				.connectTo (conv [2].shallowCopy())
				.connectTo (norm [2].shallowCopy())
				.connectTo (_sum)
				.connectTo (_relu);

		/* Wire backward nodes */
		relugradient[2]
				.connectTo(sumgradientLeft)
				.connectTo(normgradient [2])
				.connectTo(convgradient [2])
				.connectTo(relugradient [1])
				.connectTo(normgradient [1])
				.connectTo(convgradient [1])
				.connectTo(relugradient [0])
				.connectTo(normgradient [0])
				.connectTo(convgradient [0])
				.connectTo(merge)
				.connectTo(gradient[0]);
		
		/* Since we unrolled sumgradient, create another connection */
		relugradient[2].connectTo(sumgradientRight);
		
		/* Configure shortcut */
		boolean match = (nInputPlane == (n * 4));
		DataflowNode [] g = new DataflowNode [] { sumgradientRight };
		DataflowNode [] shortcut = buildResNetShortcut (input, prefix, match, n * 4, stride, g);
		
		/* Update nInputPlane */
		nInputPlane = n * 4;
		
		/* Wire forward nodes */
		shortcut [0].connectTo( sum);
		shortcut [1].connectTo(_sum);
		/* Wire backward nodes */
		g [0].connectTo(merge);
		
		/* Set current value for gradient */
		gradient[0] = relugradient [2];
		
		/* Set current value for input */
		input [0] =  relu [2];
		input [1] = _relu;
		
		return input;
	}
	
	public static SubGraph [] buildResNetImageNetDebug (SolverConf solverConf, int numclasses) {
		
		SubGraph [] graphs = new SubGraph [] { null, null }; /* Return value */
		
		int numberOfOutputs = numclasses;
		
		DataflowNode [] head = new DataflowNode [] { null, null };
		DataflowNode [] node = new DataflowNode [] { null, null };
		
		DataflowNode innerproduct, softmax, loss, trainAccuracy;
        DataflowNode innerproductgradient, lossgradient;
		DataflowNode optimiser;
		
		InnerProductConf  innerproductConf = new InnerProductConf ();
        SoftMaxConf            softmaxConf = new SoftMaxConf ();
        LossConf                  lossConf = new LossConf ();
        AccuracyConf     trainAccuracyConf = new AccuracyConf ();
		
		innerproductConf.setNumberOfOutputs (numberOfOutputs);
		innerproductConf.setWeightInitialiser
			(new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setMean(0).setStd(0.0221F).truncate(true));
		innerproductConf.setBiasInitialiser
			(new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));

        innerproduct = new DataflowNode (new Operator ("InnerProduct", new InnerProduct (innerproductConf)));
        softmax      = new DataflowNode (new Operator ("SoftMax",      new SoftMax           (softmaxConf)));
        loss         = new DataflowNode (new Operator ("SoftMaxLoss",  new SoftMaxLoss          (lossConf)));
		
		innerproductgradient = new DataflowNode 
			(new Operator ("InnerProductGradient", 
				new InnerProductGradient (innerproductConf)).setPeer(innerproduct.getOperator()));
		
        lossgradient = new DataflowNode 
			(new Operator ("SoftMaxLossGradient",  
				new SoftMaxLossGradient(lossConf)).setPeer(loss.getOperator()));
        
		trainAccuracy = new DataflowNode (new Operator ("Accuracy", new Accuracy (trainAccuracyConf)));
        optimiser = new DataflowNode (new Operator ("Optimiser", new GradientDescentOptimiser (solverConf)));
		
		/* Wire forward nodes */
		head [0] = innerproduct;
		node [0] = head [0];
        node [0] = node [0].connectTo(softmax).connectTo(loss);

        /* Wire backward nodes */
        node [0] = node [0].connectTo(lossgradient).connectTo(innerproductgradient).connectTo(optimiser);

        /* Complete training dataflow */
        softmax.connectTo(trainAccuracy);

        graphs [0] = new SubGraph (head [0]);
		
		return graphs;
	}

	public static SubGraph [] buildResNetImageNet (int [] blocks, int [] features, SolverConf solverConf, boolean bottleneck, int numclasses, boolean transform) {
		
		int numberOfOutputs = numclasses;

		SubGraph [] graphs = new SubGraph [] { null, null }; /* Return value */

		/* The tail of the dataflow points to gradient of the first operator 
		 * (the head) of the training dataflow. 
		 */
		DataflowNode tail = null;
		
		DataflowNode [] head = new DataflowNode [] { null, null };
		DataflowNode [] node = new DataflowNode [] { null, null };
		
		DataflowNode [] gradient = new DataflowNode [1];
		
		DataflowNode datatransform = null, conv, batchnorm, relu, pool, innerproduct, softmax, loss, trainAccuracy;

		DataflowNode convgradient, batchnormgradient, relugradient, poolgradient, innerproductgradient, lossgradient;

		DataTransformConf datatransformConf = null;
		ConvConf                   convConf;
		BatchNormConf         batchnormConf;
		ReLUConf                   reluConf;
		PoolConf                   poolConf;
		InnerProductConf   innerproductConf;
		SoftMaxConf             softmaxConf;
		LossConf                   lossConf;
		AccuracyConf      trainAccuracyConf;
		
		if (transform) { 
			datatransformConf = new DataTransformConf ();
			datatransformConf.setCropSize(224);
			/* TODO Configure correct path for mean image values */
			datatransformConf.setMeanImageFilename("/data/crossbow/imagenet/ilsvrc2012/imagenet-train.mean");
		}
		
		convConf = new ConvConf ();

		convConf.setNumberOfOutputs (64);
		convConf.setKernelSize  (2).setKernelHeight  (7).setKernelWidth  (7);
		convConf.setStrideSize  (2).setStrideHeight  (2).setStrideWidth  (2);
		/* 
		 * Removed padding for TF comparisons...
		 */
		convConf.setPaddingSize (2).setPaddingHeight (3).setPaddingWidth (3);
		/* */
		convConf.setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));
		
		/* Explicitly state there is no bias term */
		convConf.setBias (false);

		batchnormConf = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		batchnormConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		batchnormConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		reluConf = new ReLUConf ();

		poolConf = new PoolConf ().setMethod (PoolMethod.MAX);
		poolConf.setKernelSize (3); // Change 3 to 2?
		poolConf.setStrideSize (2);
		/*
		 * Removed padding for TF comparisons...
		 */
		 // poolConf.setPaddingSize(1);
		 /* */
		poolConf.setPaddingSize(0);
		
		if (transform)
			datatransform = new DataflowNode (new Operator ("DataTransform", new DataTransform (datatransformConf)));
		
		conv          = new DataflowNode (new Operator ("Conv",          new Conv                   (convConf)));
		batchnorm     = new DataflowNode (new Operator ("BatchNorm",     new BatchNorm         (batchnormConf)));
		relu          = new DataflowNode (new Operator ("ReLU",          new ReLU                   (reluConf)));
		pool          = new DataflowNode (new Operator ("Pool",          new Pool                   (poolConf)));

		convgradient      = new DataflowNode (new Operator ("ConvGradient",      new ConvGradient           (convConf)).setPeer(     conv.getOperator()));
		batchnormgradient = new DataflowNode (new Operator ("BatchNormGradient", new BatchNormGradient (batchnormConf)).setPeer(batchnorm.getOperator()));
		relugradient      = new DataflowNode (new Operator ("ReLUGradient",      new ReLUGradient           (reluConf)).setPeer(     relu.getOperator()));
		poolgradient      = new DataflowNode (new Operator ("PoolGradient",      new PoolGradient           (poolConf)).setPeer(     pool.getOperator()));
		
		
		if (transform) {
			/* Wire forward nodes */
			head [0] = datatransform;
			node [0] = head [0].connectTo(conv).connectTo(batchnorm).connectTo(relu).connectTo(pool);
			
			/* Create a shallow copy */
			head [1] = new DataflowNode (datatransform.getOperator());
			node [1] = head [1]
					.connectTo(new DataflowNode (     conv.getOperator()))
					.connectTo(new DataflowNode (batchnorm.getOperator()))
					.connectTo(new DataflowNode (     relu.getOperator()))
					.connectTo(new DataflowNode (     pool.getOperator()));
		}
		else {
			/* Wire forward nodes */
			head [0] = conv;
			node [0] = head [0].connectTo(batchnorm).connectTo(relu).connectTo(pool);
			
			/* Create a shallow copy */
			head [1] = new DataflowNode (conv.getOperator());
			node [1] = head [1]
					.connectTo(new DataflowNode (batchnorm.getOperator()))
					.connectTo(new DataflowNode (     relu.getOperator()))
					.connectTo(new DataflowNode (     pool.getOperator()));
		}
		
		/* Wire backward nodes */
		poolgradient.connectTo(relugradient).connectTo(batchnormgradient).connectTo(convgradient);
		gradient[0] = poolgradient;
		tail = convgradient;
		
		String prefix = null;
		int stride = 1;
		
		nInputPlane = convConf.numberOfOutputs();
		
		/* 4 stages */
		for (int stage = 1; stage <= 4; ++stage) {

			for (int unit = 1; unit <= blocks[(stage - 1)]; ++unit) {

				if ((unit == 1) && (stage != 1)) {
					stride = 2;
				} else {
					stride = 1;
				}

				prefix = String.format("stage-%d-unit-%d", stage, unit);
				node = buildResNetBlock (features[stage - 1], stride, node, prefix, bottleneck, gradient);
			}
		}

		poolConf = new PoolConf ().setMethod (PoolMethod.AVERAGE);

		poolConf.setKernelSize(7);
		poolConf.setStrideSize(1);
		poolConf.setPaddingSize(0);

		innerproductConf = new InnerProductConf ();

		innerproductConf.setNumberOfOutputs (numberOfOutputs);

		innerproductConf.setWeightInitialiser 
			(new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setMean(0).setStd(0.0221F).truncate(true));
		innerproductConf.setBiasInitialiser   
			(new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));
		
		softmaxConf = new SoftMaxConf ();

		lossConf = new LossConf ();
		
		trainAccuracyConf = new AccuracyConf ();

		pool         = new DataflowNode (new Operator ("Pool",         new Pool                 (poolConf)));
		innerproduct = new DataflowNode (new Operator ("InnerProduct", new InnerProduct (innerproductConf)));
		softmax      = new DataflowNode (new Operator ("SoftMax",      new SoftMax           (softmaxConf)));
		loss         = new DataflowNode (new Operator ("SoftMaxLoss",  new SoftMaxLoss          (lossConf)));

		poolgradient         = new DataflowNode (new Operator ("PoolGradient",         new PoolGradient                 (poolConf)).setPeer(        pool.getOperator()));
		innerproductgradient = new DataflowNode (new Operator ("InnerProductGradient", new InnerProductGradient (innerproductConf)).setPeer(innerproduct.getOperator()));
		lossgradient         = new DataflowNode (new Operator ("SoftMaxLossGradient",  new SoftMaxLossGradient          (lossConf)).setPeer(        loss.getOperator()));

		/* Wire forward nodes */
		node [0] = node [0].connectTo(pool).connectTo(innerproduct).connectTo(softmax).connectTo(loss);
		
		/* Create a shallow copy */
		node [1] = node [1]
				.connectTo(new DataflowNode (        pool.getOperator()))
				.connectTo(new DataflowNode (innerproduct.getOperator()))
				.connectTo(new DataflowNode (     softmax.getOperator()));
		
		/* Wire backward nodes */
		lossgradient.connectTo(innerproductgradient).connectTo(poolgradient).connectTo(gradient[0]);
		gradient[0] = lossgradient;
		
		/* Complete training dataflow */
		
		node [0] = node [0].connectTo (gradient[0]);
		
		/* Connect the accuracy node */
		trainAccuracy = new DataflowNode (new Operator ("Accuracy", new Accuracy (trainAccuracyConf)));

		softmax.connectTo(trainAccuracy);

		/* Create solver operator */
		DataflowNode optimiser = new DataflowNode (new Operator ("Optimiser", new GradientDescentOptimiser (solverConf)));

		tail.connectTo(optimiser);

		graphs [0] = new SubGraph (head [0]);

		/* Complete testing dataflow */
		
		DataflowNode accuracy = trainAccuracy.shallowCopy(); // new DataflowNode (new Operator ("Accuracy", new Accuracy (accuracyConf)));

		node [1].connectTo(accuracy);
		
		graphs [1] = new SubGraph (head [1]);
		
		/* Return subgraphs */
		return graphs;
	}
	
	public static SubGraph [] buildResNetCifar10 (int blocks, int [] features, SolverConf solverConf) {
		
		int numberOfOutputs = 10;

		SubGraph [] graphs = new SubGraph [] { null, null }; /* Return value */
		
		/* The tail of the dataflow points to gradient of the first operator 
		 * (the head) of the training dataflow. 
		 */
		DataflowNode tail = null;
		
		DataflowNode [] head = new DataflowNode [] { null, null };
		DataflowNode [] node = new DataflowNode [] { null, null };
		
		DataflowNode [] gradient = new DataflowNode [1];
		
		DataflowNode datatransform, conv, batchnorm, relu, pool, innerproduct, softmax, loss, trainAccuracy;
		
		DataflowNode convgradient, batchnormgradient, relugradient, poolgradient, innerproductgradient, lossgradient;

		DataTransformConf datatransformConf;
		ConvConf                   convConf;
		BatchNormConf         batchnormConf;
		ReLUConf                   reluConf;
		PoolConf                   poolConf;
		InnerProductConf   innerproductConf;
		SoftMaxConf             softmaxConf;
		LossConf                   lossConf;
		AccuracyConf      trainAccuracyConf;
		
		datatransformConf = new DataTransformConf ();
		datatransformConf.setCropSize (32);
		datatransformConf.setMirror (true);
		
		convConf = new ConvConf ();
		convConf.setNumberOfOutputs (16);
		convConf.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convConf.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convConf.setWeightInitialiser 
			(new InitialiserConf ().setType (initialiserType).setNorm(VarianceNormalisation.AVG));

		convConf.setBias (false);
		
		batchnormConf = new BatchNormConf ().setEpsilon (EPSILON).setMovingAverageFraction (ALPHA);
		batchnormConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		batchnormConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));
		
		reluConf = new ReLUConf ();
		
		datatransform = new DataflowNode (new Operator ("DataTransform", new DataTransform (datatransformConf)));
		conv          = new DataflowNode (new Operator ("Conv",          new Conv                   (convConf)));
		batchnorm     = new DataflowNode (new Operator ("BatchNorm",     new BatchNorm         (batchnormConf)));
		relu          = new DataflowNode (new Operator ("ReLU",          new ReLU                   (reluConf)));
		
		convgradient      = new DataflowNode (new Operator ("ConvGradient",      new ConvGradient           (convConf)).setPeer(     conv.getOperator()));
		batchnormgradient = new DataflowNode (new Operator ("BatchNormGradient", new BatchNormGradient (batchnormConf)).setPeer(batchnorm.getOperator()));
		relugradient      = new DataflowNode (new Operator ("ReLUGradient",      new ReLUGradient           (reluConf)).setPeer(     relu.getOperator()));
		
		/* Wire forward nodes */
		head [0] = datatransform;
		node [0] = datatransform.connectTo(conv).connectTo(batchnorm).connectTo(relu);
		
		/* Create a shallow copy */
		head [1] = datatransform.shallowCopy();
		node [1] = head [1]
				.connectTo(     conv.shallowCopy())
				.connectTo(batchnorm.shallowCopy())
				.connectTo(     relu.shallowCopy());
		
		/* Wire backward nodes */
		relugradient.connectTo(batchnormgradient).connectTo(convgradient);
		gradient[0] = relugradient;
		tail = convgradient;
		
		String prefix = null;
		int stride = 1;
		
		/* 3 stages */
		for (int stage = 1; stage <= 3; ++stage) {

			for (int unit = 1; unit <= blocks; ++unit) {

				if ((unit == 1) && (stage != 1)) {
					stride = 2;
				} else {
					stride = 1;
				}

				prefix = String.format("stage-%d-unit-%d", stage, unit);
				node = buildResNetBlock (features[stage - 1], stride, node, prefix, false, gradient);
			}
		}
		
		poolConf = new PoolConf ().setMethod(PoolMethod.AVERAGE);
		
		poolConf.setKernelSize  (7); 
		poolConf.setStrideSize  (1);
		poolConf.setPaddingSize (0);
		
		innerproductConf = new InnerProductConf ();

		innerproductConf.setNumberOfOutputs(numberOfOutputs);

		innerproductConf.setWeightInitialiser 
			(new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setMean(0).setStd(1).truncate(true));
		innerproductConf.setBiasInitialiser   
			(new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0));

		softmaxConf = new SoftMaxConf ();

		lossConf = new LossConf ();
		
		trainAccuracyConf = new AccuracyConf ();

		pool         = new DataflowNode (new Operator ("Pool",         new Pool                 (poolConf)));
		innerproduct = new DataflowNode (new Operator ("InnerProduct", new InnerProduct (innerproductConf)));
		softmax      = new DataflowNode (new Operator ("SoftMax",      new SoftMax           (softmaxConf)));
		loss         = new DataflowNode (new Operator ("SoftMaxLoss",  new SoftMaxLoss          (lossConf)));

		poolgradient         = new DataflowNode (new Operator ("PoolGradient",         new PoolGradient                 (poolConf)).setPeer(        pool.getOperator()));
		innerproductgradient = new DataflowNode (new Operator ("InnerProductGradient", new InnerProductGradient (innerproductConf)).setPeer(innerproduct.getOperator()));
		lossgradient         = new DataflowNode (new Operator ("SoftMaxLossGradient",  new SoftMaxLossGradient          (lossConf)).setPeer(        loss.getOperator()));
		
		/* Wire forward nodes */
		node [0] = node [0].connectTo(pool).connectTo(innerproduct).connectTo(softmax).connectTo(loss);
		
		/* Connect the accuracy node */
		trainAccuracy = new DataflowNode (new Operator ("Accuracy", new Accuracy (trainAccuracyConf)));

		softmax.connectTo(trainAccuracy);
		
		/* Create a shallow copy */
		node [1] = node [1]
				.connectTo(new DataflowNode (        pool.getOperator()))
				.connectTo(new DataflowNode (innerproduct.getOperator()))
				.connectTo(new DataflowNode (     softmax.getOperator()));
		
		/* Wire backward nodes */
		lossgradient.connectTo(innerproductgradient).connectTo(poolgradient).connectTo(gradient[0]);
		gradient [0] = lossgradient;
		
		/* Complete training dataflow */
		
		node [0] = node [0].connectTo (gradient[0]);
		
		/* Create solver operator */
		DataflowNode optimiser = new DataflowNode (new Operator ("Optimiser", new GradientDescentOptimiser (solverConf)));
		
		tail.connectTo(optimiser);
		
		graphs [0] = new SubGraph (head [0]);
		
		/* Complete testing dataflow */

		DataflowNode accuracy = trainAccuracy.shallowCopy(); // new DataflowNode (new Operator ("Accuracy", new Accuracy (accuracyConf)));

		node [1].connectTo(accuracy);
		
		graphs [1] = new SubGraph (head [1]);
		
		/* Return sub-graphs */
		return graphs;
	}
	
	public static void main (String [] args) throws Exception {

		/* What version of ResNet we want to build (e.g. ResNet-50) and for which dataset (e.g. imagenet) */
		/* 
		 * ImageNet: 18, 34, 50, 101, 152, ...
		 * 
		 * Cifar-10: 20, 32, 44, 56, 110, ...
		 * 
		 */
		
		long startTime, dt;
		
		Options options = new Options (ResNetv1.class.getName());
		
		options.addOption ("--training-unit",   "Training unit",          true,   String.class,   "epochs");
		options.addOption ("--N",             	"Train for N units",      true,  Integer.class,      "1");
		
		options.addOption ("--time-limit",   	"Time-limited training",  true,  Boolean.class,    "false");
		options.addOption ("--duration-unit",   "Duration time unit",     true,   String.class,  "minutes");
		options.addOption ("--D",   	        "Train for D time units", true,  Integer.class,        "1");
		options.addOption ("--target-loss",   	"Target loss",            false,   Float.class,      "0.0");
		
		options.addOption ("--dataset-name",   	"Dataset name",           true,   String.class, "imagenet");
		options.addOption ("--layers",          "Number of layers",       true,  Integer.class,       "50");
		options.addOption ("--data-directory",  "Data directory",         true,   String.class, "/data/crossbow/imagenet/ilsvrc2012/records/");
		
		/* Override batch-normalisation hyper-parameters */
		EPSILON = 0.00001D;
		ALPHA   = 0.9D;
        
		CommandLine commandLine = new CommandLine (options);
		
		int numberofreplicas = 1;
		int [] devices = new int [] { 0, 1 };
		int wpc = numberofreplicas * devices.length * 1;
		
		/* Override default system configuration (before parsing) */
		SystemConf.getInstance ()
			.setCPU (false)
			.setGPU (true)
			.setNumberOfWorkerThreads (1)
			.setNumberOfCPUModelReplicas (1)
			.setNumberOfReadersPerModel (1)
			.setNumberOfGPUModelReplicas (numberofreplicas)
			.setNumberOfGPUStreams (numberofreplicas)
			.setNumberOfGPUCallbackHandlers (4)
			.setNumberOfGPUTaskHandlers (4)
			.setNumberOfFileHandlers (4)
			.setMaxNumberOfMappedPartitions (1)
			.setDisplayInterval (1)
			.setDisplayIntervalUnit (TrainingUnit.EPOCHS)
			.displayAccumulatedLossValue (true)
			.setRandomSeed (123456789L)
			.queueMeasurements (true)
			.setTaskQueueSizeLimit (32)
			.setPerformanceMonitorInterval (1000)
			.setNumberOfResultSlots (128)
			.setGPUDevices (devices)
			.allowMemoryReuse (true)
			.autotuneModels (false)
			.useDirectScheduling(true);
		
		/* Override default model configuration (before parsing)  */
		ModelConf.getInstance ().setBatchSize (32).setWpc (wpc).setSlack (0).setUpdateModel(UpdateModel.SYNCHRONOUSEAMSGD);
		ModelConf.getInstance ().getSolverConf ().setAlpha (0.1f).setTau (3);
		ModelConf.getInstance ().setTestInterval (1).setTestIntervalUnit (TrainingUnit.EPOCHS);
		
		/* Override default solver configuration (before parsing) */
		ModelConf.getInstance ().getSolverConf ()
			.setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (0.1F)
			.setMomentum (0.9F)
			.setMomentumMethod(MomentumMethod.POLYAK)
			.setWeightDecay(0.00001F)
			.setLearningRateStepUnit(TrainingUnit.EPOCHS)
			.setStepValues(new int [] { 30, 60, 80 })
			.setGamma(0.1F);
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		int __N = options.getOption("--N").getIntValue ();
		TrainingUnit taskunit = TrainingUnit.fromString (options.getOption("--training-unit").getStringValue());
		
		boolean timelimit = options.getOption("--time-limit").getBooleanValue();
		
		int __D = options.getOption("--D").getIntValue ();
		DurationUnit timeunit = DurationUnit.fromString (options.getOption("--duration-unit").getStringValue());
		
		/* Parse command line arguments */
		commandLine.parse (args);
		
		String datasetName = options.getOption("--dataset-name").getStringValue();
		int layers         = Integer.parseInt(options.getOption("--layers").getStringValue());
//		int batch_size     = ModelConf.getInstance().getBatchSize();
		
		SolverConf solverConf = ModelConf.getInstance().getSolverConf ();
		
		/* Build dataflows for both training and testing phases */
		SubGraph [] graphs  = null;
		
		IDataset []  dataset = new IDataset [] { null, null };
		
		/* Set dataflow */

		Dataflow [] dataflows = new Dataflow [2];
				
		if (datasetName.equals("imagenet")) {
			
			System.out.println("Load imagenet data set");
			
			int numclasses = 1000; 
			
			/* Create dataset */
			
			String dataDirectory = options.getOption("--data-directory").getStringValue();
			
			// dataset [0] = new LightWeightDataset (DatasetUtils.buildPath(dataDirectory, "imagenet-train.metadata", true));
			// dataset [1] = new LightWeightDataset (DatasetUtils.buildPath(dataDirectory, "imagenet-test.metadata",  true));
			
			dataset [0] = new RecordDataset (
				DatasetUtils.buildPath(
					"/fast/crossbow/imagenet/train", 
					"imagenet-train.metadata", true
				)
			);
			
			dataset [1] = new RecordDataset (
				DatasetUtils.buildPath(
					"/fast/crossbow/imagenet/validation", 
					"imagenet-test.metadata", true
				)
			);
			
			int [] features = new int [] { 64, 128, 256, 512 }; /* 4 stages */
			
			int [] blocks = null; /* Number of blocks for each stage */
			boolean bottleneck = false;

			if      (layers ==  18) { blocks = new int [] { 2,  2,  2, 2 }; bottleneck = false; }
			else if (layers ==  34) { blocks = new int [] { 3,  4,  6, 3 }; bottleneck = false; }
			else if (layers ==  50) { blocks = new int [] { 3,  4,  6, 3 }; bottleneck =  true; }
			else if (layers == 101) { blocks = new int [] { 3,  4, 23, 3 }; bottleneck =  true; }
			else if (layers == 152) { blocks = new int [] { 3,  8, 36, 3 }; bottleneck =  true; }
			else {
				System.err.println(String.format("error: invalid number of layers for %s dataset", datasetName));
				System.exit(1);
			}
			
			/* Do not transform data by default */
			graphs = buildResNetImageNet (blocks, features, solverConf, bottleneck, numclasses, false);
			
			dataflows [0] = new Dataflow (graphs [0]).setPhase(Phase.TRAIN);
			dataflows [1] = (dataset [1] ==  null) ? null : new Dataflow (graphs [1]).setPhase(Phase.CHECK);
		}
		else if (datasetName.equals("cifar-10")) {
			
			/* Create dataset */
			
			String dataDirectory = options.getOption("--data-directory").getStringValue();
			
			dataset [0] = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
			dataset [1] = new Dataset (DatasetUtils.buildPath(dataDirectory,  "cifar-test.metadata", true));
			
			int [] features = new int [] { 16, 32, 64 }; /* 3 stages */

			if ((layers - 2) % 6 != 0) {
				System.err.println(String.format("error: invalid number of layers for %s dataset", datasetName));
				System.exit(1);
			}
			
			int blocks = (layers - 2) / 6;
			
			graphs = buildResNetCifar10 (blocks, features, solverConf);
			
			dataflows [0] = new Dataflow (graphs [0]).setPhase(Phase.TRAIN);
			dataflows [1] = new Dataflow (graphs [1]).setPhase(Phase.CHECK);
		}
		else {
			System.err.println(String.format("error: invalid dataset name: %s", datasetName));
			System.exit(1);
		}
		
		/* Model dataset configuration */
		ModelConf.getInstance ().setDataset (Phase.TRAIN, dataset[0]).setDataset (Phase.CHECK, dataset[1]);
		
		if (! (wpc > 0))
			throw new IllegalArgumentException();
		
		log.info(String.format("Synchronise every %d tasks", wpc));
	
		ExecutionContext context = new ExecutionContext (dataflows);

		context.init();
		
		/* Dump configuration on screen */
		SystemConf.getInstance ().dump ();
		ModelConf.getInstance ().dump ();
		
		context.getDataflow(Phase.TRAIN).dump();
		context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		/*
		context.getDataflow(Phase.CHECK).exportDot("resnet.dot");
		*/
		startTime = System.nanoTime ();
		
		try {
			if (timelimit) {
				context.trainForDuration(__D, timeunit);
			} else {
				if (__N > 0){
					
					if (dataset [1] == null) {
						context.train(__N, taskunit);
					} 
					else {
						context.trainAndTest(__N, taskunit);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		if (SystemConf.getInstance().queueMeasurements()) {
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
			
			if (dataset [1] != null) {
				context.getDataflow(Phase.CHECK).getResultHandler().getMeasurementQueue().dump();
			}
		}
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
