package uk.ac.imperial.lsds.crossbow;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.kernel.*;
import uk.ac.imperial.lsds.crossbow.kernel.conf.*;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.LearningRateDecayPolicy;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class BabyResNet {

	public static final String usage = "usage: BabyResNetv1";

	private final static Logger log = LogManager.getLogger (BabyResNet.class);

    private static int nInputPlane;

	public static DataflowNode [] buildResNetShortcut (DataflowNode [] input, String prefix, boolean match, int numberofoutputs, int stride, DataflowNode [] gradient) {
		
		if (! match) {

			ConvConf convConf = new ConvConf ();

			convConf.setNumberOfOutputs (numberofoutputs);

			convConf.setKernelSize (2).setKernelHeight (  1   ).setKernelWidth (  1   );
			convConf.setStrideSize (2).setStrideHeight (stride).setStrideWidth (stride);
			
			BatchNormConf batchnormConf = new BatchNormConf ();
			batchnormConf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
			batchnormConf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

			DataflowNode conv      = new DataflowNode (new Operator (String.format("%s-Conv (shortcut)",      prefix), new Conv          (convConf)));
			DataflowNode batchnorm = new DataflowNode (new Operator (String.format("%s-BatchNorm (shortcut)", prefix), new BatchNorm (batchnormConf)));

			DataflowNode convgradient      = new DataflowNode (new Operator (String.format("%s-ConvGradient (shortcut)",      prefix), new ConvGradient           (convConf)).setPeer (     conv.getOperator()));
			DataflowNode batchnormgradient = new DataflowNode (new Operator (String.format("%s-BatchNormGradient (shortcut)", prefix), new BatchNormGradient (batchnormConf)).setPeer (batchnorm.getOperator()));
			
			/* Wire forward nodes */
			input [0] = input [0].connectTo(conv).connectTo(batchnorm);
			
			/* Create a shallow copy */
			input [1] = input [1].connectTo(conv.shallowCopy()).connectTo(batchnorm.shallowCopy());
			
			gradient [0] = gradient [0].connectTo(batchnormgradient).connectTo (convgradient);
		}

		return input;
	}

	public static DataflowNode [] buildResNetBlock (int features, int stride, DataflowNode [] input, String prefix, boolean bottleneck, DataflowNode [] gradient) {

		if (bottleneck)
			throw new IllegalStateException ();
		
		return buildResNetBasicUnit (features, stride, input, prefix, gradient);
	}

	public static DataflowNode [] buildResNetBasicUnit (int n, int stride, DataflowNode [] input, String prefix, DataflowNode [] gradient) {
		
		DataflowNode sum, sumgradientLeft, sumgradientRight;
		
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
		convConf [0].setPaddingSize (2).setPaddingHeight (  1   ).setPaddingWidth (  1   );
		convConf [0].setStrideSize  (2).setStrideHeight  (stride).setStrideWidth  (stride);

		convConf [0].setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1.0F));
		convConf [0].setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.0F));

		normConf [0] = new BatchNormConf ();
		
		normConf [0].setWeightInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1.0F));
		normConf [0].setBiasInitialiser	 (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0.0F));

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
		convConf [1].setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		convConf [1].setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);

		convConf [1].setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1.0F));
		convConf [1].setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.0F));

		normConf [1] = new BatchNormConf ();
		
		normConf [1].setWeightInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1.0F));
		normConf [1].setBiasInitialiser	 (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0.0F));

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
		
		/* Create a shallow copy of `sum` node for the test dataflow */
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
        nInputPlane = n ;

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
		
		DataflowNode datatransform, conv, batchnorm, relu, pool, innerproduct, softmax, loss;
		
		DataflowNode convgradient, batchnormgradient, relugradient, poolgradient, innerproductgradient, lossgradient;

		DataTransformConf datatransformConf;
		ConvConf                   convConf;
		BatchNormConf         batchnormConf;
		ReLUConf                   reluConf;
		PoolConf                   poolConf;
		InnerProductConf   innerproductConf;
		SoftMaxConf             softmaxConf;
		LossConf                   lossConf;
		
		datatransformConf = new DataTransformConf ();
		datatransformConf.setMeanImageFilename("/data/crossbow/cifar-10/mean.image");

		convConf = new ConvConf ();
		convConf.setNumberOfOutputs (16);

		convConf.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convConf.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convConf.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		convConf.setWeightInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1.0F));
		convConf.setBiasInitialiser	 (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0.0F));

		batchnormConf = new BatchNormConf ();
		
		batchnormConf.setWeightInitialiser(new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1.0F));
		batchnormConf.setBiasInitialiser  (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0.0F));
		
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
        nInputPlane = convConf.numberOfOutputs() ;

		/* 1 stage */
		for (int stage = 1; stage <= features.length ; ++stage) {

			for (int unit = 1; unit <= blocks; ++unit) {

                if ((unit == 1) && (stage != 1)) {
                    stride = 2;
                } else {
                    stride = 1;
                }

				prefix = String.format("stage-%d-unit-%d", stage, unit);
				boolean isBottleNeck = false;
				node = buildResNetBlock (features[stage - 1], stride, node, prefix, isBottleNeck, gradient);
			}
		}
		
		poolConf = new PoolConf ().setMethod(PoolMethod.AVERAGE);
		
		poolConf.setKernelSize  (7);
		poolConf.setStrideSize  (1);
		poolConf.setPaddingSize (0);
		
		innerproductConf = new InnerProductConf ();

		innerproductConf.setNumberOfOutputs(numberOfOutputs);

		innerproductConf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(1.0F));
		innerproductConf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue(0.0F));

		softmaxConf = new SoftMaxConf ();

		lossConf = new LossConf ();

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
				.connectTo(new DataflowNode (     softmax.getOperator()))
				.connectTo(new DataflowNode (        loss.getOperator()));
		
		/* Wire backward nodes */
		lossgradient.connectTo(innerproductgradient).connectTo(poolgradient).connectTo(gradient[0]);
		gradient[0] = lossgradient;
		
		/* Complete training dataflow */
		
		node [0] = node [0].connectTo (gradient[0]);
		
		/* Create solver operator */
		DataflowNode optimiser = new DataflowNode (new Operator ("Optimiser", new GradientDescentOptimiser (solverConf)));

		tail.connectTo(optimiser);

		graphs [0] = new SubGraph (head [0]);
		
		/* Complete testing dataflow */
		
		AccuracyConf accuracyConf = new AccuracyConf ();

		DataflowNode accuracy = new DataflowNode (new Operator ("Accuracy", new Accuracy (accuracyConf)));

		node [1].connectTo(accuracy);
		
		graphs [1] = new SubGraph (head [1]);
		
		/* Return sub-graphs */
		return graphs;
	}

	public static void main (String [] args) throws Exception {

		long startTime, dt;
		
		/* Parse command line arguments */
		int i, j;
		
		boolean __cpu = true, __gpu = false;
		
		/* GPU */
		int __number_of_gpu_model_replicas = 1;
		int __number_of_gpu_streams = 1;
		int __number_of_callback_handlers = 8;
		
		/* CPU */
		int __number_of_workers = 16;
		int __number_of_cpu_model_replicas = 16;
		int __readers_per_model = 1;
		
		int __batch_size = 128;
		int __wpc = 1000000;
		
		int __N = 1000;
		TrainingUnit __unit = TrainingUnit.TASKS;
		int __display_interval = 100;
		
		float __learning_rate = 0.01F;
		float __momentum = 0.0F;
		float __weight_decay = 0F;
		
		boolean __queue_measurements = true;
		
		long __random_seed = 123456789L;

		for (i = 0; i < args.length; ) {
			if ((j = i + 1) == args.length) {
				System.err.println(usage);
				System.exit(1);
			}
			if (args[i].equals("--cpu")) {
				__cpu = Boolean.parseBoolean(args[j]);
			} else
			if (args[i].equals("--gpu")) {
				__gpu = Boolean.parseBoolean(args[j]);
			} else
			if (args[i].equals("--wpc")) {
				__wpc = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-gpu-model-replicas")) {
				__number_of_gpu_model_replicas =  Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-gpu-streams")) {
				__number_of_gpu_streams = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--number-of-callback-handlers")) {
				__number_of_callback_handlers = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--unit")) {
				__unit = TrainingUnit.fromString(args[j]);
			} else
			if (args[i].equals("--N")) {
				__N = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--display-interval")) {
				__display_interval = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--batch-size")) {
				__batch_size = Integer.parseInt(args[j]);
			} else
			if (args[i].equals("--queue-measurements")) {
				__queue_measurements = Boolean.parseBoolean(args[j]);
			} else
			if (args[i].equals("--queue-measurements")) {
				__random_seed = Long.parseLong(args[j]);
			} else
			if (args[i].equals("--learning-rate")) {
				__learning_rate = Float.parseFloat(args[j]);
			} else
			if (args[i].equals("--momentum")) {
				__momentum = Float.parseFloat(args[j]);
			} else
			if (args[i].equals("--weight-decay")) {
				__weight_decay = Float.parseFloat(args[j]);
			} else {
				System.err.println(String.format("error: unknown flag %s %s", args[i], args[j]));
				System.exit(1);
			}
			i = j + 1;
		}
        
		startTime = System.nanoTime ();
		
		/* Create solver configuration */
		SolverConf solverConf = new SolverConf ().setLearningRateDecayPolicy (LearningRateDecayPolicy.FIXED)
			.setBaseLearningRate (__learning_rate)
			.setMomentum (__momentum)
			.setWeightDecay (__weight_decay);
		
		/* Build dataflows for both training and testing phases */
		SubGraph [] graphs = null;
		Dataset [] dataset = new Dataset [] { null, null };
					
		/* Create dataset */
		String dataDirectory = String.format("/data/crossbow/cifar-10/b-%03d/", __batch_size);
			
		dataset [0] = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));
		dataset [1] = null ; // new Dataset (DatasetUtils.buildPath(dataDirectory,  "cifar-test.metadata", true));

		int [] features = new int [] { 16 }; /* Only 1 stage */
		
		int blocks = 1;  /* Only 1 block in the stage */

		graphs = buildResNetCifar10 (blocks, features, solverConf);
		
		/* Model configuration (Part I) */
		
		ModelConf.getInstance ().setBatchSize (__batch_size).setDataset (Phase.TRAIN, dataset[0]).setDataset (Phase.CHECK, dataset[1]);
		
		int [] tasksize = ModelConf.getInstance ().getTaskSize ();
		
		log.info(String.format("%d examples/task; %d tasks/epoch; %d and %d bytes/task for training and testing respectively", 
				ModelConf.getInstance().getBatchSize(), 
				ModelConf.getInstance().numberOfTasksPerEpoch(),
				tasksize[0], 
				tasksize[1]
				));
		
		/* Model configuration (Part II) */
		
		if (! (__wpc > 0))
			throw new IllegalArgumentException();
		
		log.info(String.format("Synchronise every %d tasks", __wpc));
		
		ModelConf.getInstance().setWpc(__wpc).setSlack(0).setTestInterval(ModelConf.getInstance().numberOfTasksPerEpoch());
		
		/* Model configuration (Part III) */
		
		ModelConf.getInstance().setSolverConf(solverConf);
		
		/* Set dataflow */

		Dataflow [] dataflows = new Dataflow [2];

        /* For ResNet */
		dataflows [0] = new Dataflow (graphs [0]).setPhase(Phase.TRAIN);
		dataflows [1] = null; // new Dataflow (graphs [1]).setPhase(Phase.CHECK);
		
		SystemConf.getInstance()
			.setCPU(__cpu)
			.setGPU(__gpu)
			.setNumberOfWorkerThreads (__number_of_workers)
			.setNumberOfCPUModelReplicas (__number_of_cpu_model_replicas)
			.setNumberOfReadersPerModel(__readers_per_model)
			.setNumberOfGPUModelReplicas(__number_of_gpu_model_replicas)
			.setNumberOfGPUStreams(__number_of_gpu_streams)
			.setNumberOfGPUCallbackHandlers(__number_of_callback_handlers)
			.setDisplayInterval(__display_interval)
			.displayAccumulatedLossValue(true)
			.setRandomSeed(__random_seed)
			.queueMeasurements(__queue_measurements);
		
		SystemConf.getInstance().allowMemoryReuse(true);
		
		ExecutionContext context = new ExecutionContext (dataflows);

		context.init();
		
		context.getDataflow(Phase.TRAIN).dump();
		// context.getDataflow(Phase.TRAIN).dumpMemoryRequirements();
		// context.getDataflow(Phase.CHECK).dump();
		
		context.getModel().dump();
		
//		try {
//			context.train(__N, __unit);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
		
		dt = System.nanoTime() - startTime;
		System.out.println(String.format("dt = %10.2f secs", (double)(dt) / 1000000000));
		
		if (SystemConf.getInstance().queueMeasurements())
			context.getDataflow(Phase.TRAIN).getResultHandler().getMeasurementQueue().dump();
		
		context.destroy();
		
		System.out.println("Bye.");
		System.exit(0);
	}
}
