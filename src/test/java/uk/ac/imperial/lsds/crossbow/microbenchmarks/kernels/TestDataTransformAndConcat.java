package uk.ac.imperial.lsds.crossbow.microbenchmarks.kernels;

import uk.ac.imperial.lsds.crossbow.Dataflow;
import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Dataset;
import uk.ac.imperial.lsds.crossbow.ExecutionContext;
import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.SubGraph;
import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.kernel.Concat;
import uk.ac.imperial.lsds.crossbow.kernel.ConcatGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Conv;
import uk.ac.imperial.lsds.crossbow.kernel.ConvGradient;
import uk.ac.imperial.lsds.crossbow.kernel.DataTransform;
import uk.ac.imperial.lsds.crossbow.kernel.ElementWiseOp;
import uk.ac.imperial.lsds.crossbow.kernel.GradientDescentOptimiser;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Pool;
import uk.ac.imperial.lsds.crossbow.kernel.PoolGradient;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMax;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLoss;
import uk.ac.imperial.lsds.crossbow.kernel.SoftMaxLossGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConcatConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.DataTransformConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ElementWiseOpConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LossConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SoftMaxConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.SolverConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.preprocess.DatasetUtils;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.Phase;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;
import uk.ac.imperial.lsds.crossbow.types.TrainingUnit;

public class TestDataTransformAndConcat {

	public static void main (String [] args) throws Exception {

		int batchSize = 32;

		int N = 100;
		TrainingUnit unit = TrainingUnit.TASKS;

		SystemConf.getInstance().setCPU(false).setGPU(true);

		/* Configure dataset */
		String dataDirectory = String.format("/data/crossbow/cifar-10/padded/b-%03d/", batchSize);

		Dataset dataset = new Dataset (DatasetUtils.buildPath(dataDirectory, "cifar-train.metadata", true));

		ModelConf.getInstance().setDataset(Phase.TRAIN, dataset).setSolverConf(new SolverConf ());

		/* Set dataflow */

		DataTransformConf transformconf = new DataTransformConf ();

		transformconf.setMeanImageFilename ("/data/crossbow/cifar-10/padded/cifar-train.mean");
		// transformconf.setMirror(true);
		// transformconf.setCropSize(32);
		
		ConvConf convconf = new ConvConf ();
		
		convconf.setNumberOfOutputs (16);
		convconf.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (1));
		convconf.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (1));

		PoolConf pconf = new PoolConf ().setMethod (PoolMethod.AVERAGE).setKernelSize (3).setStrideSize (2).setPaddingSize (0);
		PoolConf qconf = new PoolConf ().setMethod (PoolMethod.AVERAGE).setKernelSize (3).setStrideSize (2).setPaddingSize (0);

		ConcatConf concatconf = new ConcatConf ();

		InnerProductConf innerproductconf = new InnerProductConf ().setNumberOfOutputs (10);

		innerproductconf.setWeightInitialiser (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(1));
		innerproductconf.setBiasInitialiser   (new InitialiserConf().setType(InitialiserType.CONSTANT).setValue(0));

		SoftMaxConf softmaxconf = new SoftMaxConf ();

		LossConf lossconf = new LossConf ();

		ConcatConf lconf = new ConcatConf ();
		ConcatConf rconf = new ConcatConf ();
		
		ElementWiseOpConf eltwiseopconf = new ElementWiseOpConf ();
		
		SolverConf solverconf = ModelConf.getInstance().getSolverConf();

		DataflowNode transform = new DataflowNode (new Operator ("DataTransform",        new DataTransform          (transformconf)));
		DataflowNode conv      = new DataflowNode (new Operator ("Conv",                 new Conv                        (convconf)));
		DataflowNode P         = new DataflowNode (new Operator ("Pool (a)",             new Pool                           (pconf)));
		DataflowNode Q         = new DataflowNode (new Operator ("Pool (b)",             new Pool                           (qconf)));
		DataflowNode concat    = new DataflowNode (new Operator ("Concat",               new Concat                    (concatconf)));
		DataflowNode ip        = new DataflowNode (new Operator ("InnerProduct",         new InnerProduct        (innerproductconf)));
		DataflowNode softmax   = new DataflowNode (new Operator ("SoftMax",              new SoftMax                  (softmaxconf)));
		DataflowNode loss      = new DataflowNode (new Operator ("SoftMaxLoss",          new SoftMaxLoss                 (lossconf)));
		DataflowNode loss_     = new DataflowNode (new Operator ("SoftMaxLossGradient",  new SoftMaxLossGradient          (lossconf)).setPeer(  loss.getOperator()));
		DataflowNode ip_       = new DataflowNode (new Operator ("InnerProductGradient", new InnerProductGradient (innerproductconf)).setPeer(    ip.getOperator()));
		DataflowNode L         = new DataflowNode (new Operator ("ConcatGradient (a)",   new ConcatGradient                  (lconf)).setPeer(concat.getOperator()));
		DataflowNode R         = new DataflowNode (new Operator ("ConcatGradient (b)",   new ConcatGradient                  (rconf)).setPeer(concat.getOperator()));
		DataflowNode P_        = new DataflowNode (new Operator ("PoolGradient",         new PoolGradient                    (pconf)).setPeer(     P.getOperator()));
		DataflowNode Q_        = new DataflowNode (new Operator ("PoolGradient",         new PoolGradient                    (qconf)).setPeer(     Q.getOperator()));
		DataflowNode merge     = new DataflowNode (new Operator ("Merge",                new ElementWiseOp           (eltwiseopconf)));
		DataflowNode conv_     = new DataflowNode (new Operator ("ConvGradient",         new ConvGradient                 (convconf)).setPeer(   conv.getOperator()));
		DataflowNode optimiser = new DataflowNode (new Operator ("Optimiser",            new GradientDescentOptimiser  (solverconf)));
		
		transform.connectTo(conv);
		
		conv.connectTo(P).connectTo(concat);
		conv.connectTo(Q).connectTo(concat);
		
		concat.connectTo(ip);

		ip.connectTo(softmax).connectTo(loss).connectTo(loss_).connectTo(ip_);

		ip_.connectTo(L).connectTo(P_).connectTo(merge);
		ip_.connectTo(R).connectTo(Q_).connectTo(merge);
		
		merge.connectTo(conv_).connectTo(optimiser);

		SubGraph graph = new SubGraph (transform);

		Dataflow [] dataflows = new Dataflow [] { new Dataflow (graph).setPhase(Phase.TRAIN), null };

		ExecutionContext context = new ExecutionContext (dataflows);

		context.init();

		context.getDataflow(Phase.TRAIN).dump();

		try {
			context.train(N, unit);
		} catch (Exception e) {
			e.printStackTrace();
		}

		context.destroy();

		System.out.println("Bye.");
		System.exit(0);
	}
}
