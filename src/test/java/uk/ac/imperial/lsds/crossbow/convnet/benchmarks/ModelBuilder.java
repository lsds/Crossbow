package uk.ac.imperial.lsds.crossbow.convnet.benchmarks;

import uk.ac.imperial.lsds.crossbow.DataflowNode;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.kernel.Concat;
import uk.ac.imperial.lsds.crossbow.kernel.ConcatGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Conv;
import uk.ac.imperial.lsds.crossbow.kernel.ConvGradient;
import uk.ac.imperial.lsds.crossbow.kernel.ElementWiseOp;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProduct;
import uk.ac.imperial.lsds.crossbow.kernel.InnerProductGradient;
import uk.ac.imperial.lsds.crossbow.kernel.Pool;
import uk.ac.imperial.lsds.crossbow.kernel.PoolGradient;
import uk.ac.imperial.lsds.crossbow.kernel.ReLU;
import uk.ac.imperial.lsds.crossbow.kernel.ReLUGradient;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConcatConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ConvConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ElementWiseOpConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.InnerProductConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.PoolConf;
import uk.ac.imperial.lsds.crossbow.kernel.conf.ReLUConf;
import uk.ac.imperial.lsds.crossbow.model.InitialiserConf;
import uk.ac.imperial.lsds.crossbow.types.InitialiserType;
import uk.ac.imperial.lsds.crossbow.types.PoolMethod;

public class ModelBuilder {
	
	public static DataflowNode buildAlexNet (boolean gradient) {
		
		/* Input dimension: 128 x 3 x 224 x 224 */
		
		/* Stage 1 */
		
		ConvConf convconf1 = new ConvConf ();
		
		convconf1.setNumberOfOutputs (64);
		
		convconf1.setKernelSize  (2).setKernelHeight  (11).setKernelWidth  (11);
		convconf1.setStrideSize  (2).setStrideHeight  ( 4).setStrideWidth  ( 4);
		convconf1.setPaddingSize (2).setPaddingHeight ( 2).setPaddingWidth ( 2);
		
		convconf1.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf1 = new ReLUConf ();
		
		PoolConf poolconf1 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf1.setKernelSize (3).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv1 = new DataflowNode (new Operator ("Conv-Stage-1", new Conv (convconf1)));
		DataflowNode relu1 = new DataflowNode (new Operator ("ReLU-Stage-1", new ReLU (reluconf1)));
		DataflowNode pool1 = new DataflowNode (new Operator ("Pool-Stage-1", new Pool (poolconf1)));
		
		/* Stage 2 */
		
		ConvConf convconf2 = new ConvConf ();
		
		convconf2.setNumberOfOutputs (192);
		
		convconf2.setKernelSize  (2).setKernelHeight  (5).setKernelWidth  (5);
		convconf2.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf2.setPaddingSize (2).setPaddingHeight (2).setPaddingWidth (2);
		
		convconf2.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf2 = new ReLUConf ();
		
		PoolConf poolconf2 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf2.setKernelSize (3).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv2 = new DataflowNode (new Operator ("Conv-Stage-2", new Conv (convconf2)));
		DataflowNode relu2 = new DataflowNode (new Operator ("ReLU-Stage-2", new ReLU (reluconf2)));
		DataflowNode pool2 = new DataflowNode (new Operator ("Pool-Stage-2", new Pool (poolconf2)));
		
		/* Stage 3 */
		
		ConvConf convconf3a = new ConvConf ();
		
		convconf3a.setNumberOfOutputs (384);
		
		convconf3a.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3a.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3a.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3a.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3a.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3a.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3a = new ReLUConf ();
		
		DataflowNode conv3a = new DataflowNode (new Operator ("Conv-Stage-3 (a)", new Conv (convconf3a)));
		DataflowNode relu3a = new DataflowNode (new Operator ("ReLU-Stage-3 (a)", new ReLU (reluconf3a)));
		
		ConvConf convconf3b = new ConvConf ();
		
		convconf3b.setNumberOfOutputs (256);
		
		convconf3b.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3b.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3b.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3b.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3b.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3b.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3b = new ReLUConf ();
		
		DataflowNode conv3b = new DataflowNode (new Operator ("Conv-Stage-3 (b)", new Conv (convconf3b)));
		DataflowNode relu3b = new DataflowNode (new Operator ("ReLU-Stage-3 (b)", new ReLU (reluconf3b)));
		
		ConvConf convconf3c = new ConvConf ();
		
		convconf3c.setNumberOfOutputs (256);
		
		convconf3c.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3c.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3c.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3c.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3c.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3c.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3c = new ReLUConf ();
		
		PoolConf poolconf3c = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf3c.setKernelSize (3).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv3c = new DataflowNode (new Operator ("Conv-Stage-3 (c)", new Conv (convconf3c)));
		DataflowNode relu3c = new DataflowNode (new Operator ("ReLU-Stage-3 (c)", new ReLU (reluconf3c)));
		DataflowNode pool3c = new DataflowNode (new Operator ("Pool-Stage-3 (c)", new Pool (poolconf3c)));
		
		/* Stage 4 */
		
		InnerProductConf ipconf4 = new InnerProductConf ().setNumberOfOutputs (4096);
		
		ReLUConf reluconf4 = new ReLUConf ();
		
		DataflowNode   ip4 = new DataflowNode (new Operator (  "Ip-Stage-4", new InnerProduct   (ipconf4)));
		DataflowNode relu4 = new DataflowNode (new Operator ("ReLU-Stage-4", new ReLU         (reluconf4)));
		
		/* Stage 5 */
		
		InnerProductConf ipconf5 = new InnerProductConf ().setNumberOfOutputs (4096);
		
		ReLUConf reluconf5 = new ReLUConf ();
		
		DataflowNode   ip5 = new DataflowNode (new Operator (  "Ip-Stage-5", new InnerProduct   (ipconf5)));
		DataflowNode relu5 = new DataflowNode (new Operator ("ReLU-Stage-5", new ReLU         (reluconf5)));
		
		/* Stage 6 */
		
		InnerProductConf ipconf6 = new InnerProductConf ().setNumberOfOutputs (1000);
		
		DataflowNode ip6 = new DataflowNode (new Operator ("Ip-Stage-6", new InnerProduct (ipconf6)));
		
		/* Put dataflow together */
		
		DataflowNode head, tail;
		head = conv1;
		tail = conv1.connectTo(relu1).connectTo(pool1).connectTo(conv2).connectTo(relu2).connectTo(pool2).connectTo(conv3a).connectTo(relu3a).connectTo(conv3b).connectTo(relu3b).connectTo(conv3c).connectTo(relu3c).connectTo(pool3c).connectTo(ip4).connectTo(relu4).connectTo(ip5).connectTo(relu5).connectTo(ip6);
		
		if (gradient) {
			
			DataflowNode _conv1  = new DataflowNode (new Operator ("ConvGradient-Stage-1",     new         ConvGradient (convconf1 )).setPeer ( conv1.getOperator()));
			DataflowNode _relu1  = new DataflowNode (new Operator ("ReLUGradient-Stage-1",     new         ReLUGradient (reluconf1 )).setPeer ( relu1.getOperator()));
			DataflowNode _pool1  = new DataflowNode (new Operator ("PoolGradient-Stage-1",     new         PoolGradient (poolconf1 )).setPeer ( pool1.getOperator()));
			DataflowNode _conv2  = new DataflowNode (new Operator ("ConvGradient-Stage-2",     new         ConvGradient (convconf2 )).setPeer ( conv2.getOperator()));
			DataflowNode _relu2  = new DataflowNode (new Operator ("ReLUGradient-Stage-2",     new         ReLUGradient (reluconf2 )).setPeer ( relu2.getOperator()));
			DataflowNode _pool2  = new DataflowNode (new Operator ("PoolGradient-Stage-2",     new         PoolGradient (poolconf2 )).setPeer ( pool2.getOperator()));
			DataflowNode _conv3a = new DataflowNode (new Operator ("ConvGradient-Stage-3 (a)", new         ConvGradient (convconf3a)).setPeer (conv3a.getOperator()));
			DataflowNode _relu3a = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (a)", new         ReLUGradient (reluconf3a)).setPeer (relu3a.getOperator()));
			DataflowNode _conv3b = new DataflowNode (new Operator ("ConvGradient-Stage-3 (b)", new         ConvGradient (convconf3b)).setPeer (conv3b.getOperator()));
			DataflowNode _relu3b = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (b)", new         ReLUGradient (reluconf3b)).setPeer (relu3b.getOperator()));
			DataflowNode _conv3c = new DataflowNode (new Operator ("ConvGradient-Stage-3 (c)", new         ConvGradient (convconf3c)).setPeer (conv3c.getOperator()));
			DataflowNode _relu3c = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (c)", new         ReLUGradient (reluconf3c)).setPeer (relu3c.getOperator()));
			DataflowNode _pool3c = new DataflowNode (new Operator ("PoolGradient-Stage-3 (c)", new         PoolGradient (poolconf3c)).setPeer (pool3c.getOperator()));
			DataflowNode   _ip4  = new DataflowNode (new Operator (  "IpGradient-Stage-4",     new InnerProductGradient (  ipconf4 )).setPeer (   ip4.getOperator()));
			DataflowNode _relu4  = new DataflowNode (new Operator ("ReLUGradient-Stage-4",     new         ReLUGradient (reluconf4 )).setPeer ( relu4.getOperator()));
			DataflowNode   _ip5  = new DataflowNode (new Operator (  "IpGradient-Stage-5",     new InnerProductGradient (  ipconf5 )).setPeer (   ip5.getOperator()));
			DataflowNode _relu5  = new DataflowNode (new Operator ("ReLUGradient-Stage-5",     new         ReLUGradient (reluconf5 )).setPeer ( relu5.getOperator()));
			DataflowNode   _ip6  = new DataflowNode (new Operator (  "IpGradient-Stage-6",     new InnerProductGradient (  ipconf6 )).setPeer (   ip6.getOperator()));
			
			tail.connectTo(_ip6).connectTo(_relu5).connectTo(_ip5).connectTo(_relu4).connectTo(_ip4).connectTo(_pool3c).connectTo(_relu3c).connectTo(_conv3c).connectTo(_relu3b).connectTo(_conv3b).connectTo(_relu3a).connectTo(_conv3a).connectTo(_pool2).connectTo(_relu2).connectTo(_conv2).connectTo(_pool1).connectTo(_relu1).connectTo(_conv1);
		}
		
		return head;
	}
	
	public static DataflowNode buildOverfeat (boolean gradient) {
		
		/* Input dimension: 128 x 3 x 231 x 231 */
		
		/* Stage 1 */
				
		ConvConf convconf1 = new ConvConf ();
		
		convconf1.setNumberOfOutputs (96);
		
		convconf1.setKernelSize  (2).setKernelHeight  (11).setKernelWidth  (11);
		convconf1.setStrideSize  (2).setStrideHeight  ( 4).setStrideWidth  ( 4);
		convconf1.setPaddingSize (2).setPaddingHeight ( 0).setPaddingWidth ( 0);
		
		convconf1.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf1 = new ReLUConf ();
		
		PoolConf poolconf1 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf1.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv1 = new DataflowNode (new Operator ("Conv-Stage-1", new Conv (convconf1)));
		DataflowNode relu1 = new DataflowNode (new Operator ("ReLU-Stage-1", new ReLU (reluconf1)));
		DataflowNode pool1 = new DataflowNode (new Operator ("Pool-Stage-1", new Pool (poolconf1)));
		
		/* Stage 2 */
				
		ConvConf convconf2 = new ConvConf ();
		
		convconf2.setNumberOfOutputs (256);
		
		convconf2.setKernelSize  (2).setKernelHeight  (5).setKernelWidth  (5);
		convconf2.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf2.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf2.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf2 = new ReLUConf ();
		
		PoolConf poolconf2 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf2.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv2 = new DataflowNode (new Operator ("Conv-Stage-2", new Conv (convconf2)));
		DataflowNode relu2 = new DataflowNode (new Operator ("ReLU-Stage-2", new ReLU (reluconf2)));
		DataflowNode pool2 = new DataflowNode (new Operator ("Pool-Stage-2", new Pool (poolconf2)));
		
		/* Stage 3 */
		
		ConvConf convconf3a = new ConvConf ();
		
		convconf3a.setNumberOfOutputs (512);
		
		convconf3a.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3a.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3a.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3a.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3a.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3a.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3a = new ReLUConf ();
		
		DataflowNode conv3a = new DataflowNode (new Operator ("Conv-Stage-3 (a)", new Conv (convconf3a)));
		DataflowNode relu3a = new DataflowNode (new Operator ("ReLU-Stage-3 (a)", new ReLU (reluconf3a)));
		
		ConvConf convconf3b = new ConvConf ();
		
		convconf3b.setNumberOfOutputs (1024);
		
		convconf3b.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3b.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3b.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3b.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3b.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3b.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3b = new ReLUConf ();
		
		DataflowNode conv3b = new DataflowNode (new Operator ("Conv-Stage-3 (b)", new Conv (convconf3b)));
		DataflowNode relu3b = new DataflowNode (new Operator ("ReLU-Stage-3 (b)", new ReLU (reluconf3b)));
		
		ConvConf convconf3c = new ConvConf ();
		
		convconf3c.setNumberOfOutputs (1024);
		
		convconf3c.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3c.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3c.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3c.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3c.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3c.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3c = new ReLUConf ();
		
		PoolConf poolconf3c = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf3c.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv3c = new DataflowNode (new Operator ("Conv-Stage-3 (c)", new Conv (convconf3c)));
		DataflowNode relu3c = new DataflowNode (new Operator ("ReLU-Stage-3 (c)", new ReLU (reluconf3c)));
		DataflowNode pool3c = new DataflowNode (new Operator ("Pool-Stage-3 (c)", new Pool (poolconf3c)));
		
		/* Stage 4 */
		
		InnerProductConf ipconf4 = new InnerProductConf ().setNumberOfOutputs (3072);
		
		DataflowNode ip4 = new DataflowNode (new Operator ("Ip-Stage-4", new InnerProduct (ipconf4)));
		
		/* Stage 5 */
		
		InnerProductConf ipconf5 = new InnerProductConf ().setNumberOfOutputs (4096);
		
		DataflowNode ip5 = new DataflowNode (new Operator ("Ip-Stage-5", new InnerProduct (ipconf5)));
		
		/* Stage 6 */
		
		InnerProductConf ipconf6 = new InnerProductConf ().setNumberOfOutputs (1000);
		
		DataflowNode ip6 = new DataflowNode (new Operator ("Ip-Stage-6", new InnerProduct (ipconf6)));
		
		/* Put dataflow together */
		
		DataflowNode head, tail;
		head = conv1;
		tail = conv1.connectTo(relu1).connectTo(pool1).connectTo(conv2).connectTo(relu2).connectTo(pool2).connectTo(conv3a).connectTo(relu3a).connectTo(conv3b).connectTo(relu3b).connectTo(conv3c).connectTo(relu3c).connectTo(pool3c).connectTo(ip4).connectTo(ip5).connectTo(ip6);
		
		if (gradient) {
			
			DataflowNode _conv1  = new DataflowNode (new Operator ("ConvGradient-Stage-1",     new         ConvGradient (convconf1 )).setPeer ( conv1.getOperator()));
			DataflowNode _relu1  = new DataflowNode (new Operator ("ReLUGradient-Stage-1",     new         ReLUGradient (reluconf1 )).setPeer ( relu1.getOperator()));
			DataflowNode _pool1  = new DataflowNode (new Operator ("PoolGradient-Stage-1",     new         PoolGradient (poolconf1 )).setPeer ( pool1.getOperator()));
			DataflowNode _conv2  = new DataflowNode (new Operator ("ConvGradient-Stage-2",     new         ConvGradient (convconf2 )).setPeer ( conv2.getOperator()));
			DataflowNode _relu2  = new DataflowNode (new Operator ("ReLUGradient-Stage-2",     new         ReLUGradient (reluconf2 )).setPeer ( relu2.getOperator()));
			DataflowNode _pool2  = new DataflowNode (new Operator ("PoolGradient-Stage-2",     new         PoolGradient (poolconf2 )).setPeer ( pool2.getOperator()));
			DataflowNode _conv3a = new DataflowNode (new Operator ("ConvGradient-Stage-3 (a)", new         ConvGradient (convconf3a)).setPeer (conv3a.getOperator()));
			DataflowNode _relu3a = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (a)", new         ReLUGradient (reluconf3a)).setPeer (relu3a.getOperator()));
			DataflowNode _conv3b = new DataflowNode (new Operator ("ConvGradient-Stage-3 (b)", new         ConvGradient (convconf3b)).setPeer (conv3b.getOperator()));
			DataflowNode _relu3b = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (b)", new         ReLUGradient (reluconf3b)).setPeer (relu3b.getOperator()));
			DataflowNode _conv3c = new DataflowNode (new Operator ("ConvGradient-Stage-3 (c)", new         ConvGradient (convconf3c)).setPeer (conv3c.getOperator()));
			DataflowNode _relu3c = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (c)", new         ReLUGradient (reluconf3c)).setPeer (relu3c.getOperator()));
			DataflowNode _pool3c = new DataflowNode (new Operator ("PoolGradient-Stage-3 (c)", new         PoolGradient (poolconf3c)).setPeer (pool3c.getOperator()));
			DataflowNode   _ip4  = new DataflowNode (new Operator (  "IpGradient-Stage-4",     new InnerProductGradient (  ipconf4 )).setPeer (   ip4.getOperator()));
			DataflowNode   _ip5  = new DataflowNode (new Operator (  "IpGradient-Stage-5",     new InnerProductGradient (  ipconf5 )).setPeer (   ip5.getOperator()));
			DataflowNode   _ip6  = new DataflowNode (new Operator (  "IpGradient-Stage-6",     new InnerProductGradient (  ipconf6 )).setPeer (   ip6.getOperator()));
			
			tail.connectTo(_ip6).connectTo(_ip5).connectTo(_ip4).connectTo(_pool3c).connectTo(_relu3c).connectTo(_conv3c).connectTo(_relu3b).connectTo(_conv3b).connectTo(_relu3a).connectTo(_conv3a).connectTo(_pool2).connectTo(_relu2).connectTo(_conv2).connectTo(_pool1).connectTo(_relu1).connectTo(_conv1);
		}
		
		return head;
	}
	
	/*
	 * VGG (A): 64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M ...
	 * VGG (B): 64, 64, M, 128, 128, M, 256, 256, M, 512, 512, M, 512, 512, M ...
	 * VGG (D): 64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M ...
	 * VGG (E): 64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512, M ...
	 */
	public static DataflowNode buildOxfordNet (boolean gradient) { /* A.k.a. VGG (A) */
		
		/* Input is 64 x 3 x 224 x 224 */
		
		/* Stage 1 */
				
		ConvConf convconf1 = new ConvConf ();
		
		convconf1.setNumberOfOutputs (64);
		
		convconf1.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf1.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf1.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf1.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf1.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf1 = new ReLUConf ();
		
		PoolConf poolconf1 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf1.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv1 = new DataflowNode (new Operator ("Conv-Stage-2", new Conv (convconf1)));
		DataflowNode relu1 = new DataflowNode (new Operator ("ReLU-Stage-2", new ReLU (reluconf1)));
		DataflowNode pool1 = new DataflowNode (new Operator ("Pool-Stage-2", new Pool (poolconf1)));
		
		/* Stage 2 */

		ConvConf convconf2 = new ConvConf ();
		
		convconf2.setNumberOfOutputs (128);
		
		convconf2.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf2.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf2.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf2.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf2.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf2 = new ReLUConf ();
		
		PoolConf poolconf2 = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf2.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv2 = new DataflowNode (new Operator ("Conv-Stage-2", new Conv (convconf2)));
		DataflowNode relu2 = new DataflowNode (new Operator ("ReLU-Stage-2", new ReLU (reluconf2)));
		DataflowNode pool2 = new DataflowNode (new Operator ("Pool-Stage-2", new Pool (poolconf2)));
		
		/* Stage 3 */
		
		ConvConf convconf3a = new ConvConf ();
		
		convconf3a.setNumberOfOutputs (256);
		
		convconf3a.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3a.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3a.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3a.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3a.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3a.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3a = new ReLUConf ();
		
		ConvConf convconf3b = new ConvConf ();
		
		convconf3b.setNumberOfOutputs (256);
		
		convconf3b.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3b.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3b.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3b.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf3b.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf3b.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf3b = new ReLUConf ();
		
		PoolConf poolconf3b = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf3b.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv3a = new DataflowNode (new Operator ("Conv-Stage-4 (a)", new Conv (convconf3a)));
		DataflowNode relu3a = new DataflowNode (new Operator ("ReLU-Stage-4 (a)", new ReLU (reluconf3a)));
		
		DataflowNode conv3b = new DataflowNode (new Operator ("Conv-Stage-4 (b)", new Conv (convconf3b)));
		DataflowNode relu3b = new DataflowNode (new Operator ("ReLU-Stage-4 (b)", new ReLU (reluconf3b)));
		
		DataflowNode pool3b = new DataflowNode (new Operator ("Pool-Stage-4 (b)", new Pool (poolconf3b)));
		
		/* Stage 4 */
			
		ConvConf convconf4a = new ConvConf ();
		
		convconf4a.setNumberOfOutputs (512);
		
		convconf4a.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf4a.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf4a.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf4a.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf4a.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf4a.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf4a = new ReLUConf ();
		
		ConvConf convconf4b = new ConvConf ();
		
		convconf4b.setNumberOfOutputs (512);
		
		convconf4b.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf4b.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf4b.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf4b.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf4b.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf4b.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf4b = new ReLUConf ();
		
		PoolConf poolconf4b = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf4b.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv4a = new DataflowNode (new Operator ("Conv-Stage-4 (a)", new Conv (convconf4a)));
		DataflowNode relu4a = new DataflowNode (new Operator ("ReLU-Stage-4 (a)", new ReLU (reluconf4a)));
		
		DataflowNode conv4b = new DataflowNode (new Operator ("Conv-Stage-4 (b)", new Conv (convconf4b)));
		DataflowNode relu4b = new DataflowNode (new Operator ("ReLU-Stage-4 (b)", new ReLU (reluconf4b)));
		DataflowNode pool4b = new DataflowNode (new Operator ("Pool-Stage-4 (b)", new Pool (poolconf4b)));
		
		/* Stage 5 */
		
		ConvConf convconf5a = new ConvConf ();
		
		convconf5a.setNumberOfOutputs (512);
		
		convconf5a.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf5a.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf5a.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf5a.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf5a.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf5a.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf5a = new ReLUConf ();
		
		ConvConf convconf5b = new ConvConf ();
		
		convconf5b.setNumberOfOutputs (512);
		
		convconf5b.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf5b.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf5b.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf5b.setWeightsLearningRateMultiplier (1).setBiasLearningRateMultiplier (2);
		
		convconf5b.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.XAVIER).setStd     (0.1F));
		convconf5b.setBiasInitialiser   (new InitialiserConf ().setType (InitialiserType.CONSTANT).setValue (0.2F));
		
		ReLUConf reluconf5b = new ReLUConf ();
		
		PoolConf poolconf5b = new PoolConf ().setMethod (PoolMethod.MAX);
		
		poolconf5b.setKernelSize (2).setStrideSize (2).setPaddingSize (0);
		
		DataflowNode conv5a = new DataflowNode (new Operator ("Conv-Stage-5 (a)", new Conv (convconf5a)));
		DataflowNode relu5a = new DataflowNode (new Operator ("ReLU-Stage-5 (a)", new ReLU (reluconf5a)));
		
		DataflowNode conv5b = new DataflowNode (new Operator ("Conv-Stage-5 (b)", new Conv (convconf5b)));
		DataflowNode relu5b = new DataflowNode (new Operator ("ReLU-Stage-5 (b)", new ReLU (reluconf5b)));
		DataflowNode pool5b = new DataflowNode (new Operator ("Pool-Stage-5 (b)", new Pool (poolconf5b)));
		
		/* Stage 6 */
		
		InnerProductConf ipconf6 = new InnerProductConf ().setNumberOfOutputs (4096);
		
		DataflowNode ip6 = new DataflowNode (new Operator ("Ip-Stage-6", new InnerProduct (ipconf6)));
		
		/* Stage 7 */
		
		InnerProductConf ipconf7 = new InnerProductConf ().setNumberOfOutputs (3072);
		
		DataflowNode ip7 = new DataflowNode (new Operator ("Ip-Stage-7", new InnerProduct (ipconf7)));
		
		/* Stage 8 */
		
		InnerProductConf ipconf8 = new InnerProductConf ().setNumberOfOutputs (3072);
		
		DataflowNode ip8 = new DataflowNode (new Operator ("Ip-Stage-8", new InnerProduct (ipconf8)));
		
		/* Put dataflow together */
		
		DataflowNode head, tail;
		head = conv1;
		tail = conv1.connectTo(relu1).connectTo(pool1).connectTo(conv2).connectTo(relu2).connectTo(pool2).connectTo(conv3a).connectTo(relu3a).connectTo(conv3b).connectTo(relu3b).connectTo(pool3b).connectTo(conv4a).connectTo(relu4a).connectTo(conv4b).connectTo(relu4b).connectTo(pool4b).connectTo(conv5a).connectTo(relu5a).connectTo(conv5b).connectTo(relu5b).connectTo(pool5b).connectTo(ip6).connectTo(ip7).connectTo(ip8);
		
		if (gradient) {
			
			DataflowNode _conv1  = new DataflowNode (new Operator ("ConvGradient-Stage-1",     new         ConvGradient (convconf1 )).setPeer ( conv1.getOperator()));
			DataflowNode _relu1  = new DataflowNode (new Operator ("ReLUGradient-Stage-1",     new         ReLUGradient (reluconf1 )).setPeer ( relu1.getOperator()));
			DataflowNode _pool1  = new DataflowNode (new Operator ("PoolGradient-Stage-1",     new         PoolGradient (poolconf1 )).setPeer ( pool1.getOperator()));
			DataflowNode _conv2  = new DataflowNode (new Operator ("ConvGradient-Stage-2",     new         ConvGradient (convconf2 )).setPeer ( conv2.getOperator()));
			DataflowNode _relu2  = new DataflowNode (new Operator ("ReLUGradient-Stage-2",     new         ReLUGradient (reluconf2 )).setPeer ( relu2.getOperator()));
			DataflowNode _pool2  = new DataflowNode (new Operator ("PoolGradient-Stage-2",     new         PoolGradient (poolconf2 )).setPeer ( pool2.getOperator()));
			DataflowNode _conv3a = new DataflowNode (new Operator ("ConvGradient-Stage-3 (a)", new         ConvGradient (convconf3a)).setPeer (conv3a.getOperator()));
			DataflowNode _relu3a = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (a)", new         ReLUGradient (reluconf3a)).setPeer (relu3a.getOperator()));
			DataflowNode _conv3b = new DataflowNode (new Operator ("ConvGradient-Stage-3 (b)", new         ConvGradient (convconf3b)).setPeer (conv3b.getOperator()));
			DataflowNode _relu3b = new DataflowNode (new Operator ("ReLUGradient-Stage-3 (b)", new         ReLUGradient (reluconf3b)).setPeer (relu3b.getOperator()));
			DataflowNode _pool3b = new DataflowNode (new Operator ("PoolGradient-Stage-3 (b)", new         PoolGradient (poolconf3b)).setPeer (pool3b.getOperator()));
			DataflowNode _conv4a = new DataflowNode (new Operator ("ConvGradient-Stage-4 (a)", new         ConvGradient (convconf4a)).setPeer (conv4a.getOperator()));
			DataflowNode _relu4a = new DataflowNode (new Operator ("ReLUGradient-Stage-4 (a)", new         ReLUGradient (reluconf4a)).setPeer (relu4a.getOperator()));
			DataflowNode _conv4b = new DataflowNode (new Operator ("ConvGradient-Stage-4 (b)", new         ConvGradient (convconf4b)).setPeer (conv4b.getOperator()));
			DataflowNode _relu4b = new DataflowNode (new Operator ("ReLUGradient-Stage-4 (b)", new         ReLUGradient (reluconf4b)).setPeer (relu4b.getOperator()));
			DataflowNode _pool4b = new DataflowNode (new Operator ("PoolGradient-Stage-4 (b)", new         PoolGradient (poolconf4b)).setPeer (pool4b.getOperator()));
			DataflowNode _conv5a = new DataflowNode (new Operator ("ConvGradient-Stage-5 (a)", new         ConvGradient (convconf5a)).setPeer (conv5a.getOperator()));
			DataflowNode _relu5a = new DataflowNode (new Operator ("ReLUGradient-Stage-5 (a)", new         ReLUGradient (reluconf5a)).setPeer (relu5a.getOperator()));
			DataflowNode _conv5b = new DataflowNode (new Operator ("ConvGradient-Stage-5 (b)", new         ConvGradient (convconf5b)).setPeer (conv5b.getOperator()));
			DataflowNode _relu5b = new DataflowNode (new Operator ("ReLUGradient-Stage-5 (b)", new         ReLUGradient (reluconf5b)).setPeer (relu5b.getOperator()));
			DataflowNode _pool5b = new DataflowNode (new Operator ("PoolGradient-Stage-5 (b)", new         PoolGradient (poolconf5b)).setPeer (pool5b.getOperator()));
			DataflowNode   _ip6  = new DataflowNode (new Operator (  "IpGradient-Stage-6",     new InnerProductGradient (  ipconf6 )).setPeer (   ip6.getOperator()));
			DataflowNode   _ip7  = new DataflowNode (new Operator (  "IpGradient-Stage-7",     new InnerProductGradient (  ipconf7 )).setPeer (   ip7.getOperator()));
			DataflowNode   _ip8  = new DataflowNode (new Operator (  "IpGradient-Stage-8",     new InnerProductGradient (  ipconf8 )).setPeer (   ip8.getOperator()));
			
			tail.connectTo(_ip8).connectTo(_ip7).connectTo(_ip6).connectTo(_pool5b).connectTo(_relu5b).connectTo(_conv5b).connectTo(_relu5a).connectTo(_conv5a).connectTo(_pool4b).connectTo(_relu4b).connectTo(_conv4b).connectTo(_relu4a).connectTo(_conv4a).connectTo(_pool3b).connectTo(_relu3b).connectTo(_conv3b).connectTo(_relu3a).connectTo(_conv3a).connectTo(_pool2).connectTo(_relu2).connectTo(_conv2).connectTo(_pool1).connectTo(_relu1).connectTo(_conv1);
		}
		
		return head;
	}
	
	private static DataflowNode [] buildInceptionUnit (DataflowNode [] input, String prefix, DataflowNode [] gradient, int [][] config) {
		
		ConvConf convconf_t1_1x1 = new ConvConf ();
		
		convconf_t1_1x1.setNumberOfOutputs (config [0][0]);
		
		convconf_t1_1x1.setKernelSize  (2).setKernelHeight  (1).setKernelWidth  (1);
		convconf_t1_1x1.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t1_1x1.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf_t1_1x1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t1_1x1.setBias (false);
		
		ReLUConf reluconf_t1_1x1 = new ReLUConf ();
		
		ConvConf convconf_t2_1x1 = new ConvConf ();
		
		convconf_t2_1x1.setNumberOfOutputs (config [1][0]);
		
		convconf_t2_1x1.setKernelSize  (2).setKernelHeight  (1).setKernelWidth  (1);
		convconf_t2_1x1.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t2_1x1.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf_t2_1x1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t2_1x1.setBias (false);
		
		ReLUConf reluconf_t2_1x1 = new ReLUConf ();
		
		ConvConf convconf_t2_3x3 = new ConvConf ();
		
		convconf_t2_3x3.setNumberOfOutputs (config [1][1]);
		
		convconf_t2_3x3.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf_t2_3x3.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t2_3x3.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf_t2_3x3.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t2_3x3.setBias (false);
		
		ReLUConf reluconf_t2_3x3 = new ReLUConf ();
		
		ConvConf convconf_t3_1x1 = new ConvConf ();
		
		convconf_t3_1x1.setNumberOfOutputs (config [2][0]);
		
		convconf_t3_1x1.setKernelSize  (2).setKernelHeight  (1).setKernelWidth  (1);
		convconf_t3_1x1.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t3_1x1.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf_t3_1x1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t3_1x1.setBias (false);
		
		ReLUConf reluconf_t3_1x1 = new ReLUConf ();
		
		ConvConf convconf_t3_5x5 = new ConvConf ();
		
		convconf_t3_5x5.setNumberOfOutputs (config[2][1]);
		
		convconf_t3_5x5.setKernelSize  (2).setKernelHeight  (5).setKernelWidth  (5);
		convconf_t3_5x5.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t3_5x5.setPaddingSize (2).setPaddingHeight (2).setPaddingWidth (2);
		
		convconf_t3_5x5.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t3_5x5.setBias (false);
		
		ReLUConf reluconf_t3_5x5 = new ReLUConf ();
		
		PoolConf poolconf_t4_1x1 = new PoolConf ().setMethod(PoolMethod.MAX).setKernelSize(config [3][0]).setStrideSize(1).setPaddingSize(1);
		
		ConvConf convconf_t4_1x1 = new ConvConf ();
		
		convconf_t4_1x1.setNumberOfOutputs (config[3][1]);
		
		convconf_t4_1x1.setKernelSize  (2).setKernelHeight  (1).setKernelWidth  (1);
		convconf_t4_1x1.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf_t4_1x1.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf_t4_1x1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf_t4_1x1.setBias (false);
		
		ReLUConf reluconf_t4_1x1 = new ReLUConf ();
		
		DataflowNode conv_t1_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-1-1x1", prefix), new Conv (convconf_t1_1x1)));
		DataflowNode relu_t1_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-1-1x1", prefix), new ReLU (reluconf_t1_1x1)));
		
		DataflowNode conv_t2_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-2-1x1", prefix), new Conv (convconf_t2_1x1)));
		DataflowNode relu_t2_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-2-1x1", prefix), new ReLU (reluconf_t2_1x1)));
		
		DataflowNode conv_t2_3x3 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-2-3x3", prefix), new Conv (convconf_t2_3x3)));
		DataflowNode relu_t2_3x3 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-2-3x3", prefix), new ReLU (reluconf_t2_3x3)));
		
		DataflowNode conv_t3_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-3-1x1", prefix), new Conv (convconf_t3_1x1)));
		DataflowNode relu_t3_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-3-1x1", prefix), new ReLU (reluconf_t3_1x1)));
		
		DataflowNode conv_t3_5x5 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-3-5x3", prefix), new Conv (convconf_t3_5x5)));
		DataflowNode relu_t3_5x5 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-3-5x3", prefix), new ReLU (reluconf_t3_5x5)));
		
		DataflowNode pool_t4_1x1 = new DataflowNode (new Operator (String.format("%s-Pool-Tower-4-1x1", prefix), new Pool (poolconf_t4_1x1)));
		
		DataflowNode conv_t4_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-4-1x1", prefix), new Conv (convconf_t4_1x1)));
		DataflowNode relu_t4_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-4-1x1", prefix), new ReLU (reluconf_t4_1x1)));
		
		DataflowNode concat  = new DataflowNode (new Operator (String.format("%s-Concat", prefix), new Concat (new ConcatConf ())));
		
		input[0].connectTo(conv_t1_1x1).connectTo(relu_t1_1x1).connectTo(concat);
		input[0].connectTo(conv_t2_1x1).connectTo(relu_t2_1x1).connectTo(conv_t2_3x3).connectTo(relu_t2_3x3).connectTo(concat);
		input[0].connectTo(conv_t3_1x1).connectTo(relu_t3_1x1).connectTo(conv_t3_5x5).connectTo(relu_t3_5x5).connectTo(concat);
		input[0].connectTo(pool_t4_1x1).connectTo(conv_t4_1x1).connectTo(relu_t4_1x1).connectTo(concat);
		
		if (gradient != null) {
			
			DataflowNode _conv_t1_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-1-1x1", prefix), new ConvGradient (convconf_t1_1x1)).setPeer(conv_t1_1x1.getOperator()));
			DataflowNode _relu_t1_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-1-1x1", prefix), new ReLUGradient (reluconf_t1_1x1)).setPeer(relu_t1_1x1.getOperator()));
			
			DataflowNode _conv_t2_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-2-1x1", prefix), new ConvGradient (convconf_t2_1x1)).setPeer(conv_t2_1x1.getOperator()));
			DataflowNode _relu_t2_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-2-1x1", prefix), new ReLUGradient (reluconf_t2_1x1)).setPeer(relu_t2_1x1.getOperator()));
			
			DataflowNode _conv_t2_3x3 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-2-3x3", prefix), new ConvGradient (convconf_t2_3x3)).setPeer(conv_t2_3x3.getOperator()));
			DataflowNode _relu_t2_3x3 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-2-3x3", prefix), new ReLUGradient (reluconf_t2_3x3)).setPeer(relu_t2_3x3.getOperator()));
			
			DataflowNode _conv_t3_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-3-1x1", prefix), new ConvGradient (convconf_t3_1x1)).setPeer(conv_t3_1x1.getOperator()));
			DataflowNode _relu_t3_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-3-1x1", prefix), new ReLUGradient (reluconf_t3_1x1)).setPeer(relu_t3_1x1.getOperator()));
			
			DataflowNode _conv_t3_5x5 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-3-5x3", prefix), new ConvGradient (convconf_t3_5x5)).setPeer(conv_t3_5x5.getOperator()));
			DataflowNode _relu_t3_5x5 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-3-5x3", prefix), new ReLUGradient (reluconf_t3_5x5)).setPeer(relu_t3_5x5.getOperator()));
			
			DataflowNode _pool_t4_1x1 = new DataflowNode (new Operator (String.format("%s-Pool-Tower-4-1x1", prefix), new PoolGradient (poolconf_t4_1x1)).setPeer(pool_t4_1x1.getOperator()));
			
			DataflowNode _conv_t4_1x1 = new DataflowNode (new Operator (String.format("%s-Conv-Tower-4-1x1", prefix), new ConvGradient (convconf_t4_1x1)).setPeer(conv_t4_1x1.getOperator()));
			DataflowNode _relu_t4_1x1 = new DataflowNode (new Operator (String.format("%s-ReLU-Tower-4-1x1", prefix), new ReLUGradient (reluconf_t4_1x1)).setPeer(relu_t4_1x1.getOperator()));
			
			DataflowNode merge = new DataflowNode (new Operator (String.format("%s-Merge", prefix), new ElementWiseOp (new ElementWiseOpConf ())));
			
			for (int i = 0; i < gradient.length; i++)
				if (gradient[i] != null)
					merge.connectTo (gradient[i]);
			
			DataflowNode [] concatgradient = new DataflowNode [4];
			for (int i = 0; i < 4; i++)
				concatgradient [i] = new DataflowNode (new Operator (String.format("%s-ConcatGradient (%d)", prefix, i), new ConcatGradient (new ConcatConf ())).setPeer(concat.getOperator()));
			
			concatgradient[0].connectTo(_relu_t1_1x1).connectTo(_conv_t1_1x1).connectTo(merge);
			concatgradient[1].connectTo(_relu_t2_3x3).connectTo(_conv_t2_3x3).connectTo(_relu_t2_1x1).connectTo(_conv_t2_1x1).connectTo(merge);
			concatgradient[2].connectTo(_relu_t3_5x5).connectTo(_conv_t3_5x5).connectTo(_relu_t3_1x1).connectTo(_conv_t3_1x1).connectTo(merge);
			concatgradient[3].connectTo(_relu_t4_1x1).connectTo(_conv_t4_1x1).connectTo(_pool_t4_1x1).connectTo(merge);
			
			for (int i = 0; i < 4; i++)
				gradient[i] = concatgradient [i];
		}
		
		input[0] = concat;
		
		return input;
	}
	
	public static DataflowNode buildGoogleNetv1 (boolean gradient) {
		
		/* Input is 128 x 3 x 224 x 224 */
		
		DataflowNode head = null;
		
		DataflowNode [] tail = new DataflowNode [1];
		DataflowNode [] grad = new DataflowNode [4]; /* There are four towers in each inception unit */
		
		for (int i = 0; i < grad.length; i++)
			grad[i] = null;
		
		/* Stage 1 */
		
		ConvConf convconf1 = new ConvConf ();
		
		convconf1.setNumberOfOutputs (64);
		
		convconf1.setKernelSize  (2).setKernelHeight  (7).setKernelWidth  (7);
		convconf1.setStrideSize  (2).setStrideHeight  (2).setStrideWidth  (2);
		convconf1.setPaddingSize (2).setPaddingHeight (3).setPaddingWidth (3);
		
		convconf1.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf1.setBias (false);
		
		ReLUConf reluconf1 = new ReLUConf ();
		
		PoolConf poolconf1 = new PoolConf ().setMethod (PoolMethod.MAX).setKernelSize (3).setStrideSize (2).setPaddingSize (1);
		
		DataflowNode conv1 = new DataflowNode (new Operator ("Stage-1-Conv", new Conv (convconf1)));
		DataflowNode relu1 = new DataflowNode (new Operator ("Stage-1-ReLU", new ReLU (reluconf1)));
		DataflowNode pool1 = new DataflowNode (new Operator ("Stage-1-Pool", new Pool (poolconf1)));
		
		head = conv1;
		tail[0] = conv1.connectTo(relu1).connectTo(pool1);
		
		if (gradient) {
			
			DataflowNode _conv1 = new DataflowNode (new Operator ("Stage-1-ConvGradient", new ConvGradient (convconf1)).setPeer (conv1.getOperator()));
			DataflowNode _relu1 = new DataflowNode (new Operator ("Stage-1-ReLUGradient", new ReLUGradient (reluconf1)).setPeer (relu1.getOperator()));
			DataflowNode _pool1 = new DataflowNode (new Operator ("Stage-1-PoolGradient", new PoolGradient (poolconf1)).setPeer (pool1.getOperator()));
			
			_pool1.connectTo(_relu1).connectTo(_conv1);
			grad[0] = _pool1;
		}
		
		/* Stage 2 */
		
		ConvConf convconf2 = new ConvConf ();
		
		convconf2.setNumberOfOutputs (64);
		
		convconf2.setKernelSize  (2).setKernelHeight  (1).setKernelWidth  (1);
		convconf2.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf2.setPaddingSize (2).setPaddingHeight (0).setPaddingWidth (0);
		
		convconf2.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf2.setBias (false);
		
		ReLUConf reluconf2 = new ReLUConf ();
		
		DataflowNode conv2 = new DataflowNode (new Operator ("Stage-2-Conv", new Conv (convconf2)));
		DataflowNode relu2 = new DataflowNode (new Operator ("Stage-2-ReLU", new ReLU (reluconf2)));
		
		tail[0] = tail[0].connectTo(conv2).connectTo(relu2);
		
		if (gradient) {
			
			DataflowNode _conv2 = new DataflowNode (new Operator ("Stage-2-ConvGradient", new ConvGradient (convconf2)).setPeer (conv2.getOperator()));
			DataflowNode _relu2 = new DataflowNode (new Operator ("Stage-2-ReLUGradient", new ReLUGradient (reluconf2)).setPeer (relu2.getOperator()));
			
			_relu2.connectTo(_conv2).connectTo(grad[0]);
			grad[0] = _relu2;
		}
		
		/* Stage 3 */
		
		ConvConf convconf3 = new ConvConf ();
		
		convconf3.setNumberOfOutputs (192);
		
		convconf3.setKernelSize  (2).setKernelHeight  (3).setKernelWidth  (3);
		convconf3.setStrideSize  (2).setStrideHeight  (1).setStrideWidth  (1);
		convconf3.setPaddingSize (2).setPaddingHeight (1).setPaddingWidth (1);
		
		convconf3.setWeightInitialiser (new InitialiserConf ().setType (InitialiserType.GAUSSIAN).setStd (0.1F));
		convconf3.setBias (false);
		
		ReLUConf reluconf3 = new ReLUConf ();
		
		PoolConf poolconf3 = new PoolConf ().setMethod(PoolMethod.MAX).setKernelSize (3).setStrideSize (2).setPaddingSize (1);
		
		DataflowNode conv3 = new DataflowNode (new Operator ("Stage-3-Conv", new Conv (convconf3)));
		DataflowNode relu3 = new DataflowNode (new Operator ("Stage-3-ReLU", new ReLU (reluconf3)));
		DataflowNode pool3 = new DataflowNode (new Operator ("Stage-3-Pool", new Pool (poolconf3)));
		
		tail[0] = tail[0].connectTo(conv3).connectTo(relu3).connectTo(pool3);
		
		if (gradient) {
			
			DataflowNode _conv3 = new DataflowNode (new Operator ("Stage-3-ConvGradient", new ConvGradient (convconf3)).setPeer (conv3.getOperator()));
			DataflowNode _relu3 = new DataflowNode (new Operator ("Stage-3-ReLUGradient", new ReLUGradient (reluconf3)).setPeer (relu3.getOperator()));
			DataflowNode _pool3 = new DataflowNode (new Operator ("Stage-3-PoolGradient", new PoolGradient (poolconf3)).setPeer (pool3.getOperator()));
			
			_pool3.connectTo(_relu3).connectTo(_conv3).connectTo(grad[0]);
			grad[0] = _pool3;
		}
		
		tail = buildInceptionUnit (tail, "Stage-3 (a)", gradient ? grad : null, new int [][] { {  64 }, {  96, 128 }, { 16, 32 }, {3, 32 } });
		tail = buildInceptionUnit (tail, "Stage-3 (b)", gradient ? grad : null, new int [][] { { 128 }, { 128, 192 }, { 32, 96 }, {3, 64 } });
		
		/* Stage 4 */
		
		PoolConf poolconf4 = new PoolConf ().setMethod (PoolMethod.MAX).setKernelSize (3).setStrideSize (2).setPaddingSize (1);
		
		DataflowNode pool4 = new DataflowNode (new Operator ("Stage-4-Pool", new Pool (poolconf4)));
		
		tail[0] = tail[0].connectTo(pool4);
		
		if (gradient) {
			
			DataflowNode _pool4 = new DataflowNode (new Operator ("Stage-4-PoolGradient", new PoolGradient (poolconf4)).setPeer(pool4.getOperator()));
			
			for (int i = 0; i < grad.length; i++) {
				if (grad[i] != null)
					_pool4.connectTo (grad[i]);
			}
			grad[0] = _pool4;
			for (int i = 1; i < grad.length; i++)
				grad[i] = null;
		}
		
		tail = buildInceptionUnit (tail, "Stage-4 (a)", gradient ? grad : null, new int [][] { { 192 }, {  96, 208 }, { 16,  48 }, { 3,  64 } });
		tail = buildInceptionUnit (tail, "Stage-4 (b)", gradient ? grad : null, new int [][] { { 160 }, { 112, 224 }, { 24,  64 }, { 3,  64 } });
		tail = buildInceptionUnit (tail, "Stage-4 (c)", gradient ? grad : null, new int [][] { { 128 }, { 128, 256 }, { 24,  64 }, { 3,  64 } });
		tail = buildInceptionUnit (tail, "Stage-4 (d)", gradient ? grad : null, new int [][] { { 112 }, { 144, 288 }, { 32,  64 }, { 3,  64 } });
		tail = buildInceptionUnit (tail, "Stage-4 (e)", gradient ? grad : null, new int [][] { { 256 }, { 160, 320 }, { 32, 128 }, { 3, 128 } });	
		
		/* Stage 5 */
		
		PoolConf poolconf5  = new PoolConf ().setMethod(PoolMethod.MAX).setKernelSize(3).setStrideSize(2).setPaddingSize(1);
		
		DataflowNode pool5 = new DataflowNode (new Operator ("Stage-5-Pool", new Pool (poolconf5)));
		
		tail[0] = tail[0].connectTo(pool5);
		
		if (gradient) {
			
			DataflowNode _pool5 = new DataflowNode (new Operator ("Stage-5-PoolGradient", new PoolGradient (poolconf5)).setPeer(pool5.getOperator()));
			
			for (int i = 0; i < grad.length; i++) {
				if (grad[i] != null)
					_pool5.connectTo (grad[i]);
			}
			grad[0] = _pool5;
			for (int i = 1; i < grad.length; i++)
				grad[i] = null;
		}
		
		tail = buildInceptionUnit (tail, "Stage-5 (a)", gradient ? grad : null, new int [][] { { 256 }, { 160, 320 }, { 32, 128 }, { 3, 128 } });
		tail = buildInceptionUnit (tail, "Stage-5 (b)", gradient ? grad : null, new int [][] { { 384 }, { 192, 384 }, { 48, 128 }, { 3, 128 } });
		
		/* Stage 6 */
		
		PoolConf poolconf6  = new PoolConf ().setMethod (PoolMethod.AVERAGE).setKernelSize (7).setStrideSize (1).setPaddingSize (0);
		
		DataflowNode pool6 = new DataflowNode (new Operator ("Stage-6-Pool", new Pool (poolconf6)));
		
		if (gradient) {
			
			DataflowNode _pool6 = new DataflowNode (new Operator ("Stage-6-PoolGradient", new PoolGradient (poolconf6)).setPeer(pool6.getOperator()));
			
			for (int i = 0; i < grad.length; i++) {
				if (grad[i] != null)
					_pool6.connectTo (grad[i]);
			}
			grad[0] = _pool6;
			for (int i = 1; i < grad.length; i++)
				grad[i] = null;
		}
		
		/* Stage 7 */
		
		InnerProductConf ipconf7 = new InnerProductConf ().setNumberOfOutputs (1000);
		
		DataflowNode ip7 = new DataflowNode (new Operator ("Stage-7-Ip", new InnerProduct (ipconf7)));
		
		tail[0] = tail[0].connectTo(pool6).connectTo(ip7);
		
		if (gradient) {
			
			DataflowNode _ip7 = new DataflowNode (new Operator ("Stage-7-IpGradient", new InnerProductGradient (ipconf7)).setPeer(ip7.getOperator()));
			
			_ip7.connectTo(grad[0]);
			grad[0] = _ip7;
		}
		
		/* Connect forward and backward dataflow */
		if (gradient)
			tail[0].connectTo(grad[0]);
		
		return head;
	}
}
