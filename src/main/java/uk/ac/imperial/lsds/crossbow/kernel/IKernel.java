package uk.ac.imperial.lsds.crossbow.kernel;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.DataType;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

public interface IKernel {
	
	public void setOperator (Operator operator);
	
	public IKernel setup (Shape [] inputShape, Model model);
	
	public void GPURegister ();
	
	public void compute (Operator [] previous, Batch batch, Model model, ITask api);
	
	public Shape getOutputShape ();
	public DataType getOutputType ();
	public int getOutputSize ();
	
	public ModelAccess getModelAccessType ();
	
	public boolean isLossKernel ();
	public boolean isAccuracyKernel ();
	public boolean isDataTransformationKernel ();
	
	public boolean allowsOutputOverwrite ();
	
	public boolean allowsInputOverwrite ();
	
	public KernelMemoryRequirements getKernelMemoryRequirements ();
}
