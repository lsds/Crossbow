package uk.ac.imperial.lsds.crossbow.kernel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.Batch;
import uk.ac.imperial.lsds.crossbow.Operator;
import uk.ac.imperial.lsds.crossbow.kernel.conf.LRNConf;
import uk.ac.imperial.lsds.crossbow.model.Model;
import uk.ac.imperial.lsds.crossbow.model.Shape;
import uk.ac.imperial.lsds.crossbow.task.ITask;
import uk.ac.imperial.lsds.crossbow.types.ModelAccess;

/*
 * The Local Response Normalisation (LRN) operator 
 */
public class LRN extends Kernel {

	@SuppressWarnings("unused")
	private final static Logger log = LogManager.getLogger (LRN.class);

	LRNConf conf;

	public LRN (LRNConf conf) {
		this.conf = conf;
	}

	public LRN setup (Shape [] inputShape, Model model) {
		return this;
	}

	public void GPURegister () {
	}

	public void compute (Operator [] previous, Batch batch, Model model, ITask api) {
	}

	public ModelAccess getModelAccessType () {
		return ModelAccess.NA;
	}
	
	public boolean isLossKernel () {
		return false;
	}

	public boolean isAccuracyKernel () {
		return false;
	}

	public boolean isDataTransformationKernel () {
		return false;
	}
	
	public boolean allowsOutputOverwrite () {
		return false;
	}
	
	public boolean allowsInputOverwrite () {
		return false;
	}
}
