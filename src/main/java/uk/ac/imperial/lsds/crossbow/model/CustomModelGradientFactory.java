package uk.ac.imperial.lsds.crossbow.model;

import uk.ac.imperial.lsds.crossbow.utils.IObjectFactory;

public class CustomModelGradientFactory implements IObjectFactory<ModelGradient> {
	
	private Model model;
	
	public CustomModelGradientFactory (Model model) {
		this.model = model;
	}
	
	@Override
	public ModelGradient newInstance () {
		
		return new ModelGradient (model);
	}

	@Override
	public ModelGradient newInstance (int ndx) {
		
		return new ModelGradient (ndx, model);
	}
}
