package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.types.ElementWiseOpType;

public class ElementWiseOpConf implements IConf {
	
	private ElementWiseOpType type;
	
	private float [] coefficients;
	
	public ElementWiseOpConf () {
		
		this.type = ElementWiseOpType.SUM;
		this.coefficients = null;
	}
	
	public ElementWiseOpConf setType (ElementWiseOpType type) {
		this.type = type;
		return this;
	}
	
	public ElementWiseOpType getType () {
		return type;
	}
	
	public ElementWiseOpConf setCoefficients (float ... coefficients) {
		this.coefficients = coefficients;
		return this;
	}
	
	public int numberOfCoefficients () {
		return (coefficients == null) ? 0 : coefficients.length;
	}
	
	public float [] getCoefficients () {
		return coefficients;
	}
	
	public void resizeCoefficients (int length) {
		
		if (coefficients == null) {
			
			coefficients = new float [length];
			for (int i = 0; i < length; ++i)
				coefficients [i] = 1;
			
		} else {
		
			if (coefficients.length > length)
				throw new IllegalArgumentException ("error: new length must be greater than current array length");
		
			if (coefficients.length == length)
				return;
			
			float [] _coefficients = new float [length];
			for (int i = 0; i < length; ++i)
				_coefficients [i] = (i < coefficients.length) ? coefficients [i] : 1;
			
			coefficients = _coefficients;
		}
	}
}
