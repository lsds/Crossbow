package uk.ac.imperial.lsds.crossbow.types;

public enum CudnnKernelType {
	
	UNDEFINED(0), CONV(1), POOL(2), RELU(3), SOFTMAX(4), BATCHNORM(5), DROPOUT(6);
	
	private int id;
	
	CudnnKernelType (int id) {
		this.id = id;
	}
	
	public int getId () {
		return id;
	}
	
	public String toString () {
		switch (id) {
		case 0: return "Undefined";
		case 1: return "Convolution";
		case 2: return "Pooling";
		case 3: return "ReLU";
		case 4: return "SoftMax";
		case 5: return "BatchNorm";
		case 6: return "Dropout";
		default:
			throw new IllegalArgumentException ("error: invalid cuDNN kernel type");
		}
	}
}
