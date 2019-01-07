package uk.ac.imperial.lsds.crossbow.kernel;

public class KernelMemoryRequirements {
	
	long output, model;
	
	/* Local variable length used for CPU and/or GPU execution. */
	long localOnCPU, localOnGPU;
	
	public KernelMemoryRequirements () {
		
		output = model = localOnCPU = localOnGPU = 0L;
	}
	
	public KernelMemoryRequirements setOutputMemoryRequirements (long bytes) {
		output = bytes;
		return this;
	}
	
	public KernelMemoryRequirements incOutputMemoryRequirements (long bytes) {
		output += bytes;
		return this;
	}
	
	public long getOutputMemoryRequirements (boolean inplace) {
		return (inplace ? 0 : output);
	}
	
	public KernelMemoryRequirements setModelMemoryRequirements (long bytes) {
		model = bytes;
		return this;
	}
	
	public KernelMemoryRequirements incModelMemoryRequirements (long bytes) {
		model += bytes;
		return this;
	}
	
	public long getModelMemoryRequirements () {
		return model;
	}
	
	public KernelMemoryRequirements setLocalCPUMemoryRequirements (long bytes) {
		localOnCPU = bytes;
		return this;
	}
	
	public KernelMemoryRequirements incLocalCPUMemoryRequirements (long bytes) {
		localOnCPU += bytes;
		return this;
	}
	
	public long getLocalCPUMemoryRequirements () {
		return localOnCPU;
	}
	
	public KernelMemoryRequirements setLocalGPUMemoryRequirements (long bytes) {
		localOnGPU = bytes;
		return this;
	}
	
	public KernelMemoryRequirements incLocalGPUMemoryRequirements (long bytes) {
		localOnGPU += bytes;
		return this;
	}
	
	public long getLocalGPUMemoryRequirements () {
		return localOnGPU;
	}
	
	public static String bytesToString (long bytes) {
		
		long KB = 1024;
		long MB = 1024 * KB;
		long GB = 1024 * MB;
		long TB = 1024 * GB;
		
		StringBuilder s = new StringBuilder ();
		
		double value = 0D;
		String suffix = null;
		
		if (bytes >=  0 && bytes < KB) {
			value = (double) bytes;
			suffix = "B";
		} else
		if (bytes >= KB && bytes < MB) {
			value = (double) bytes / (double) KB;
			suffix = "KB";
		} else 
		if (bytes >= MB && bytes < GB) {
			value = (double) bytes / (double) MB;
			suffix = "MB";
		} else
		if (bytes >= GB && bytes < TB) {
			value = (double) bytes / (double) GB;
			suffix = "GB";
		} else {
			throw new IllegalStateException (String.format("Failed to translate %d bytes", bytes));
		}
		
		s.append(String.format("%6.2f %-2s", value, suffix));
		return s.toString();
	}
}
