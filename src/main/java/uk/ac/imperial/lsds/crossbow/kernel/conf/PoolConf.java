package uk.ac.imperial.lsds.crossbow.kernel.conf;

import uk.ac.imperial.lsds.crossbow.types.PoolMethod;

public class PoolConf implements IConf {
	
	PoolMethod method;
	
	int paddingSize;
	int paddingHeight, paddingWidth;
	
	int kernelSize;
	int kernelHeight, kernelWidth;
	
	int strideSize;
	int strideHeight, strideWidth;
	
	boolean global;
	
	public PoolConf () {
		
		method = PoolMethod.MAX;
		
		paddingSize = -1;
		paddingHeight = paddingWidth = -1;
		
		kernelSize = 0;
		kernelHeight = kernelWidth = 0;
		
		strideSize = 1;
		strideHeight = strideWidth = 0;
		
		global = false;
	}
	
	public PoolConf setMethod (PoolMethod method) {
		this.method = method;
		return this;
	}
	
	public PoolMethod getMethod () {
		return method;
	}
	
	public PoolConf setPaddingSize (int paddingSize) {
		this.paddingSize = paddingSize;
		return this;
	}
	
	public PoolConf setPaddingHeight (int paddingHeight) {
		this.paddingHeight = paddingHeight;
		return this;
	}
	
	public PoolConf setPaddingWidth (int paddingWidth) {
		this.paddingWidth = paddingWidth;
		return this;
	}
	
	public int getPaddingSize () {
		return paddingSize;
	}
	
	public int getPaddingHeight () {
		return paddingHeight;
	}
	
	public int getPaddingWidth () {
		return paddingWidth;
	}
	
	public PoolConf setKernelSize (int kernelSize) {
		this.kernelSize = kernelSize;
		return this;
	}
	
	public PoolConf setKernelHeight (int kernelHeight) {
		this.kernelHeight = kernelHeight;
		return this;
	}
	
	public PoolConf setKernelWidth (int kernelWidth) {
		this.kernelWidth = kernelWidth;
		return this;
	}
	
	public int getKernelSize () {
		return kernelSize;
	}
	
	public int getKernelHeight () {
		return kernelHeight;
	}
	
	public int getKernelWidth () {
		return kernelWidth;
	}
	
	public PoolConf setStrideSize (int strideSize) {
		this.strideSize = strideSize;
		return this;
	}
	
	public PoolConf setStrideHeight (int strideHeight) {
		this.strideHeight = strideHeight;
		return this;
	}
	
	public PoolConf setStrideWidth (int strideWidth) {
		this.strideWidth = strideWidth;
		return this;
	}
	
	public int getStrideSize () {
		return strideSize;
	}
	
	public int getStrideHeight () {
		return strideHeight;
	}
	
	public int getStrideWidth () {
		return strideWidth;
	}
	
	public PoolConf setGlobal (boolean global) {
		this.global = global;
		return this;
	}
	
	public boolean isGlobal () {
		return global;
	}
}
