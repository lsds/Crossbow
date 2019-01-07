package uk.ac.imperial.lsds.crossbow.kernel.conf;

/**
 * Based on Caffe/data_transformer.cpp
 *
 */
public class DataTransformConf implements IConf {

	private int cropSize;
	
	private float scaleFactor;
	
	private boolean mirror;
	
	private String meanImageFilename;
	
	private float [] meanPixelValues;
	
	boolean trainingPhase;
	
	public DataTransformConf () {
		
		cropSize = 0;
		
		scaleFactor = 1F;
		
		mirror = false;
		
		meanImageFilename = null;
		meanPixelValues = null;
		
		trainingPhase = false;
	}
	
	public int getCropSize () {
		return cropSize;
	}
	
	public DataTransformConf setCropSize (int cropSize) {
		this.cropSize = cropSize;
		return this;
	}
	
	public float getScaleFactor () {
		return scaleFactor;
	}
	
	public DataTransformConf setScaleFactor (float scaleFactor) {
		this.scaleFactor = scaleFactor;
		return this;
	}
	
	public boolean getMirror () {
		return mirror;
	}
	
	public DataTransformConf setMirror (boolean mirror) {
		this.mirror = mirror;
		return this;
	}
	
	public String getMeanImageFilename () {
		return meanImageFilename;
	}
	
	public boolean hasMeanImage () {
		return (meanImageFilename != null);
	}
	
	public DataTransformConf setMeanImageFilename (String meanImageFilename) {
		this.meanImageFilename = meanImageFilename;
		return this;
	}
	
	public float [] getMeanPixelValues () {
		return meanPixelValues;
	}
	
	public boolean hasMeanPixel () {
		return (meanPixelValues != null);
	}
	
	public DataTransformConf setMeanPixelValues (float ... meanPixelValues) {
		this.meanPixelValues = meanPixelValues;
		return this;
	}
	
	public boolean subtractMean () {
		return (hasMeanImage() || hasMeanPixel()); 
	}
}
