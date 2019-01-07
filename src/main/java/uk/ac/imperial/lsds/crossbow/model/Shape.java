package uk.ac.imperial.lsds.crossbow.model;

import java.util.Arrays;

public class Shape {
	
	private int [] shape;
	
	public Shape () {
		shape = null;
	}
	
	public Shape (int size) {
		this(new int [size]);
	}
	
	public Shape (int [] shape) {
		this.shape = shape;
	}
	
	public Shape(String value) {
		String s = value.substring(1, value.length() - 1); /* Skip [ ] */
		String [] t = s.split(",");
		shape = new int [t.length];
		for (int i = 0; i < t.length; ++i)
			shape[i] = Integer.parseInt(t[i].trim());
	}
	
	private void checkBounds (int axis, boolean max) {
		int limit = dimensions();
		if (! max)
			limit--;
		if (axis < 0 || axis > limit)
			throw new ArrayIndexOutOfBoundsException (String.format("error: invalid shape axis %d", axis));
	}
	
	public void push (int value) {
		if (shape == null) {
			shape = new int [1];
			shape[0] = value;
		} else {
			/* Resize array */
			int [] newShape = Arrays.copyOf (shape, shape.length + 1);
			newShape[shape.length] = value;
			shape = newShape;
		}
	}
	
	public void set (int axis, int value) {
		checkBounds (axis, false);
		shape[axis] = value; 
	}
	
	public int get (int axis) {
		checkBounds (axis, false);
		return shape[axis];
	}
	
	public int countElements (int start, int end) {
		
		checkBounds (start, true);
		checkBounds (end,   true);
		
	    int count = 1;
	    for (int axis = start; axis < end; ++axis) {
	      count *= get(axis);
	    }
	    return count;
	}
	
	public int countElements (int offset) {
		return countElements (offset, dimensions());
	}
	
	public Shape copy () {
		return new Shape (Arrays.copyOf(shape, shape.length));
	}
	
	public String toString () {
		StringBuilder s = new StringBuilder();
		s.append("[");
		for (int i = 0; i < dimensions(); ++i) {
			s.append(shape[i]);
			if (i < dimensions() - 1)
				s.append(", ");
		}
		s.append("]");
		return s.toString();
	}

	public int dimensions () {
		return shape.length;
	}
	
	public int countAllElements () {
		return countElements (0);
	}
	
	public int numberOfExamples () {
		
		return legacyShape (0);
	}

	public int numberOfChannels () {
		
		return legacyShape (1);
	}
	
	public int height () {
		
		return legacyShape (2);
	}
	
	public int width () {
		
		return legacyShape (3);
	}
	
	private int legacyShape (int axis) {
		
		if (shape.length > 4)
			throw new IllegalStateException (String.format("error: cannot use legacy accessors on %d-D tensors", shape.length));
		
		if (axis > (shape.length - 1))
			return 1;
		else
			return shape[axis];
	}
	
	public int offset (int n, int c) {
		
		return ((n * numberOfChannels() + c) * height()) * width();
	}
	
	public int [] array () {
		return shape;
	}

	public void getNCHW (int [] dimensions) {
		
		dimensions [0] = numberOfExamples ();
		dimensions [1] = numberOfChannels ();
		dimensions [2] = height ();
		dimensions [3] = width ();
	}
	
	public boolean equals (Shape other) {
		
		if (other == null)
			return false;
		
		/* Compare number of dimensions */
		if (shape.length != other.shape.length)
			return false;
		
		/* Compare dimensions */
		for (int i = 0; i < shape.length; ++i) {
			if (shape[i] != other.shape[i])
				return false;
		}
		
		return true;
	}
}
