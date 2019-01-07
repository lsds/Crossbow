package uk.ac.imperial.lsds.crossbow;

import uk.ac.imperial.lsds.crossbow.device.ObjectRef;

public class TestObjectRef {
	
	public static void main (String [] args) throws Exception {
		
		ObjectRef.getInstance().init();
		
		ObjectRef.getInstance().create(5);
		
		Integer x = ObjectRef.getInstance().get();
		if (x == null)
			System.out.println("Object is null");
		else
			System.out.println("Object is " + x);
		
		Integer y = x;
		Integer z = y;
		
		if (ObjectRef.getInstance().test(z) != 0)
			System.out.println("Object global reference holds!");
		else
			System.out.println("Object global reference does not hold.");
		
		System.out.println("Bye.");
		System.exit(1);
	}
}
