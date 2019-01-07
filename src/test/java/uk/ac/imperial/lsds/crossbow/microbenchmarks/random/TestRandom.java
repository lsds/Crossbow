package uk.ac.imperial.lsds.crossbow.microbenchmarks.random;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import uk.ac.imperial.lsds.crossbow.device.random.RandomGenerator;

public class TestRandom {
	
	public static void main (String [] args) {
		
		RandomGenerator.getInstance().load();
		RandomGenerator.getInstance().test();
		RandomGenerator.getInstance().init(123456789L);
		
		ByteBuffer buffer = ByteBuffer.allocateDirect(1048576).order(ByteOrder.LITTLE_ENDIAN);
		
		RandomGenerator.getInstance().randomGaussianFill(buffer, 32768, 0F, 1F, 0);
		
		float checksum = 0;
		
		for (int i = 0; i < 32768; ++i)
			checksum += buffer.getFloat(i * 4);
		
		System.out.println("Checksum is " + checksum);
		
		System.out.println("Bye.");
		System.exit (0);
	}
}
