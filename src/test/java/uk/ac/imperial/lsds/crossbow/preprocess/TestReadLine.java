package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.IOException;
import java.nio.ByteBuffer;

import uk.ac.imperial.lsds.crossbow.SystemConf;

public class TestReadLine {
	
	public static void main (String [] args) throws IOException {
		
		DatasetFile file = new DatasetFile (SystemConf.getInstance().getHomeDirectory() + "/data/ratings/ratings.txt");
		
		ByteBuffer buffer = file.getByteBuffer();
		
		System.out.println (String.format ("Buffer position %d, limit %d", buffer.position (), buffer.limit ()));
		
		for (int i = 0; i < 10; i++)
			System.out.println (i + ": " + DatasetUtils.readLine (buffer));
		
		System.out.println (String.format("New buffer position %d, limit %d", buffer.position (), buffer.limit ()));
	}
}
