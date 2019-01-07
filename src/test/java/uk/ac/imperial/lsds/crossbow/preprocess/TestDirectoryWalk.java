package uk.ac.imperial.lsds.crossbow.preprocess;

import java.util.ArrayList;

public class TestDirectoryWalk {
	
	public static void main (String [] args) {
		
		String directory = "/Users/akolious/Development/";
		
		ArrayList<String> fileList = DatasetUtils.walk (directory);
		
		System.out.println (String.format("%d files in %s", fileList.size (), directory));
		
		int id = 1;
		for (String s: fileList) {
			
			System.out.println (String.format("%3d: %s", id++, s));
		}
	}
}
