package uk.ac.imperial.lsds.crossbow.preprocess;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;

public class DatasetUtils {
	
	private static CharBuffer cb = CharBuffer.allocate (1); /* 1 character */
	
	private static CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder();
	
	private static char nextChar (ByteBuffer buffer, boolean [] eof) {
		
		if (! buffer.hasRemaining()) 
		{	
			if (eof != null)
				eof[0] = true;
			
			return 0;
		}
		else
		{
			if (eof != null)
				eof[0] = false;
			
			/* Reset character's buffer position to 0 */
			cb.clear();
			
			/* Decode next character */
			CoderResult result = decoder.decode (buffer, cb, true);
			
			/* The result is an overflow since we can only decode a 
			 * single character at the time (due to cb.limit())
			 * 
			 * However, when the last character is read, the result
			 * is an underflow.
			 */
			if (! (result.isOverflow () || (result.isUnderflow() && ! buffer.hasRemaining()))) 
			{	
				System.err.println ("error: character decoding failed: " + result.toString ());
				System.exit (1);
			}

			return cb.get (0);
		}
	}
	
	/**
	 * 
	 * Reads a line of text, based on BufferedReader.readLine(). 
	 * 
	 * A line is considered to be terminated by any one of a linefeed ('\n'), 
	 * a carriage return ('\r'), or a carriage return followed immediately by 
	 * a linefeed.
	 * 
	 * @param buffer
	 * @return 
	 */
	public static String readLine (ByteBuffer buffer) {
		
		boolean [] eof = new boolean [1];
		
		StringBuffer s = new StringBuffer ();
		
		char c = 0;
		
		/* Skip any leading terminating chars (incl. empty lines) */
		for (;;)
		{
			c = nextChar (buffer, eof);
			if (eof[0] || (! (c == '\n' || c == '\r')))
				break;
		}
		
		/* 
		 * We break from the loop is we reached the end of buffer
		 * or we read a non-terminating character.
		 */
		if (eof [0]) 
		{
			return null;
		}
		else 
		{
			/* Append the current character */
			s.append(c);
			
			/* Continue reading until a terminating chatacter is found */
			for (;;)
			{
				c = nextChar (buffer, eof);
				if (eof[0] || (c == '\n' || c == '\r'))
					break;
				
				s.append(c);
			}
			
			/* There's at least one character in `s` */
			return s.toString();
		}
	}
	
	public static String buildPath (String prefix, String suffix, boolean check) {
		
		StringBuilder s = new StringBuilder ();
		
		s.append(prefix);
		
		if (prefix.charAt(prefix.length() - 1) != '/') s.append("/");
		
		s.append(suffix);
		
		String filename = s.toString();
		
		if (check) {
			
			File file = new File (filename);
			if (! file.exists()) {
				
				System.err.println(String.format("error: %s: file not found", filename));
				System.exit(1);
			}
		}
		
		return filename;
	}
	
	/**
	 * Given a directory, it returns an array of all
	 * the image files therein that will be encoded.
	 */
	public static ArrayList<String> walk (String directory) {

		ArrayList<String> arrayList = new ArrayList<String> ();
		
		File root = new File (directory);
		
		if (! root.isDirectory ())
			throw new IllegalArgumentException (String.format("error: %s: not a directory", directory));
		
		File [] list = root.listFiles ();
		
		for (File file: list) {
			
			if (Files.isSymbolicLink(file.toPath ()))
				continue;
			
			if (file.isDirectory ())
				arrayList.addAll (walk (file.getAbsolutePath()));
			else
				arrayList.add (file.getAbsolutePath());
		}
		
		return arrayList;
	}
	
	public static String copy (String filename, String temporaryDirectoryOrFile) throws IOException {
		
		Path source = Paths.get (filename);
		
		/* Path destination = Paths.get (temporaryDirectory).resolve (source.getFileName()); */
		Path destination = Paths.get(temporaryDirectoryOrFile);
		
		Files.copy (source, destination , StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.COPY_ATTRIBUTES);
		
		return destination.toString();
	}
	
	public static void delete (String filename) throws IOException {
		
		Path path = Paths.get(filename);
		Files.delete(path);
	}
}
