package uk.ac.imperial.lsds.crossbow.cli;

import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SystemConf;

public class CommandLine {
	
	private Options options;
	
	public CommandLine (Options options) {
		this.options = options;
	}
	
	public void parse (String [] args) {
		
		Option option;
		
		/* Parse command line arguments */
		int i, j;
		
		for (i = 0; i < args.length; ) {
			
			if ((j = i + 1) == args.length) {
				System.err.println(options.toString());
				System.exit(1);
			}
			
			if ((option = options.find(args[i])) != null) {
				
				/* setValue() exits on error */
				option.setValue(args[j]);
				
				/* Is it a system or model configuration option? */
				parseOther (args[i], option);
				
			} else {
				
				System.err.println(String.format("error: invalid option: %s %s", args[i], args[j]));
				System.exit(1);
			}
			
			i = j + 1;
		}
		
		options.check();
		
		return;
	}
	
	private boolean parseOther (String arg, Option opt) {
		
		return (SystemConf.getInstance().parse(arg, opt) || ModelConf.getInstance().parse(arg, opt));
	}
}
