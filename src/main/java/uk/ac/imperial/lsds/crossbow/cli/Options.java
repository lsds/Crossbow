package uk.ac.imperial.lsds.crossbow.cli;

import java.util.LinkedHashMap;
import java.util.Map;

import uk.ac.imperial.lsds.crossbow.ModelConf;
import uk.ac.imperial.lsds.crossbow.SystemConf;

public class Options {
	
	private Map<String, Option> opts;
	
	private String program;
	
	public Options () {
		
		this("[class]");
	}
	
	public Options (String program) {
		
		opts = new LinkedHashMap<String, Option>();
		
		/* Add default system and model configuration options */
		
		for (Option opt: SystemConf.getInstance().getOptions())
			addOption (opt);
		
		for (Option opt:  ModelConf.getInstance().getOptions())
			addOption (opt);
		
		this.program = program;
	}
	
	public Options addRequiredOption (String opt, String description, Class<?> type) {
		
		return addOption (opt, description, true, type, null);
	}
	
	public Options addRequiredOption (String opt, String description, Class<?> type, String defaultValue) {
		
		return addOption (opt, description, true, type, defaultValue);
	}
	
	public Options addOption (String opt, String description, Class<?> type) {
		
		return addOption (opt, description, false, type, null);
	}
	
	public Options addOption (String opt, String description, Class<?> type, String defaultValue) {
		
		return addOption (opt, description, false, type, defaultValue);
	}
	
	public Options addOption (String opt, String description, boolean required, Class<?> type) {
		
		return addOption (opt, description, required, type, null);
	}
	
	public Options addOption (String opt, String description, boolean required, Class<?> type, String defaultValue) {
		
		Option option = new Option (opt, description, required, type, defaultValue);
		return addOption(option);
	}
	
	public Options addOption (Option option) {
		String key = option.getOpt();
		if (opts.containsKey(key)) {
			System.err.println(String.format("error: option \"%s\" is already set", key));
			System.exit(1);
		}
		opts.put(key, option);
		return this;
	}
	
	public Option getOption (String key) {
		
		return opts.get(key);
	}
	
	public Option find (String opt) {
		
		return getOption(opt);
	}
	
	/**
	 * Check if required options are set.
	 * Exit on the first error.
	 */
	public void check () {
		
		for (Map.Entry<String, Option> entry: opts.entrySet()) {
			
			String key = entry.getKey();
			Option opt = entry.getValue();
			
			if (opt.isRequired() && ! opt.isInitialised()) {
			
				System.err.println(String.format("error: option \"%s\" not set", key));
				System.exit(1);
			}
		}
	}
	
	@Override
	public String toString () {
		
		StringBuilder s = new StringBuilder(String.format("Usage: java %s\n", program));
		
		s.append("\nOptions:\n\n");
		
		int L1 = 0, L2 = 0;
		
		for (Option opt: opts.values()) {
			
			if (L1 < opt.getOpt().length())         
				L1 = opt.getOpt().length();
			
			if (L2 < opt.getDescription().length()) 
				L2 = opt.getDescription().length();
		}
		
		String fmt = String.format("   %%-%ds : %%-%ds (type: %%-7s default value: %%s)\n", L1, L2); /* %-{L1}s: %-{L2}s ... */
		
		for (Option opt: opts.values())
			s.append(String.format(fmt, opt.getOpt(), opt.getDescription(), opt.getTypeString(), opt.getDefaultValue()));
		
		return s.toString();
	}
}
