package uk.ac.imperial.lsds.crossbow.cli;

import java.io.File;

public class Option {
	
	private boolean initialised = false;
	
	private String opt;
	
	private boolean required;
	
	private String description;
	
	private Class<?> type = String.class;
	
	private String value, defaultValue = null;
	
	public Option (String opt) {
		
		this (opt, null, false, String.class, null);
	}
	
	public Option (String opt, String description) {
		
		this (opt, description, false, String.class, null);
	}
	
	public Option (String opt, String description, boolean required) {
		
		this (opt, description, required, String.class, null);
	}
	
	public Option (String opt, String description, boolean required, Class<?> type) {
		
		this(opt, description, required, type, null);
	}
	
	public Option (String opt, String description, boolean required, Class<?> type, String defaultValue) {
		
		this.opt = opt;
		
		this.description = description;
		
		this.required = required;
		
		this.type = type;
		
		this.value = null;
		
		if (defaultValue != null && ! isValid (defaultValue)) {
			
			System.err.println(String.format("error: invalid option: %s %s", opt, defaultValue));
			System.exit(1);
		}
		
		this.defaultValue = defaultValue;
		
		this.initialised = (defaultValue == null) ? false : true;
	}
	
	public Option setOpt (String opt) {
		
		this.opt = opt;
		
		return this;
	}
	
	public String getOpt () {
		
		return opt;
	}
	
	public Option setRequired (boolean required) {
		
		this.required = required;
		
		return this;
	}
	
	public boolean isRequired () {
		
		return required;
	}
	
	public Option setDescription (String description) {
		
		this.description = description;
		
		return this;
	}
	
	public String getDescription () {
		
		return description;
	}
	
	public Option setType (Class<?> type) {
		
		this.type = type;
		
		return this;
	}
	
	public Class<?> getType () {
		
		return type;
	}
	
	public Option setDefaultValue (String defaultValue) {
		
		if (defaultValue == null) {
			
			System.err.println(String.format("error: option %s value is null", opt));
			System.exit(1);
		}
		
		if (! isValid(defaultValue)) {
			
			System.err.println(String.format("error: invalid option: %s %s", opt, defaultValue));
			System.exit(1);
		}
		
		this.defaultValue = defaultValue;
		this.initialised = true;
		
		return this;
	}
	
	public String getDefaultValue () {
		
		return defaultValue;
	}
	
	public Option setValue (String value) {
		
		if (value == null) {
			
			System.err.println(String.format("error: option %s value is null", opt));
			System.exit(1);
		}
		
		if (! isValid(value)) {
			
			System.err.println(String.format("error: invalid option: %s %s", opt, value));
			System.exit(1);
		}
		
		this.value = value;
		this.initialised = true;
		
		return this;
	}
	
	public String getStringValue () {
		
		return (value == null) ? defaultValue : value;
	}
	
	public int getIntValue () {
		
		return Integer.parseInt(getStringValue ());
	}
	
	public long getLongValue () {
		
		return Long.parseLong(getStringValue ());
	}
	
	public float getFloatValue () {
		
		return Float.parseFloat(getStringValue ());
	}
	
	public double getDoubleValue () {
		
		return Double.parseDouble(getStringValue ());
	}
	
	public File getFileValue () {
		
		return new File (getStringValue ());
	}
	
	public boolean getBooleanValue () {
		
		return Boolean.parseBoolean(getStringValue ());
	}
	
	public boolean isInitialised () {
		return initialised;
	}
	
	private boolean isValid (String v) {
		if (type == String.class) 
		{
			return true;
		} 
		else if (type == Integer.class) 
		{	
			try 
			{ 
				Integer.parseInt(v);
				return true;
			} 
			catch (NumberFormatException e) 
			{ 
				return false; 
			}	
		} 
		else if (type == Long.class) 
		{	
			try 
			{ 
				Long.parseLong(v);
				return true;
			} 
			catch (NumberFormatException e) 
			{ 
				return false; 
			}	
		} 
		else if (type == Float.class) 
		{	
			try 
			{ 
				Float.parseFloat(v);
				return true;
			} 
			catch (NumberFormatException e) 
			{ 
				return false; 
			}
		} 
		else if (type == Double.class) 
		{	
			try 
			{ 
				Double.parseDouble(v);
				return true;
			} 
			catch (NumberFormatException e) 
			{ 
				return false; 
			}
		} 
		else if (type == File.class) 
		{	
			return (new File(v)).exists();
		}
		else if (type == Boolean.class)
		{
			if (v.equals("true") || v.equals("false"))
				return true;
			return false;
		}
		
		return false;
	}

	public String getTypeString() {
		
		if (type == String.class) 
		{
			return "string";
		}
		else if (type == Integer.class) 
		{
			return "integer";
		}
		else if (type == Long.class) 
		{
			return "long";
		}
		else if (type == Float.class) 
		{
			return "float";
		}
		else if (type == Double.class) 
		{
			return "double";
		}
		else if (type == File.class) 
		{
			return "file";
		}
		else if (type == Boolean.class)
		{
			return "boolean";
		}
		else 
		{
			System.err.println("error: unknown option type: " + type);
			System.exit(1);
		}
		
		return null;
	}
}
