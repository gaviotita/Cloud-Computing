package regression;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class LinearRegression {

  public static class LinearRegressionMapper extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, FloatWritable>
	{

		private Path[] localFiles;
		
		FileInputStream fis = null;
		BufferedInputStream bis = null;
		
		
		@Override
		public void configure(JobConf job)
		{
			/**
			 * Read the distributed cache
			 */
			
			try {
				localFiles = DistributedCache.getLocalCacheFiles(job);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
		
		
		@Override
		public void map(LongWritable key, Text value, OutputCollector<LongWritable, FloatWritable> output,
				Reporter reporter) throws IOException {
			
			/**
			 *  
			 *  Linear-Regression costs function
			 *  
			 *  This will simply sum over the subset and calculate the predicted value y_predict(x) for the given features values and the current theta values
			 *  Then it will subtract the true y values from the y_predict(x) value for every input record in the subset
			 *  
			 *  J(theta) = sum((y_predict(x)-y)^2)
			 *  y_predict(x) = theta(0)*x(0) + .... + theta(i)*x(i)
			 * 
			 */
			
			String line = value.toString();
			String[] features = line.split(",");
			
			List<Float> values = new ArrayList<Float>();
			
			/**
			 * read the values and convert them to floats
			 */
			for(int i = 0; i<features.length; i++)
			{
				values.add(new Float(features[i]));
			}
			
			/**
			 * calculate the costs
			 * 
			 */
			
			output.collect(new LongWritable(1), new FloatWritable(costs(values)));
			
			
			
		}
		
		private final float costs(List<Float> values)
		{
			/**
			 * Load the cache files
			 */
			
			File file = new File(localFiles[0].toString());
			
			float costs = 0;
			
			try {
				fis = new FileInputStream(file);
				bis = new BufferedInputStream(fis);
				
				BufferedReader d = new BufferedReader(new InputStreamReader(bis));
				String line = d.readLine();
				
				//all right we have all the theta values, lets convert them to floats
				String[] theta = line.split(",");
				
				//first value is the y value
				float y = values.get(0);
				
				/**
				 * Calculate the costs for each record in values
				 */
				for(int j = 0; j < values.size(); j++)
				{

						//bias calculation
						if(j == 0)
							costs += (new Float(theta[j]))*1;
						else
							costs += (new Float(theta[j]))*values.get(j);

					
				}
				
				// Subtract y and square the costs
				costs = (costs -y)*(costs - y);
				
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			
			
			return costs;
			
		}
		
	}
	
	
	
	
	public static class LinearRegressionReducer extends MapReduceBase implements Reducer<LongWritable, FloatWritable, LongWritable, FloatWritable>
	{

		@Override
		public void reduce(LongWritable key, Iterator<FloatWritable> value,
				OutputCollector<LongWritable, FloatWritable> output, Reporter reporter)
				throws IOException {
			
			
			/**
			 * The reducer just has to sum all the values for a given key
			 * 
			 */
			
			float sum = 0;
			
			while(value.hasNext())
			{
				sum += value.next().get();
			}
			
			output.collect(key, new FloatWritable(sum));
			
		}
		
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		//the class is LinearRegression
		JobConf conf = new JobConf(LinearRegression.class);
		
		//the jobname is linearregression (this can be anything)
		conf.setJobName("linearregression");
		
		/**
		* Try to load the theta values into the distributed cache
		*/
		try {
		  //make sure this is your path to the cache file in the hadoop file system
			DistributedCache.addCacheFile(
			  new URI(args[2]), conf);
		} catch (URISyntaxException e1) {
			e1.printStackTrace();
		}
		
		//set the output key class 
		conf.setOutputKeyClass(LongWritable.class);
		//set the output value class
		conf.setOutputValueClass(FloatWritable.class);
		
		//set the mapper
		conf.setMapperClass(LinearRegressionMapper.class);
		//set the combiner
		conf.setCombinerClass(LinearRegressionReducer.class);
		//set the reducer
		conf.setReducerClass(LinearRegressionReducer.class);
		
		//set the input format
		conf.setInputFormat(TextInputFormat.class);
		//set the output format
		conf.setOutputFormat(TextOutputFormat.class);
		
		//set the input path (from args)
		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		//set the output path (from args)
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));
		
		//try to run the job
		try {
			JobClient.runJob(conf);
		} catch (IOException e) {
			e.printStackTrace();
		}

		

	}

}
