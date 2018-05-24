package org.clustering.demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public abstract class ConsensusCluster{
	// Number of cluster we would like to construct
	protected static int numClusters = 3;

	// Stores how many times two data points were assigned to the same cluster across all iterations.
	//
	// A data point is represented as a Row object. Key to the HashMap is a data point (outer point) 
	// and the corresponding value is a HashMap that maps a data point (inner point) to an integer. 
	// Hence, to find how many times two data points were assigned to the same cluster across all 
	// iterations pass one of those data points (outer point) to countSameCluster to get the corresponding
	// HashMap. Then, pass the second data point (inner point) to that HashMap to get the integer value,
	// which is how many times those data points were assigned to the same cluster.
	protected HashMap<String, HashMap<String, Integer>> countSameCluster = new HashMap<String, 
			HashMap<String, Integer>>();
	
	// Stores how many times two data points were assigned to the same data sample across all iterations.
	//
	// A data point is represented as a Row object. Key to the HashMap is a data point (outer point) 
	// and the corresponding value is a HashMap that maps a data point (inner point) to an integer.
	// Hence, to find how many times two data points were in the same data sample across all 
	// iterations pass one of those data points (outer point) to countSameSample to get the corresponding 
	// HashMap. Then, pass the second data point (inner point) to that HashMap to get the integer value,
	// which is how many times those data points were in the same data sample.
	protected HashMap<String, HashMap<String, Integer>> countSameSample = new HashMap<String, 
			HashMap<String, Integer>>();
	


	/**
	 * Initialize 'countSameCluster' & 'countSameCluster' from full data set
	 * 
	 * @param fullData: full data set
	 */
	protected void initializeCounts(JavaRDD<String> fullData) {
		fullData.cache();
		List<String> pointsOuter = fullData.collect();
		List<String> pointsInner = fullData.collect();

		System.out.println("# unique data points: " + pointsInner.size());

		for (String pointOuter : pointsOuter) {
			HashMap<String, Integer> map = new HashMap<String, Integer>();
			for (String pointInner : pointsInner) {
				map.put(pointInner, 0);
			}
			countSameCluster.put(pointOuter, map);
		}

		for (String pointOuter : pointsOuter) {
			HashMap<String, Integer> map = new HashMap<String, Integer>();
			for (String pointInner : pointsInner) {
				map.put(pointInner, 0);
			}
			countSameSample.put(pointOuter, map);
		}
	}

	/**
	 * Calculate and print out histogram data. Data for 10 bins will be printed out:
	 * 0.0 <number to display in bin [0.0,0.1)>
	 * 0.1 <number to display in bin [0.1,0.2)>
	 * 0.2 <number to display in bin [0.2,0.3)>
	 * ...
	 * 0.9 <number to display in bin [0.9,1.0]>
	 */
	protected void generateHistogram() {
		// range: 0.1
		HashMap<Integer, Integer> histogram = new HashMap<Integer, Integer>();
		
		// Initialize with all 0's.
		histogram.put(0, 0);
		histogram.put(1, 0);
		histogram.put(2, 0);
		histogram.put(3, 0);
		histogram.put(4, 0);
		histogram.put(5, 0);
		histogram.put(6, 0);
		histogram.put(7, 0);
		histogram.put(8, 0);
		histogram.put(9, 0);

		for (String sOuter : countSameCluster.keySet()) {
			// sOuter is a particular data point.  
			// inSameCluster stores how many times a particular data point was in the same cluster as sOuter
			// inSameSample stores how many times a particular data point was in the same subsample as sOuter
			HashMap<String, Integer> inSameCluster = countSameCluster.get(sOuter);
			HashMap<String, Integer> inSameSample = countSameSample.get(sOuter);

			for (String sInner : inSameCluster.keySet()) {
				// sInner is a particular data point that was in the same cluster as sOuter
				if (sOuter.equals(sInner)) 
					continue; // Ignore self
				
				// how many times sInner and sOuter were in the same cluster
				int numTimesInSameCluster = inSameCluster.get(sInner);
				
				// how many times sInner and sOuter were in the same subsample
				int numTimesInSameSample = inSameSample.get(sInner);
				
				// Calculate the ratio and place into the corresponding bin
				float ratio = (numTimesInSameCluster == 0) ? 0f : 
					(float) numTimesInSameCluster / numTimesInSameSample;

				if (0 <= ratio && ratio < 0.1) {
					int val = histogram.get(0);
					val++;
					histogram.put(0, val);
				} else if (0.1 <= ratio && ratio < 0.2) {
					int val = histogram.get(1);
					val++;
					histogram.put(1, val);
				} else if (0.2 <= ratio && ratio < 0.3) {
					int val = histogram.get(2);
					val++;
					histogram.put(2, val);
				} else if (0.3 <= ratio && ratio < 0.4) {
					int val = histogram.get(3);
					val++;
					histogram.put(3, val);
				} else if (0.4 <= ratio && ratio < 0.5) {
					int val = histogram.get(4);
					val++;
					histogram.put(4, val);
				} else if (0.5 <= ratio && ratio < 0.6) {
					int val = histogram.get(5);
					val++;
					histogram.put(5, val);
				} else if (0.6 <= ratio && ratio < 0.7) {
					int val = histogram.get(6);
					val++;
					histogram.put(6, val);
				} else if (0.7 <= ratio && ratio < 0.8) {
					int val = histogram.get(7);
					val++;
					histogram.put(7, val);
				} else if (0.8 <= ratio && ratio < 0.9) {
					int val = histogram.get(8);
					val++;
					histogram.put(8, val);
				} else if (0.9 <= ratio && ratio <= 1) {
					int val = histogram.get(9);
					val++;
					histogram.put(9, val);
				}

			}
		}
		// Display histogram data
		System.out.println("HISTOGRAM");
		for (int i : histogram.keySet()) {
			System.out.println(String.format("%.1f", (double)(i * 0.1)) + " " + histogram.get(i));			
		}
	}

	/**
	 * Given a subsample update the 'countSameSample' data structure.
	 * @param collectedSampledData - Data points in a subsample
	 */
	protected void updateSameSampleCounts(List<String> collectedSampledData) {
		ArrayList<String> clonedData = new ArrayList<String>();
		for (String s : collectedSampledData) {
			clonedData.add(s);
		}
				 
		for (String s : collectedSampledData) {
			// Get all the points in 'countSameSample' for the particular
			// data point s in subsample
			HashMap<String, Integer> allPoints = countSameSample.get(s);
			
			for (String c : clonedData) {
				if(s.equals(c)){
					continue; // ignore self
				}
				// Increment # times c & s were together in a subsample
				int currentVal = allPoints.get(c);
				currentVal++;
				allPoints.put(c, currentVal);
			}
		}
	}

	/**
	 * Given a cluster update 'countSameCluster' data structure
	 * from data points in a cluster
	 * @param clusterPoints - Data points in a cluster
	 */
	protected void updateSameClusterCounts(HashMap<Integer, Set<String>> clusterPoints) {
		// Each element in keys corresponds to a particular cluster, 0, 1, 2
		Set<Integer> keys = clusterPoints.keySet();
		
		for (int i : keys) {
			// Obtain points in that cluster
			Set<String> pointsOuter = clusterPoints.get(i);
			Set<String> pointsInner = clusterPoints.get(i);
			
			for (String pointOuter : pointsOuter) {
				// Get all the points in 'countSameCluster' for the 
				// particular data point pointOuter in cluster
				HashMap<String, Integer> allPoints = countSameCluster.get(pointOuter);
				
				for (String pointInner : pointsInner) {
					if(pointOuter.equals(pointInner)){
						continue; // ignore self
					}
					// Increment # times pointInner & pointOuter were together in a cluster
					int currentVal = allPoints.get(pointInner);
					currentVal++;
					allPoints.put(pointInner, currentVal);
				}
			}
		}
	}
	

	/**
	 * This is the main method that performs all the tasks. 
	 */
	protected void iterations() {
		// Set application name
		String appName = "Consensus Clustering Demo";
		
		// # iterations to determine clusters 
		int numIterations = 125;
		
		// Initialize Spark configuration & context
		SparkConf sparkConf = new SparkConf().setAppName(appName).
				setMaster("local[1]").set("spark.executor.memory",
				"1g");

		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Read data file from file system.
		String path = "resources/TP53-VarClsf.txt";

		// Read the data file and return it as JavaRDD of strings
		JavaRDD<String> nonUniqueData = sc.textFile(path);
		System.out.println(nonUniqueData.count());

		// Remove any duplicates
		JavaRDD<String> fullData = nonUniqueData.distinct();
		System.out.println(fullData.count());
		
		// Initialize global data structures
		initializeCounts(fullData);

		
		// Number of iterations
		int MAX_ITER = 20;
		
		// Each execution of this loop corresponds to an iteration 
		for (int iteration = 0; iteration <= MAX_ITER; iteration++) {
			// Obtain a random subsample, consisting of 90% of the original data set
			JavaRDD<String> sampledData = fullData.sample(false, 0.9, (new Random().nextLong()));
			
			// Convert data point, expressed as a string consisting of 3 numbers, into a 3-D vector
			// of doubles using the map function
			JavaRDD<Vector> data = sampledData.map(mapFunction);

			data.cache();

			// Rely on concrete subclasses for this method. This is where clusters are created
			// by a specific algorithm. Each element in clusterIndexes belongs to a cluster, 0, 1, 2
			JavaRDD<Integer> clusterIndexes = dataCenters(data, numClusters, numIterations);

			// Bring all data to driver node for displaying results.
			// 'collectedSampledData' and 'collectedClusterIndexes' are ordered collections
			// with the same size: i-th element in 'collectedSampledData' is a data point that 
			// belongs to the cluster defined in the i-th element in 'collectedClusterIndexes'. 
			// For example, if i-th element of collectedSampledData is [3 6 1] and the i-th element
			// of collectedClusterIndexes is [1] then data point [3 6 1] belongs to cluster 1.
			List<String> collectedSampledData = sampledData.collect();
			List<Integer> collectedClusterIndexes = clusterIndexes.collect();

			System.out.println(collectedSampledData.size() + " " + collectedClusterIndexes.size());

			// Update global data structures based on cluster data
			updateHistogramData(collectedSampledData, collectedClusterIndexes);
		}

		// Generate histogram
		generateHistogram();
		
		// Stop spark
		sc.stop();
		sc.close();
	}

	/**
	 * Converts a data point, expressed as a string consisting of 3 numbers, into a 3-D vector
	 * of doubles.
	 */
	@SuppressWarnings("serial")
	static Function<String, Vector> mapFunction = new Function<String, Vector>() {
		public Vector call(String s) {
			String[] sarray = s.split(" ");
			double[] values = new double[sarray.length]; 																														
																															
			for (int i = 0; i < sarray.length; i++)
				values[i] = Double.parseDouble(sarray[i]);
			
			return Vectors.dense(values);
		}
	};

	/**
	 * Partition subsampled data into clusters. Implemented by a concrete class based on a specific algorithm.
	 * @param data - subsampled data
	 * @param numClusters - # clusters to create
	 * @param numIterations - # iterations to perform
	 * @return JavaRDD<Integer> - This data structure has the same number of elements as input parameter data.
	 * Each element in JavaRDD<Integer> indicates which cluster the corresponding element in data  belongs to 
	 * 0, 1, 2. 
	 *  
	 */
	protected abstract JavaRDD<Integer> dataCenters(JavaRDD<Vector> data, int numClusters,
			int numIterations);
	

	/**
	 * At end of an iteration, update global data structures based on the subsample used in the iteration
	 * and data points in the clusters created in that iteration 
	 * 
	 * @param collectedSampledData - An ordered collection storing data points where point at i-th element 
	 * belongs to the cluster defined in i-th element of collectedClusterIndexes
	 * @param collectedClusterIndexes - An ordered collection where each element represents a cluster, 0, 1, 2
	 * @param clusterCenters - Coordinates of each cluster
	 */
	protected void updateHistogramData(List<String> collectedSampledData, List<Integer> collectedClusterIndexes){

		// Update the 'countSameSample' data structure
		updateSameSampleCounts(collectedSampledData);
		
		// Key to 'clusterPoints' is a cluster identifier, e.g. 0, 1, 2
		// The value is a Set where each element is a data point in the corresponding
		// cluster; data point is represented by its coordinates - a string of whitespace separated 
		// numbers 
		HashMap<Integer, Set<String>> clusterPoints = new HashMap<Integer, Set<String>>();
		int j = 0;

		for (Integer i : collectedClusterIndexes) {
			Set<String> points = clusterPoints.get(i);
			if (points == null) {
				points = new HashSet<String>();
				clusterPoints.put(i, points);
			}
			String tempRow = collectedSampledData.get(j++);
			points.add(tempRow);
		}
		
		// Update the 'countSameCluster' data structure
		updateSameClusterCounts(clusterPoints);
  }

}
