package org.clustering.demo;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;

public class GaussianMixtureClustering extends ConsensusCluster {

	/**
	 * The main method. It triggers iterations() implemented by the 
	 * parent class.
	 */
	public static void main(String[] args) throws Exception {
		Logger.getRootLogger().setLevel(Level.WARN);
		(new BisectingKMeansClustering()).iterations();
	}

	/**
	 * This is the abstract method defined in ConsensusCluster whose implementation
	 * is left to child classes. Here we first create a GaussianMixture object and then call
	 * its run() method on data to obtain a GaussianMixtureModel object. Then, we call the 
	 * predict method on GaussianMixtureModel object to obtain a data structure, clusterIndexes,
	 * that gives the cluster indexes for the input parameter data. The clusterIndexes
	 * and data have the same number of elements. The elements in the clusterIndexes and
	 * data have reciprocal sequence in the sense that the i-th element of clusterIndexes
	 * defines which cluster the i-th element of data belongs to, one of 0, 1, 2 ... etc.
	 * 
	 * @param data - Data points to partition into clusters
	 * @param numClusters - number of clusters desired
	 * @param numIterations - maximum # of iterations to perform during clustering
	 * 
	 */
	protected JavaRDD<Integer> dataCenters(JavaRDD<Vector> data, int numClusters, int numIterations) {
		GaussianMixture gm = new GaussianMixture().setK(numClusters).setMaxIterations(numIterations);
		GaussianMixtureModel clusters = gm.run(data.rdd());
		JavaRDD<Integer> clusterIndexes = clusters.predict(data);
		return clusterIndexes;
	}
}
