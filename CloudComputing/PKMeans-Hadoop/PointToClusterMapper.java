import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * You can modify this class as you see fit.  You may assume that the global
 * centroids have been correctly initialized.
 */
public class PointToClusterMapper extends Mapper<Text, Text, IntWritable, Point> {

    @Override
    protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        Point nearest = null;
        float nearestDistance = Float.MAX_VALUE;

        Point point = new Point(key.toString());
        int centroidIndex = 0;

        for (int i = 0; i < KMeans.centroids.size(); i++) {
            Point centroid = KMeans.centroids.get(i);
            float distance = Point.distance(centroid, point);
            if (nearest == null || distance < nearestDistance) {
                nearest = centroid;
                centroidIndex = i;
                nearestDistance = distance;
            }
        }

        context.write(new IntWritable(centroidIndex), point);
    }
}
