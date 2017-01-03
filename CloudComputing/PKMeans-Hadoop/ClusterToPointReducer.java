import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * You can modify this class as you see fit, as long as you correctly update the
 * global centroids.
 */
public class ClusterToPointReducer extends Reducer<IntWritable, Point, IntWritable, Point> {

    protected void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
        List<Float> sumCoordinates = new ArrayList<>();
        List<Point> points = new ArrayList<>();
        for (Point point : values) {
            points.add(new Point(point));
            for (int i = 0; i < point.getDimension(); i++) {
                if (i >= sumCoordinates.size()) {
                    sumCoordinates.add(point.getCoordinates().get(i));
                } else {
                    sumCoordinates.set(i, sumCoordinates.get(i) + point.getCoordinates().get(i));
                }
            }
        }
        for (int i = 0; i < sumCoordinates.size(); i++) {
            sumCoordinates.set(i, sumCoordinates.get(i) / (float) points.size());
        }
        Point newCentroid = new Point(sumCoordinates);

        KMeans.centroids.set(key.get(), newCentroid);
        for (Point point : points) {
            context.write(key, point);
        }

    }
}
