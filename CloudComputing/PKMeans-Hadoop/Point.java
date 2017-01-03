import java.io.*; // DataInput/DataOuput
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import org.apache.hadoop.io.*; // Writable

/**
 * A Point is some ordered list of floats.
 * 
 * A Point implements WritableComparable so that Hadoop can serialize
 * and send Point objects across machines.
 *
 * NOTE: This implementation is NOT complete.  As mentioned above, you need
 * to implement WritableComparable at minimum.  Modify this class as you see fit.
 */
public class Point implements WritableComparable {
    private List<Float> coordinates = new ArrayList<>();

    public Point() {
        coordinates = new ArrayList<>();
    }

    /**
     * Construct a Point with the given dimensions [dim]. The coordinates should all be 0.
     * For example:
     * Constructing a Point(2) should create a point (x_0 = 0, x_1 = 0)
     */
    public Point(int dim) {
        for (int i = 0; i < dim; i++) {
            coordinates.add((float) 0);
        }
    }

    /**
     * Construct a point from a properly formatted string (i.e. line from a test file)
     * @param str A string with coordinates that are space-delimited.
     * For example: 
     * Given the formatted string str="1 3 4 5"
     * Produce a Point {x_0 = 1, x_1 = 3, x_2 = 4, x_3 = 5}
     */
    public Point(String str) {
        String[] coordinates = str.split("\\s+");
        for (String coordinate : coordinates) {
            this.coordinates.add(Float.parseFloat(coordinate));
        }
    }

    /**
     * Copy constructor
     */
    public Point(Point other) {
        coordinates = new ArrayList<>(other.getCoordinates());
    }

    /**
     * Construct a point from a List.
     */
    public Point(List<Float> coordinates) {
        this.coordinates = new ArrayList<>(coordinates);
    }

    /**
     *
     * @return A List of coordinates of the point.
     */
    public List<Float> getCoordinates() {
        return coordinates;
    }

    /**
     * @return The dimension of the point.  For example, the point [x=0, y=1] has
     * a dimension of 2.
     */
    public int getDimension() {
        return coordinates.size();
    }

    /**
     * Converts a point to a string.  Note that this must be formatted EXACTLY
     * for the autograder to be able to read your answer.
     * Example:
     * Given a point with coordinates {x=1, y=1, z=3}
     * Return the string "1 1 3"
     */
    public String toString() {
        String ret = "";
        if (coordinates.isEmpty()) {
            return ret;
        }
        for (Float coordinate : coordinates) {
            ret += Float.toString(coordinate) + " ";
        }
        return ret.substring(0, ret.length() - 1);
    }

    /**
     * One of the WritableComparable methods you need to implement.
     * See the Hadoop documentation for more details.
     * You should order the points "lexicographically" in the order of the coordinates.
     * Comparing two points of different dimensions results in undefined behavior.
     */
    public int compareTo(Object o) {
        assert o instanceof Point;

        Point other = (Point) o;
        assert getDimension() == other.getDimension();

        for (int i = 0; i < getDimension(); i++) {
            if (coordinates.get(i) < other.getCoordinates().get(i)) {
                return -1;
            } else if (coordinates.get(i) > other.getCoordinates().get(i)) {
                return 1;
            }
        }
        return 0;
    }

    /**
     * @return The L2 distance between two points.
     */
    public static final float distance(Point x, Point y) {
        assert x.getDimension() == y.getDimension();

        double squareSum = 0;
        for (int i = 0; i < x.getDimension(); i++) {
            squareSum += Math.pow(x.getCoordinates().get(i) - y.getCoordinates().get(i), 2);
        }
        return (float) Math.sqrt(squareSum);
    }

    /**
     * @return A new point equal to [x]+[y]
     */
    public static final Point addPoints(Point x, Point y) {
        assert x.getDimension() == y.getDimension();

        Point ret = new Point(1);
        ret.coordinates.remove(0);
        for (int i = 0; i < x.getDimension(); i++) {
            ret.coordinates.add(x.getCoordinates().get(i) + y.getCoordinates().get(i));
        }
        return ret;
    }

    /**
     * @return A new point equal to [c][x]
     */
    public static final Point multiplyScalar(Point x, float c) {
        Point ret = new Point(x);
        for (int i = 0; i < ret.getDimension(); i++) {
            ret.getCoordinates().set(i, ret.getCoordinates().get(i) * c);
        }
        return ret;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        for (float coordinate : coordinates) {
            dataOutput.writeFloat(coordinate);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        coordinates = new ArrayList<>();
        while (true) {
            try {
                coordinates.add(dataInput.readFloat());
            } catch (EOFException e) {
                break;
            }
        }
    }
}
