package geom;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/*
 * A list of roads (as LineStrings) plus error checking logic
 */
public class RoadSet {
	public List<LineString> roads = new Vector<>();
	
	public String getError() {
		// There is an EMPTY one and more than one roads
		if (roads.size() > 1) {
			for (LineString road: roads) {
				if (road.points.size() == 0) {
					return "LINESTRING EMPTY should be alone in a road network";
				}
			}
		}				
		
		// stand-alone points
		for (LineString road: roads) {
			if (road.points.size() == 1) {
				return "Unconnected point at " + road.points.get(0);
			}
		}
		
		// empty sections
		for (LineString road: roads) {
			P2 p1 = null;
			for (P2 p2: road.points) {
				if (p1 != null && p2.dist(p1) < 0.01) return "Empty section in road at " + p2;
				p1 = p2;
			}
		}
		
		// repeated sections
		String err = "";
		Set<Long> hashes = new HashSet<>();
		for (LineString road: roads) {
			P2 p1 = null;
			for (P2 p2: road.points) {
				if (p1 != null) {
					int i1 = p1.hashCode();
					int i2 = p2.hashCode();
					long h1 = i1 * (long)1e10 + i2;
					long h2 = i2 * (long)1e10 + i1;
					if (hashes.contains(h1) || hashes.contains(h2)) {
						err += "Duplicate section: " + p1 + " - " + p2 + "; ";
					}
					hashes.add(h1);
					hashes.add(h2);
				}
				p1 = p2;
			}
		}
		if (!err.isEmpty()) return err;
		
		return null;
	}
	
	/*
	 * Ignore this method. A private tool to create a RoadSet from a Geogebra export file.
	 */
	public static RoadSet fromText(String line) {
		// (1,1)(10,1)
		RoadSet rs = new RoadSet();
		Pattern p = Pattern.compile("\\(([0-9]+)\\.,([0-9]+)\\.\\)\\(([0-9]+)\\.,([0-9]+)\\.\\)");
		Matcher m = p.matcher(line);
		while (m.find()) {
			double x1 = Double.parseDouble(m.group(1));
			double y1 = Double.parseDouble(m.group(2));
			double x2 = Double.parseDouble(m.group(3));
			double y2 = Double.parseDouble(m.group(4));
			LineString e = new LineString(new P2(x1, y1), new P2(x2, y2));
			rs.roads.add(e);
		}
		
		return rs;
	}
}