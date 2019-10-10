package geom;

import java.util.List;
import java.util.Vector;

import visualizer.Utils;

public class LineString {
	public List<P2> points;
	// end points
	public P2 p1;
	public P2 p2;

	public double length;
	public double minx, miny, maxx, maxy; // bounding rect
	
	public LineString() {
		points = new Vector<>();
	}
	
	public LineString(P2 p1, P2 p2) {
		this();
		this.p1 = p1;
		this.p2 = p2;
		points.add(p1);
		points.add(p2);
		updated();
	}
	
	/*
	 * Not fool-proof parsing, but handles most cases correctly.
	 * Should be called in try-catch.
	 */
	public static LineString fromText(String s) {
		LineString ret = new LineString();
		// "LINESTRING (250 250, 250 350, 1050 350)"
		s = s.replace("\"", "");
		s = s.toUpperCase().trim();
		if (!s.startsWith("LINESTRING")) return null;
		
		if (s.contains("EMPTY")) return ret;
		s = s.replace("LINESTRING", "");
		s = s.replace("(", "");
		s = s.replace(")", "");
		String[] parts = s.split(",");
		for (String coords: parts) {
			coords = coords.trim();
			String[] xy = coords.split(" ");
			double x = Double.parseDouble(xy[0]);
			double y = Double.parseDouble(xy[1]);
			P2 p = new P2(x, y);
			ret.points.add(p);
		}
		ret.updated();
		return ret;
	}
	
	public P2 otherEnd(P2 p) {
		if (p.equals(p1)) return p2;
		return p1;
	}
	
	public void updated() {
		p1 = points.get(0);
		p2 = points.get(points.size() - 1);
		double len = 0;
		minx = miny = Double.MAX_VALUE;
		maxx = maxy = -Double.MAX_VALUE;
		P2 prev = null;
		for (P2 p: points) {
			if (prev != null) len += prev.dist(p);
			minx = Math.min(minx, p.x);
			maxx = Math.max(maxx, p.x);
			miny = Math.min(miny, p.y);
			maxy = Math.max(maxy, p.y);
			prev = p;
		}
		length = len;
	}

	// Creates two linestrings, cut at distance from p1
	public LineString[] cut(double distance) {
		if (distance < 0) distance = 0;
		if (distance > length) distance = length;
		
		LineString e1 = new LineString();
		e1.points.add(p1);
		LineString e2 = new LineString();
		double total = 0;
		// p1....e......e.....m....e...p2
		for (int i = 1; i < points.size(); i++) {
			P2 prev = points.get(i-1);
			P2 next = points.get(i);
			double d = prev.dist(next);
			if (total + d < distance) {
				total += d;
				e1.points.add(next);
			}
			else if (total + d == distance) {
				total += d;
				
				e1.points.add(next);
				e1.p1 = p1;
				e1.p2 = next;
				e1.updated();
				
				for (int j = i; j < points.size(); j++) {
					e2.points.add(points.get(j));
				}
				e2.p1 = next;
				e2.p2 = p2;
				e2.updated();
				
				break;
			}
			else { // total + d > distance: project on segment
				double r = (distance - total) / d;
				double x = prev.x + r * (next.x - prev.x);
				double y = prev.y + r * (next.y - prev.y); 
				P2 mid = new P2(x, y);
				
				e1.points.add(mid);
				e1.p1 = p1;
				e1.p2 = mid;
				e1.updated();
				
				e2.points.add(mid);
				for (int j = i; j < points.size(); j++) {
					e2.points.add(points.get(j));
				}
				e2.p1 = mid;
				e2.p2 = p2;
				e2.updated();
				
				break;
			}
		}
		
		return new LineString[] {e1, e2};
	}
	
	@Override
	public String toString() {
		String ret = "";
		for (P2 p: points) ret += p + " ";
		ret += " Len: " + Utils.f(length);
		return ret;
	}
	
	// test only
	public static void main(String[] args) {
		LineString e = LineString.fromText("LINESTRING (1 1, 3 1, 5 1, 5 3, 5 4, 3 3, 3 1)");
		LineString[] es = e.cut(5); 
		System.out.println(es[0]);
		System.out.println(es[1]);
	}
}