package geom;

import static visualizer.Utils.f;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;

public class P2 implements Comparable<P2> {
	public double x;
	public double y;
	public Set<LineString> edges;
	public double distance; // helper to return various tmp results

	public P2(double x, double y) {
		this.x = x; this.y = y;
		edges = new HashSet<>();
	}
	
	public P2 copy() {
		P2 p = new P2(x, y);
		// NOTE edges are not copied
		return p;
	}
	
	public void addEdge(LineString edge) {
		edges.add(edge);
	}
	
	public List<P2> getNeighbours() {
		List<P2> ret = new Vector<>();
		for (LineString e: edges) {
			ret.add(e.otherEnd(this));
		}
		return ret;
	}
	
	public double dist(P2 p) {
		return Math.hypot(x-p.x, y-p.y);
	}
	
	public static double dist(P2 p, P2 r) {
		return p.dist(r);
	}
	
	// Find projection point on a segment. Distance is returned in 'distance'.
	public P2 projectToLineSegment(P2 a, P2 b) {
		if (this.equals(a)) return a.copy();
		if (this.equals(b)) return b.copy();
		
		double dx = b.x - a.x;
		double dy = b.y - a.y;
		double len2 = dx*dx + dy*dy;
		double u = ((x - a.x) * dx + (y - a.y) * dy) / len2;
		P2 ret;
		if (u > 1) {
			ret = b.copy();
		}
		else if (u < 0) {
			ret = a.copy();
		}
		else {
			double px = a.x + u * dx;
			double py = a.y + u * dy;
			ret = new P2(px, py);
		}
		ret.distance = this.dist(ret);
		return ret;
	}
	
	@Override
	public String toString() {
		return "(" + f(x) + "," + f(y) + ")";
	}
	
	@Override
	public boolean equals(Object o) {
		// supporting 0.1 precision
		if (!(o instanceof P2)) return false;
		P2 p = (P2)o;
		double d2 = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
		return d2 < 1e-2;
	}
	
	@Override
	public int hashCode() {
		// 1354.3;1789.6 -> 1354317896
		long x1 = Math.round(10 * x);
		long y1 = Math.round(10 * y);
		return (int)(100000 * x1 + y1);
	}

	@Override
	public int compareTo(P2 o) {
		return this.hashCode() - o.hashCode();
	}
}