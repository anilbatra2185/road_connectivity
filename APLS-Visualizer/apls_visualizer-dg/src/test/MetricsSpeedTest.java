package test;

import static visualizer.Utils.f6;

import geom.Graph;
import geom.LineString;
import geom.Metrics;
import geom.P2;
import geom.RoadSet;

public class MetricsSpeedTest {
	
	public static void main(String[] args) {
		double d = 5 / 0.31;
		for (int n = 5; n < 40; n++) {
			double[] poss = new double[n];
			for (int i = 0; i < n; i++) poss[i] = (2*i+1) * d;
			
			RoadSet rs1 = new RoadSet();
			for (int i = 0; i < n; i++) {
				LineString e = new LineString();
				for (int j = 0; j < n; j++) {
					P2 p = new P2(poss[i], poss[j]);
					e.points.add(p);
				}
				rs1.roads.add(e);
			}
			for (int i = 0; i < n; i++) {
				LineString e = new LineString();
				for (int j = 0; j < n; j++) {
					P2 p = new P2(poss[j], poss[i]);
					e.points.add(p);
				}
				rs1.roads.add(e);
			}
			Graph g1 = Graph.fromRoads(rs1);
			
			RoadSet rs2 = new RoadSet();
			rs2.roads.add(LineString.fromText("LINESTRING (10 10, 30 10, 50 10, 50 30, 30 40, 30 10)"));
			Graph g2 = Graph.fromRoads(rs2);
			
			long startTime = System.currentTimeMillis();
			Metrics.debug = false;
			Metrics.draw = false;
			double[] scores = Metrics.score(g1, g2);
			long time = System.currentTimeMillis() - startTime;		
			System.out.println(n + ": " + f6(scores[0]) + ", " + f6(scores[1]) + " : " + f6(scores[2]) + " : " + time);
		}
	}
}
