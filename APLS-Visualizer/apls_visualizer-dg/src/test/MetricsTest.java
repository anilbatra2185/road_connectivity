package test;

import geom.Graph;
import geom.Metrics;
import geom.RoadSet;
import visualizer.Utils;

public class MetricsTest {

	// test only
	public static void main(String[] args) {
		RoadSet rs1 = RoadSet.fromText("(10.,10.)(20.,10.)(20.,10.)(20.,20.)(20.,20.)(40.,20.)(40.,20.)(40.,15.)(40.,15.)(40.,10.)(40.,10.)(25.,10.)(25.,10.)(25.,20.)(25.,15.)(40.,15.)(25.,12.)(40.,12.)(32.,12.)(32.,20.)(20.,22.)(47.,22.)(55.,22.)(55.,20.)(55.,20.)(40.,20.)(47.,24.)(47.,29.)(47.,24.)(27.,24.)(27.,24.)(27.,22.)");
		RoadSet rs2 = RoadSet.fromText("(10.,10.)(20.,10.)(20.,10.)(20.,20.)(20.,20.)(40.,20.)(40.,20.)(40.,15.)(40.,15.)(40.,10.)(40.,10.)(25.,10.)(25.,10.)(25.,20.)(25.,15.)(40.,15.)(25.,12.)(40.,12.)(32.,12.)(32.,20.)(20.,22.)(47.,22.)(55.,22.)(55.,20.)(55.,20.)(40.,20.)(47.,24.)(47.,29.)(47.,24.)(27.,24.)(27.,24.)(27.,22.)");
		String err;
		err = rs1.getError();
		if (err != null) {
			System.out.println(err);
			System.exit(0);
		}
		err = rs2.getError();
		if (err != null) {
			System.out.println(err);
			System.exit(0);
		}
		
		Graph g1 = Graph.fromRoads(rs1);
		Graph g2 = Graph.fromRoads(rs2);
		double[] scores = Metrics.score(g1, g2);
		System.out.println(Utils.f(scores[0]) + ", " + Utils.f(scores[1]) + " : " + Utils.f(scores[2]));

	}
}
