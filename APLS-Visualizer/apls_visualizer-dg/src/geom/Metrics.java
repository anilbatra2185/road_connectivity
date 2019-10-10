package geom;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import static visualizer.Utils.*;

/*
 * Implements the scoring logic. Optionally can draw the graphs 
 * and output detailed log.
 */
public class Metrics {
	private static final double PATH_DELTA = 50 / 0.31;
	private static final double MIN_CURVATURE = -1; 
	public static final double MAX_SNAP_DISTANCE = 4 / 0.31;
	
	// variables for debug only
	public static boolean debug = true;
	public static boolean draw = true;
	private static BufferedImage img;
	private static Graphics2D g2d;
	public static String imageName = "metrics";
	private static int w = 500;
	private static int callCnt = 0;
	private static double range;
	
	/*
	 * Returns a double[3] with 3 scores: G1->G2, G2->G1, harmonic mean.
	 * All 3 numbers are in [0..1], where 0 is bad, 1 is perfect.
	 * The 'debug' and 'draw' switches should be set externally before calling this.
	 */
	public static double[] score(Graph g1, Graph g2) {
		if (g1.nodes.isEmpty() && g2.nodes.isEmpty()) {
			return new double[]{1, 1, 1};
		}
		if (g1.nodes.isEmpty() || g2.nodes.isEmpty()) {
			return new double[]{0, 0, 0};
		}
		
		g1.insertMidpoints(PATH_DELTA, MIN_CURVATURE);
		g2.insertMidpoints(PATH_DELTA, MIN_CURVATURE);
		
		if (draw) {
			img = new BufferedImage(w, w + 4 * Graph.M, BufferedImage.TYPE_INT_ARGB);
			g2d = (Graphics2D) img.getGraphics();
			g2d.setColor(Color.black);
			g2d.fillRect(0, 0, img.getWidth(), img.getHeight());
			double range1 = Math.max(g1.maxx - g1.minx, g1.maxy - g1.miny);
			double range2 = Math.max(g2.maxx - g2.minx, g2.maxy - g2.miny);
			range = Math.max(range1, range2);
			g1.draw(img, Color.white, range, w/2, 0, 2 * Graph.M, true);
			g2.draw(img, Color.white, range, w/2, w/2, 2 * Graph.M, true);
			g2d.setColor(Color.white);
			g2d.drawString("G1", 20, 2 * Graph.M);
			g2d.drawString("G2", w/2 + 20, 2 * Graph.M);
			callCnt = 0;
		}
		
		double s1 = scoreOneWay(g1, g2);
		double s2 = scoreOneWay(g2, g1);
		double s = 0;
		if (s1 + s2 > 0) {
			s = 2 * s1 * s2 / (s1 + s2);
		}
		log("\nS1: =" + s1);
		log("\nS2: =" + s2);
		log("\nS: =" + s);
		log("\n");

		if (draw) {
			try {
				ImageIO.write(img, "png", new File(imageName + ".png"));
			} 
			catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		return new double[]{s1, s2, s};
	}

	private static double scoreOneWay(Graph g1, Graph g2) {
		// inject points of g1 into g2
		Map<P2, P2> nodeMap = new HashMap<>();
		Graph g2_ = g2.copy();
		for (P2 p1: g1.nodes) {
			P2 p2 = g2_.injectPoint(p1, MAX_SNAP_DISTANCE);
			if (p2 != null) {
				nodeMap.put(p1, p2);
			}
		}
		
		if (debug) {
			log("\nG1:\n" + g1);
			log("\nG2':\n" + g2_);
			log("");
		}
		double totalDiff = 0; // sum of error
		int routeCnt = 0; // number of compared routes
		for (P2 start1: g1.nodes) {
			g1.shortestPathsFromNode(start1);
			
			P2 start2 = nodeMap.get(start1);
			
			if (start2 == null) {
				// CASE 1
		        // if the start node is missing from proposal, use maximum diff for 
		        // all possible routes from the start node
		        int missCnt = 0;
				for (P2 p: g1.nodes) {
					if (p != start1 && p.distance < Double.MAX_VALUE) {
						missCnt++;
					}
				}
				totalDiff += missCnt;
				routeCnt += missCnt;
				if (debug) log("  " + start1 + ": no match for start in G2_, missed " + missCnt);
			}
			else {
				if (debug) log("  " + start1 + " ->");
				// found matching node in g2, compare routes. Use canonic nodes in g2_!
				if (g2_.aliasToCanonicNode.containsKey(start2)) {
					start2 = g2_.aliasToCanonicNode.get(start2);
				}
				g2_.shortestPathsFromNode(start2);
				for (P2 end1: g1.nodes) {
					if (end1 == start1) continue;
					
					P2 end2 = nodeMap.get(end1);
					if (end2 != null && g2_.aliasToCanonicNode.containsKey(end2)) {
						end2 = g2_.aliasToCanonicNode.get(end2);
					}
					
					double d1 = end1.distance;
					if (d1 < Double.MAX_VALUE) {
						// there is route between start1 and end1
						routeCnt++;
						if (end2 == null) {
							// CASE 3: no such node in g2, max penalty
							totalDiff++;
							if (debug) log("    " + end1 + ": no match for end in G2_");
							continue;
						}
						double d2 = end2.distance;
						if (d2 == Double.MAX_VALUE) {
							// CASE 3b: no route in g2, max penalty
							totalDiff++;
							if (debug) log("    " + end1 + ": no route in G2_");
						}
						else {
							// CASE 2: both paths exist, compare them
							double diff = routeDiff(d1, d2); 
							totalDiff += diff;
							if (debug) log("    " + end1 + ": " + f(d1) + " / " + f(d2) + " => " + f(diff));
						}
					} 
				} // for end1
			} // start2 not null		
		} // for start1
		
		if (routeCnt > 0) totalDiff /= routeCnt;
		if (debug) log("\nAverage diff: " + f(totalDiff));
		
		double score = 1 - totalDiff;
		if (draw) {
			g2_.draw(img, Color.white, range, w/2, w/2 * callCnt, w/2 + 2 * Graph.M, true);
			g2d.setColor(Color.white);
			String msg = callCnt == 0 ? "G2' (G1->G2): " : "G1' (G2->G1): ";
			msg += f(score);
			g2d.drawString(msg, 20 + w/2 * callCnt, w + 4*Graph.M-2);
			callCnt++;
		}
		
		return score;
	}

	private static double routeDiff(double d1, double d2) {
		if (d1 == 0 && d2 == 0) return 0;
		if (d1 == 0 || d2 == 0) return 1;
		return Math.min(1, Math.abs(d1 - d2) / d1);
	}
	
	private static void log(String s) {
		System.out.println(s);
	}

	// test only
	public static void main(String[] args) {
		RoadSet rs1 = new RoadSet();
		rs1.roads.add(LineString.fromText("LINESTRING (1 1, 3 1, 5 1, 5 3, 5 4, 3 4, 3 1)"));
		Graph g1 = Graph.fromRoads(rs1);
		
		RoadSet rs2 = new RoadSet();
		rs2.roads.add(LineString.fromText("LINESTRING (1 1, 3 1, 5 1, 5 3, 5 4, 3 4)"));
		rs2.roads.add(LineString.fromText("LINESTRING (1 2, 1 3)"));
		Graph g2 = Graph.fromRoads(rs2);
		
		double[] scores = score(g1, g2);
		System.out.println(f(scores[0]) + ", " + f(scores[1]) + " : " + f(scores[2]));

	}
}
