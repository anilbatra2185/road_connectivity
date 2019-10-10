package test;

import static visualizer.Utils.f;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import javax.imageio.ImageIO;

import geom.Graph;
import geom.LineString;
import geom.Metrics;
import geom.P2;
import geom.RoadSet;

/*
 * Standalone tool to run a set of tests found in a pair of truth/proposal files.
 * Outputs detailed log of the scoring and also images of the graphs.
 */
public class BatchMetricTest {
	private String truthFile = "../data/metrictest/truth1.csv";
	private String proposalFile = "../data/metrictest/proposal1.csv";
	private String singleIdToTest = null; // set to non-null to test a single image
	private boolean detailedLog = true;
	
	private List<String> idList;
	
	// for drawing
	double xmax, xmin, ymax, ymin, scale, ySign;
	private int M, w, yBase;
	
	private void run() {
		idList = new Vector<>();
		Map<String, RoadSet> idToTruthRS = load(truthFile);
		Map<String, RoadSet> idToProposalRS = load(proposalFile);
		Metrics.debug = detailedLog;
		Metrics.draw = true;
		int cnt = 0;
		
		for (String id: idList) {
			if (singleIdToTest != null && !singleIdToTest.equals(id)) continue;
			Graph g1 = Graph.fromRoads(idToTruthRS.get(id));
			Graph g2 = Graph.fromRoads(idToProposalRS.get(id));
			String cntString = "" + cnt++;
			while (cntString.length() < 3) cntString = "0" + cntString;
			String imageName = cntString + "-" + id;
			Metrics.imageName = imageName + "-details";
			log("\n============\n" + id);
			double[] scores = Metrics.score(g1, g2);
			log("G1->G2: " + f(scores[0]) + "\tG2->G1: " + f(scores[1]) + "\tAvg: " + f(scores[2]));
			draw(imageName, g1, g2, scores, true);
		}
	}
	
	// Copies most of Graph.draw functionality only to make it possible to draw two
	// graphs on the same image using the same scale
	private void draw(String name, Graph g1, Graph g2, double[] scores, boolean mirror) {
		w = 500;
		BufferedImage img = new BufferedImage(w, w + 4 * Graph.M, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2d = (Graphics2D) img.getGraphics();
		g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
		xmax = -Double.MAX_VALUE; 
		xmin = Double.MAX_VALUE;
		ymax = -Double.MAX_VALUE; 
		ymin = Double.MAX_VALUE;
		for (P2 p: g1.nodes) {
			for (LineString e: p.edges) {
				for (P2 p2: e.points) {
					xmax = Math.max(xmax, p2.x);
					xmin = Math.min(xmin, p2.x);
					ymax = Math.max(ymax, p2.y);
					ymin = Math.min(ymin, p2.y);
				}
			}
		}
		for (P2 p: g2.nodes) {
			for (LineString e: p.edges) {
				for (P2 p2: e.points) {
					xmax = Math.max(xmax, p2.x);
					xmin = Math.min(xmin, p2.x);
					ymax = Math.max(ymax, p2.y);
					ymin = Math.min(ymin, p2.y);
				}
			}
		}
		double range = Math.max(xmax - xmin, ymax - ymin);
		scale = (w - 2 * Graph.M) / range;
		ySign = mirror ? -1 : 1;
		yBase = mirror ? w + 2 * Graph.M: 2 * Graph.M;
		
		g2d.setStroke(new BasicStroke(3));
		draw(g2d, g1, new Color(0,255,255,180));
		draw(g2d, g2, new Color(255,255,0,150));
		
		g2d.setColor(Color.white);
		String res = "G1->G2: " + f(scores[0]) + ";    G2->G1: " + f(scores[1]) + ";    Avg: " + f(scores[2]);
		g2d.drawString(res, 20, 2 * Graph.M);
		try {
			ImageIO.write(img, "png", new File(name + ".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void draw(Graphics2D g2d, Graph g, Color color) {		
		g2d.setColor(color);
		for(P2 p: g.nodes) {
			for (LineString e: p.edges) {
				if (e.p1 != e.p2) {
					if (e.p2 == p) continue;
				}
				P2 prev = null;
				for (P2 p2: e.points) {
					if (prev != null) {
						int x1 = (int)(M + (p2.x - xmin) * scale);
						int y1 = (int)(yBase + (p2.y - ymin)* scale * ySign);
						int x2 = (int)(M + (prev.x - xmin) * scale);
						int y2 = (int)(yBase + (prev.y - ymin) * scale * ySign);
						g2d.drawLine(x1, y1, x2, y2);
					}
					prev = p2;
				}
			}
		}
	}
	
	private Map<String, RoadSet> load(String path) {
    	Map<String, RoadSet> ret = new HashMap<>();
		String line = null;
		int lineNo = 0;
		try {
			LineNumberReader lnr = new LineNumberReader(new FileReader(path));
			while (true) {
				line = lnr.readLine();
				lineNo++;
				if (line == null) break;
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#") || 
						line.toLowerCase().startsWith("imageid")) continue;
				// ImageId,LineString_Pix
				// AOI_5_Khartoum_img1,LINESTRING (250 250, 250 350, 1050 350)
				
				int pos = line.indexOf(",");
				String imageId = line.substring(0, pos);
				if (!idList.contains(imageId)) idList.add(imageId);
				RoadSet g = ret.get(imageId);
				if (g == null) {
					g = new RoadSet();
					ret.put(imageId, g);
				}
				String roadDef =  line.substring(pos + 1);
				LineString road = LineString.fromText(roadDef);
				if (road == null) {
					log("Error reading roads");
					log("Line #" + lineNo + ": " + line);
					System.exit(1);
				}
				g.roads.add(road);
			}
			lnr.close();
		} 
		catch (Exception e) {
			log("Error reading roads");
			log("Line #" + lineNo + ": " + line);
			e.printStackTrace();
			System.exit(1);
		}
		
		for (String id: ret.keySet()) {
			String err = ret.get(id).getError();
			if (err != null) {
				System.out.println("Error with : " + id + " : " + err);
				System.exit(1);
			}
		}
		return ret;
	}
	
	private void log(String s) {
		System.out.println(s);		
	}

	public static void main(String[] args) {
		new BatchMetricTest().run();
	}
}
