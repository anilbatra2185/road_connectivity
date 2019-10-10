package geom;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import javax.imageio.ImageIO;

import visualizer.Utils;

public class Graph {
	public Set<P2> nodes = new HashSet<>();
	/*
	 * This mapping allows multiple nodes to be injected at the same location. 
	 * All connectivity info and route calculations are done on the canonic nodes, but
	 * we bookkeep the aliases as well. 
	 */
	public Map<P2, P2> aliasToCanonicNode = new HashMap<>(); 
	public double minx, miny, maxx, maxy;
	
	// debug only
	public static final int M = 10; // drawing margin
	
	public Graph copy() { // deep copy of everything
		Graph g_ = new Graph();
		Map<P2, P2> nodeMap = new HashMap<>();
		for (P2 p: nodes) {
			P2 p_ = p.copy();
			nodeMap.put(p, p_);
			g_.nodes.add(p_);
		}
		for (P2 p: nodes) {
			for (LineString e: p.edges) {
				if (p != e.p1) continue; // enough to do this in one direction
				LineString e_ = new LineString();
				e_.p1 = nodeMap.get(p);
				e_.p2 = nodeMap.get(e.otherEnd(p));
				for (P2 n: e.points) {
					P2 n_ = nodeMap.get(n);
					if (n_ == null) {
						n_ = n.copy();
					}
					e_.points.add(n_);
				}
				e_.updated();
				e_.p1.addEdge(e_);
				e_.p2.addEdge(e_);
			}
		}
		// no need to copy aliasToCanonicNode
		g_.updated();
		return g_;
	}
	
	// Set up the graph from a set of roads
	public static Graph fromRoads(RoadSet rs) {
		Graph g = new Graph();
		
		// create nodes and connect them with 1-long segments
		Map<Integer, P2> hashToNode = new HashMap<>();
		for (LineString path: rs.roads) {
			P2 prevN = null;
			for (P2 p: path.points) {
				P2 n = new P2(p.x, p.y);
				int hash = n.hashCode();
				if (hashToNode.containsKey(hash)) {
					n = hashToNode.get(hash);
				}
				else {
					g.nodes.add(n);
					hashToNode.put(hash, n);
				}
				
				if (prevN != null) {
					LineString e = new LineString(prevN, n);
					prevN.addEdge(e);
					n.addEdge(e);					
				}
				prevN = n;
			}
		}
		
		// Simplify
		// Step 1: remove nodes with 2 neighbours, update paths to keep geometry
		Set<P2> simpleNodes = new HashSet<>();
		for (P2 p: g.nodes) if (p.edges.size() == 2) simpleNodes.add(p);
		Set<P2> toRemove = new HashSet<>();
		for (P2 p: g.nodes) {
			if (simpleNodes.contains(p)) continue;
			
			LineString[] oldEdges = p.edges.toArray(new LineString[0]);
			for (LineString edge: oldEdges) {
				P2 nextP = edge.otherEnd(p);
				
				if (toRemove.contains(nextP)) { // already processed from the other direction
					p.edges.remove(edge);
					continue;
				}
				
				// keep edge if other end is also non-simple
				if (!simpleNodes.contains(nextP)) continue;
				
				LineString newEdge = new LineString();
				newEdge.points.add(p);
				while (true) {
					newEdge.points.add(nextP);
					List<P2> neighbours = nextP.getNeighbours();
					toRemove.add(nextP);
					simpleNodes.remove(nextP);
					P2 prevP = newEdge.points.get(newEdge.points.size() - 2);
					nextP = null;
					for (P2 np: neighbours) {
						if (np != prevP) {
							nextP = np; break;
						}
					}
					if (nextP == null) { // shouldn't happen
						System.out.println("Err in simplify");
					}
					
					if (!simpleNodes.contains(nextP)) {
						newEdge.points.add(nextP);
						newEdge.p1 = p;
						newEdge.p2 = nextP;
						newEdge.updated();						 
						p.edges.remove(edge);
						nextP.edges.remove(edge);
						p.addEdge(newEdge);
						if (p != nextP) {
							nextP.addEdge(newEdge);
						}
						break;
					}
				}
			}
		} // for nodes
		for (P2 p: toRemove) {
			g.nodes.remove(p);
			p.edges.clear();
		}
		
		// Previously we processed closed loops, by keeping just one node of the cycle and 
		// keeping the rest only as path. To mimic networkx.simplify() functionality now we keep 
		// all 2-connected nodes in cycles.		
		/*
		while (!simpleNodes.isEmpty()) {
			// keep the one with smallest coords
			double miny = Double.MAX_VALUE;
			P2 startP = null;
			for (P2 p2: simpleNodes) {
				if (p2.y < miny) {
					startP = p2;
					miny = p2.y;
				}
				else if (startP != null && p2.y == startP.y) {
					if (p2.x < startP.x) {
						startP = p2;
					}
				}
			}
			simpleNodes.remove(startP);
			LineString edge = startP.edges.iterator().next();
			P2 nextP = edge.otherEnd(startP);
			startP.edges.clear();
			LineString newEdge = new LineString();
			newEdge.points.add(startP);
			while (true) {
				newEdge.points.add(nextP);
				List<P2> neighbours = nextP.getNeighbours();
				g.nodes.remove(nextP);
				nextP.edges.clear();
				simpleNodes.remove(nextP);
				P2 prevP = newEdge.points.get(newEdge.points.size() - 2);
				nextP = null;
				for (P2 np: neighbours) {
					if (np != prevP) {
						nextP = np; break;
					}
				}
				if (nextP == null) { // shouldn't happen
					System.out.println("Err in simplify");
				}
				
				if (nextP == startP) { // finished loop
					newEdge.points.add(nextP);
					newEdge.p1 = startP;
					newEdge.p2 = nextP;
					startP.addEdge(newEdge);
					newEdge.updated();						 
					break;
				}
			}
		} // there are simple nodes
		*/

		g.updated();
		return g;
	}
	
	public void updated() {
		minx = miny = Double.MAX_VALUE;
		maxx = maxy = -Double.MAX_VALUE;
		for (P2 p: nodes) {
			for (LineString e: p.edges) {
				minx = Math.min(minx, e.minx);
				maxx = Math.max(maxx, e.maxx);
				miny = Math.min(miny, e.miny);
				maxy = Math.max(maxy, e.maxy);
			}
		}
	}
	
	public void insertMidpoints(double pathDelta, double minCurvature) {
		Set<LineString> allEdges = new HashSet<>();
		Set<LineString> processedEdges = new HashSet<>();
		for (P2 p: nodes) {
			for (LineString e: p.edges) {
				allEdges.add(e);
			}
		}
		for (LineString e: allEdges) {
			if (processedEdges.contains(e)) continue; // done from the other end already
			
			double straight = new P2(e.minx, e.miny).dist(new P2(e.maxx, e.maxy));
			if (Math.abs(straight - e.length) / e.length < minCurvature) continue;
			if (e.length < 0.75 * pathDelta) continue;
			
			int n;
			double dist;
			if (e.length < pathDelta) {
				n = 1;
				dist = e.length / 2;
			}
			else {
				n = (int)(Math.floor(e.length / pathDelta));
				dist = e.length / (n+1);
			}			
			
			// p1....e....m...e......e...p2
			
			P2 p1 = e.p1;
			P2 p2 = e.p2;
			p1.edges.remove(e);
			p2.edges.remove(e);
			P2 startP = p1; // start node of the new edge to build
			LineString remainingEdge = e;
			for (int i = 0; i < n; i++) {
				LineString[] es = remainingEdge.cut(dist);
				LineString e1 = es[0];
				LineString e2 = es[1];
				P2 newP = e2.points.get(0);
				startP.addEdge(e1);
				newP.addEdge(e1);
				nodes.add(newP);
				startP = newP;
				remainingEdge = e2;
			}
			// last section
			startP.addEdge(remainingEdge);
			p2.addEdge(remainingEdge);
			
			processedEdges.add(e);
		}
	}
	
	public P2 injectPoint(P2 externalP, double maxDistance) {
		// try special case first: it exactly matches one of the old nodes
		for (P2 p: nodes) {
			if (p.equals(externalP)) {
				P2 newP = p.copy();
				// no need to change graph, only mark new node as alias of old
				aliasToCanonicNode.put(newP, p);
				return newP;
			}
		}
		
		double minDist = Double.MAX_VALUE;
		LineString bestEdge = null;
		P2 bestP1 = null; // start of line segment on bestEdge where newP is injected
		P2 bestNewP = null;
		P2 oldMatch = null; // non-null if matched to an existing node
		NODE_LOOP:
		for (P2 p: nodes) {
			for (LineString e: p.edges) {
				if (p != e.p1) continue; // enough to do this in one direction
				if (externalP.x < e.minx - minDist) continue;
				if (externalP.y < e.miny - minDist) continue;
				if (externalP.x > e.maxx + minDist) continue;
				if (externalP.y > e.maxy + minDist) continue;
				
				P2 p1 = null;
				for (P2 p2: e.points) {
					if (p1 != null) {
						P2 newP = externalP.projectToLineSegment(p1, p2);
						double d = newP.distance;
						if (d < minDist) {
							minDist = d;
							bestEdge = e;
							if (newP.equals(p2)) {
								bestP1 = p2;
							}
							else {
								bestP1 = p1;
							}
							bestNewP = newP;
							if (newP.equals(e.p1)) oldMatch = e.p1;
							else if (newP.equals(e.p2)) oldMatch = e.p2;
							else oldMatch = null;
						}
						if (d == 0) { // can't be better, stop
							break NODE_LOOP;
						}
					}
					p1 = p2;
				}
			}
		} // for nodes and edges
		
		if (minDist > maxDistance) {
			return null;
		}
		
		if (oldMatch != null) { // no need to change graph, only mark new node as alias of old
			aliasToCanonicNode.put(bestNewP, oldMatch);
		}
		else { // add newP, split edge
			
			// p1....p....newP...p......p...p2
			LineString e1 = new LineString();
			LineString e2 = new LineString();
			for (int i = 0; i < bestEdge.points.size(); i++) {
				P2 p = bestEdge.points.get(i);
				if (p.equals(bestP1)) {
					if (!bestNewP.equals(bestP1)) {
						e1.points.add(bestP1);
					}
					e1.points.add(bestNewP);
					e2.points.add(bestNewP);
					for (int j = i+1; j < bestEdge.points.size(); j++) {
						e2.points.add(bestEdge.points.get(j));
					}
					break;
				}
				e1.points.add(p);
			}
			e1.updated();
			e2.updated();
			bestEdge.p1.edges.remove(bestEdge);
			bestEdge.p2.edges.remove(bestEdge);
			nodes.add(bestNewP);
			bestNewP.addEdge(e1);
			bestNewP.addEdge(e2);
			bestEdge.p1.addEdge(e1);
			bestEdge.p2.addEdge(e2);
		}
		
		return bestNewP;
	}
	
	// A copy of a node to live in the priority queue during Dijkstra
	private class PQNode implements Comparable<PQNode> {
		public P2 p;
		public double distance;
		
		public PQNode(P2 p, double d) {
			this.p = p;
			distance = d;
		}
		
		@Override
		public int compareTo(PQNode other) {
			return Double.compare(this.distance, other.distance);
		}
		
		@Override
		public String toString() {
			return p.toString() + ": " + distance;
		}
	}
	
	// A priority queue based implementation of Dijkstra algorithm
	public void shortestPathsFromNode(P2 startP) {
		Set<P2> seen = new HashSet<>();
		PriorityQueue<PQNode> q = new PriorityQueue<>();
		for (P2 p: nodes) {
			double distance = p == startP ? 0 : Double.MAX_VALUE;
			p.distance = distance;
			q.add(new PQNode(p, distance));
		}
		while (!q.isEmpty()) {
			PQNode node = q.poll();
			if (seen.contains(node.p)) continue;
			seen.add(node.p);
			
			for (LineString e: node.p.edges) {
				P2 p2 = e.otherEnd(node.p);
				if (seen.contains(p2)) continue;
				double dist = node.distance + e.length;
				if (dist < p2.distance) {
					p2.distance = dist;
					q.add(new PQNode(p2, dist));
				}
			}
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		P2[] pArr = nodes.toArray(new P2[0]);
		Arrays.sort(pArr);
		for (P2 p: pArr) {
			sb.append(p).append("\n");
			for (LineString e: p.edges) {
				sb.append("  ").append(e).append("\n");
			}
		}
		return sb.toString();
	}
	
	public void save(String name, boolean mirror) throws IOException { // debug only
		int w = 500;
		BufferedImage img = new BufferedImage(w, w, BufferedImage.TYPE_INT_ARGB);
		draw(img, Color.white, -1, w, 0, 0, mirror);
		ImageIO.write(img, "png", new File(name));		
	}
	
	public void draw(BufferedImage img, Color c, double range, int w, int x0, int y0, boolean mirror) { // debug only
		Graphics2D g2 = (Graphics2D) img.getGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
		int r = c.getRed();
		int g = c.getGreen();
		int b = c.getBlue();
		int yBase = mirror ? w - M : M;
		if (range <= 0) range = Math.max(maxx - minx, maxy - miny);
		double scale = (w - 2*M) / range;
		double ySign = mirror ? -1 : 1;
		g2.setColor(new Color(r,g,b,180));
		for (P2 p: nodes) {
			for (LineString e: p.edges) {
				if (e.p1 != e.p2) {
					if (e.p2 == p) continue;
				}
				P2 prev = null;
				for (P2 p2: e.points) {
					if (prev != null) {
						int x1 = (int)(x0 + M + (p2.x - minx) * scale);
						int y1 = (int)(y0 + yBase + (p2.y - miny)* scale * ySign);
						int x2 = (int)(x0 + M + (prev.x - minx) * scale);
						int y2 = (int)(y0 + yBase + (prev.y - miny) * scale * ySign);
						g2.drawLine(x1, y1, x2, y2);
					}
					prev = p2;
				}
			}
		}
		
		Color c1 = new Color(r,g,b,180);
		Color c2 = new Color(255,255,0,180);
		Color c3 = new Color(255,0,255,180);
		
		for (P2 p: nodes) {
			int n = p.edges.size();
			if (n == 1) g2.setColor(c1);
			else if (n > 2) g2.setColor(c3);
			else { // two edges
				g2.setColor(c2);
				// unless there are self loops
				for (LineString e: p.edges) {
					if (e.p1 == e.p2) g2.setColor(c3);
				}
			}
			int x = (int)(x0 + M + (p.x - minx) * scale);
			int y = (int)(y0 + yBase + (p.y - miny) * scale * ySign);
			g2.fillOval(x-3, y-3, 6, 6);
		}
	}
	
	// test only
	public static void main(String[] args) throws IOException {
		RoadSet rs = new RoadSet();
		rs.roads.add(LineString.fromText("LINESTRING (1 1, 3 1, 5 1, 5 3, 5 4, 3 3, 3 1)"));
		rs.roads.add(LineString.fromText("LINESTRING (6 1, 8 1, 8 3, 6 3, 6 1)"));
		rs.roads.add(LineString.fromText("LINESTRING (8 4, 8 6, 6 6, 6 4, 8 4, 10 4, 10 6, 8 6)"));
		rs.roads.add(LineString.fromText("LINESTRING (5 4, 6 4)"));
		
		String err = rs.getError();
		if (err != null) {
			System.out.println(err);
			System.exit(0);
		}
		Graph g = fromRoads(rs);
		
		for (P2 start: g.nodes) {
			g.shortestPathsFromNode(start);
			for (P2 p2: g.nodes) {
				if (!start.equals(p2) && p2.distance < Double.MAX_VALUE) {
					System.out.println(start + " -> " + p2 + " : " + Utils.f(p2.distance));
				}
			}
		}
		
		System.out.println(g);
		g.save("out.png", true);
	}
}
