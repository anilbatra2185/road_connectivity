/*
 * Road Detector Visualizer and Offline Tester
 * by walrus71
 * 
 * Version history:
 * ================
 * 1.0 (2017.11.28)
 *      - Version at contest launch
 * 0.3 (2017.11.23)
 *      - Snap distance changed
 *      - Small UI improvements (nicer buffer drawing)
 * 0.2 (2017.11.22)
 *      - Added scoring
 *      - Many UI improvements (junction switch, xy info, etc)
 * 0.1 (2017.11.06)
 *      - Initital version
 */
package visualizer;

import static visualizer.Utils.f;
import static visualizer.Utils.f6;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.GeneralPath;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import javax.imageio.ImageIO;
import java.io.IOException;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import geom.Graph;
import geom.LineString;
import geom.Metrics;
import geom.P2;
import geom.RoadSet;

public class RoadVisualizer implements ActionListener, ItemListener, MouseListener {
	
	private boolean hasGui = true;
	private boolean isDebug = false;
	private String[] imageDirs;
	private String[] imageIds;
	private String saveImageDir;
	private Map<String, String> idToDir; // which data folder this image comes from
	private String currentImageId;
	private String currentScore;
	private String[] truthPaths;
	private String solutionPath;
	private Map<String, RoadSet> idToTruthRoadSet;
	private Map<String, RoadSet> idToSolutionRoadSet;
	private Set<P2> currentTrueJunctions;
	private Set<P2> currentSolutionJunctions;
	
	private double scale; // data size / screen size (for 3-band images)
	private double x0 = 0, y0 = 0; // x0, y0: TopLeft corner of data is shown here (in screen space, applies to all views)
	private double ratio38; // scaling factor between 3-band and 8-band images
	
	private JFrame frame;
	private JPanel viewPanel, controlsPanel;
	private JCheckBox showTruthCb, showTruthBufferCb, showSolutionCb, showJunctionsCb;
	private JLabel infoLabel;
	private JButton saveCurrentImageGraph;
	private JComboBox<String> viewSelectorComboBox;
	private JComboBox<String> imageSelectorComboBox;
	private JTextArea logArea;
	private MapView mapView;
	private Font font = new Font("SansSerif", Font.BOLD, 14);
	
	private String bandTripletPath;
	private List<BandTriplet> bandTriplets;
	private BandTriplet currentBandTriplet;
	
	private Color truthColor    = new Color(100, 255, 255, 180);
	private Color truthBufferColor;
	private Color solutionColor = new Color(255, 255,   0, 150);
	private Color roadBackgroundColor = new Color(0, 0, 0, 100);
	public Color junctionColor = Color.white;
	private Stroke strokeThin = new BasicStroke(4f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND);
	private Stroke strokeWide = new BasicStroke(6f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND);
	
	private void run() {
		idToSolutionRoadSet = load(new String[] {solutionPath}, false);
		idToTruthRoadSet = load(truthPaths, true);
		imageIds = collectImageIds();
		String[] cities = collectCities();
		
		if (idToSolutionRoadSet.isEmpty() || idToTruthRoadSet.isEmpty()) {
			// can't score, just output ids
			log("Nothing to score");
			for (String id: imageIds) {
				log(id);
			}
		}
		else {
			setInfo("Scoring");
			Metrics.debug = isDebug;
			Metrics.draw = isDebug;
			Map<String, Results> cityToScore = new HashMap<>();
			for (String c: cities) cityToScore.put(c, new Results());
			
			String detailsMarker = "Details:";
			log(detailsMarker);
			for (String id: imageIds) {
				Results result = score(id);
				if (result != null) {
					log(id + "\n"
						+ "  T->P: " + f(result.score1) + "\tP->T: " + f(result.score2) + "\tScore: " + f(result.score));
					String city = idToCity(id);
					Results m = cityToScore.get(city);
					m.score += result.score;
					m.score1 += result.score1;
					m.score2 += result.score2;
					m.cnt++;					
				}
				// else {
				// 	log(id + "\n  - not scored");
				// }
			}
			
			double sum = 0;
			int cityCnt = 0;
			String result = "";
			for (String c: cities) {
				Results m = cityToScore.get(c);
				double score = 0;
				double score1 = 0;
				double score2 = 0;
				if (m.cnt != 0) {
					m.score /= m.cnt;
					m.score1 /= m.cnt;
					m.score2 /= m.cnt;
					score = m.score;
					score1 = m.score1;
					score2 = m.score2;
				}
				sum += score;
				cityCnt++;
				result += "\n" + c + ":\n"
						+ "  score       : " + m.score + ":\n"
						+ "  T->P        : " + m.score1 + ":\n"
						+ "  P->T        : " + m.score2;
			}
			
			if (sum > 0) {
				double s = sum / cityCnt;
				result = "\nOverall score : " + f6(s) + "\n" + result;
			}
			else {
				result = "\nOverall score : 0\n\n";
			}
			
			if (hasGui) { // display final result at the top
				String allText = logArea.getText();
				int pos = allText.indexOf(detailsMarker);
				String s1 = allText.substring(0, pos);
				String s2 = allText.substring(pos);
				allText = s1 + result + "\n\n" + s2;
				logArea.setText(allText);
				logArea.setCaretPosition(0);
				System.out.println(result);
			}
			else {
				log(result);
			}
		} // anything to score
		
		// the rest is for UI, not needed for scoring
		if (!hasGui) return;
		
		DefaultComboBoxModel<String> cbm = new DefaultComboBoxModel<>(imageIds);
		imageSelectorComboBox.setModel(cbm);
		imageSelectorComboBox.setSelectedIndex(0);
		imageSelectorComboBox.addItemListener(this);
		
		currentImageId = imageIds[0];
		loadMap();
		repaintMap();
	}

	private Results score(String id) {
		RoadSet rs1 = idToTruthRoadSet.get(id);
		if (rs1 == null) return null;
		RoadSet rs2 = idToSolutionRoadSet.get(id);
		if (rs2 == null) return null;
		Graph g1 = Graph.fromRoads(rs1);
		Graph g2 = Graph.fromRoads(rs2);
		double[] scores = Metrics.score(g1, g2);
		Results ret = new Results();
		ret.score1 = scores[0];
		ret.score2 = scores[1];
		ret.score = scores[2];		
		return ret;
	}
    
    private Map<String, RoadSet> load(String[] paths, boolean truth) {
    	String what = truth ? "truth file" : "your solution";
		if (paths == null || paths.length == 0 || paths[0] == null) {
			log("  Path for " + what + " not set, nothing loaded.");
			return new HashMap<>();
		}
		log(" - Reading " + what + " from " + Arrays.toString(paths) + " ...");
		setInfo("Reading " + what);
		
		Map<String, RoadSet> ret = new HashMap<>();
		for (String path: paths) {
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
					RoadSet rs = ret.get(imageId);
					if (rs == null) {
						rs = new RoadSet();
						ret.put(imageId, rs);
					}
					String roadDef =  line.substring(pos + 1);
					LineString road = LineString.fromText(roadDef);
					if (road == null) {
						log("Error reading roads");
						log("Line #" + lineNo + ": " + line);
						System.exit(1);
					}
					rs.roads.add(road);
				}
				lnr.close();
			} 
			catch (Exception e) {
				log("Error reading roads");
				log("Line #" + lineNo + ": " + line);
				e.printStackTrace();
				System.exit(1);
			}
		} // for paths
		
		for (String id: ret.keySet()) {
			RoadSet rs = ret.get(id);
			String err = rs.getError();
			if (err != null) {
				log("Error in road network for id " + id + " : " + err);
				System.exit(1);
			}
		}
		
		return ret;
	}
    
	private String[] collectImageIds() {
		Set<String> ids = new HashSet<>();
		idToDir = new HashMap<>();
		if (imageDirs != null) {
			for (String dirName: imageDirs) {
				File dir = new File(dirName, "images"); // TODO - Change
				if (!dir.exists() || !dir.isDirectory()) {
					log("Can't find image folder " + dir.getPath());
					continue;
				}
				// TODO - Change
				for (String s: dir.list()) {
					if (!s.endsWith(".jpg")) continue;
					s = s.replace("_sat.jpg", "");
					// s = s.replace("PAN_", "");
					ids.add(s);
					idToDir.put(s, dirName);
				}
			}
		}
		ids.addAll(idToTruthRoadSet.keySet());
		ids.addAll(idToSolutionRoadSet.keySet());
		
		String[] ret = ids.toArray(new String[0]);
		Arrays.sort(ret);
		return ret;
	}
	
	private String[] collectCities() {
		Set<String> cities = new HashSet<>();
		for (String id: imageIds) {
			String c = idToCity(id);
			cities.add(c);
		}		
		String[] arr = cities.toArray(new String[0]);
		Arrays.sort(arr);
		return arr;
	}

	private String idToCity(String id) {
		// AOI_5_Khartoum_img1
		// String[] parts = id.split("_");
		// return parts[0] + "_" + parts[1] + "_" + parts[2];
		return "DG";
	}

	private class Results {
		public double score1 = 0;
		public double score2 = 0;
		public double score = 0;
		public int cnt = 0;
	}
	
	private class MapData {
		public int W;
		public int H;
		public int[][] pixels;
		public MapData(int w, int h) {
			W = w; H = h;
			pixels = new int[W][H];
		}
	}

	/**************************************************************************************************
	 * 
	 *              THINGS BELOW THIS ARE UI-RELATED, NOT NEEDED FOR SCORING
	 * 
	 **************************************************************************************************/
	
	public void setupGUI(int W) {
		if (!hasGui) return;
		
		loadBandTriplets();
		
		frame = new JFrame("Road Detector Visualizer");
		int H = W * 2 / 3;
		frame.setSize(W, H);
		frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		Container cp = frame.getContentPane();
		cp.setLayout(new GridBagLayout());
		
		GridBagConstraints c = new GridBagConstraints();
		
		c.fill = GridBagConstraints.BOTH;
		c.gridx = 0;
		c.gridy = 0;
		c.weightx = 2;
		c.weighty = 1;
		viewPanel = new JPanel();
		viewPanel.setPreferredSize(new Dimension(W, H));
		cp.add(viewPanel, c);
		
		c.fill = GridBagConstraints.BOTH;
		c.gridx = 1;
		c.gridy = 0;
		c.weightx = 1;
		controlsPanel = new JPanel();
		cp.add(controlsPanel, c);

		viewPanel.setLayout(new BorderLayout());
		mapView = new MapView();
		viewPanel.add(mapView, BorderLayout.CENTER);
		
		controlsPanel.setLayout(new GridBagLayout());
		GridBagConstraints c2 = new GridBagConstraints();
		
		int gridY = 0;
		
		showTruthCb = new JCheckBox("Show truth roads");
		showTruthCb.setSelected(true);
		showTruthCb.addActionListener(this);
		c2.fill = GridBagConstraints.BOTH;
		c2.gridx = 0;
		c2.gridy = gridY++;
		c2.weightx = 1;
		controlsPanel.add(showTruthCb, c2);
		
		showTruthBufferCb = new JCheckBox(" ... with buffer");
		showTruthBufferCb.setSelected(false);
		showTruthBufferCb.addActionListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(showTruthBufferCb, c2);
		
		showSolutionCb = new JCheckBox("Show solution roads");
		showSolutionCb.setSelected(true);
		showSolutionCb.addActionListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(showSolutionCb, c2);
		
		showJunctionsCb = new JCheckBox("Highlight junctions");
		showJunctionsCb.setSelected(false);
		showJunctionsCb.addActionListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(showJunctionsCb, c2);

		saveCurrentImageGraph = new JButton("Save Current Graph");
		saveCurrentImageGraph.addActionListener(new ActionListener(){
	         @Override
	         public void actionPerformed(ActionEvent e) {
	             // do something
	         	SaveCurrentGraphics();
	         }
	    });

		c2.gridy = gridY++;
		controlsPanel.add(saveCurrentImageGraph, c2);	

		infoLabel = new JLabel(" XY: ");
		c2.gridy = gridY++;
		controlsPanel.add(infoLabel, c2);
				
		int b = bandTriplets.size();
		String[] views = new String[b];
		for (int i = 0; i < b; i++) views[i] = bandTriplets.get(i).toString();
		viewSelectorComboBox = new JComboBox<>(views);
		viewSelectorComboBox.setSelectedIndex(0);
		viewSelectorComboBox.addItemListener(this);
		c2.gridy = gridY++;
		controlsPanel.add(viewSelectorComboBox, c2);
		
		imageSelectorComboBox = new JComboBox<>(new String[] {"..."});
		c2.gridy = gridY++;
		controlsPanel.add(imageSelectorComboBox, c2);
		
		JScrollPane sp = new JScrollPane();
		logArea = new JTextArea("", 10, 20);
		logArea.setFont(new Font("Monospaced", Font.PLAIN, 16));
		logArea.addMouseListener(this);
		sp.getViewport().setView(logArea);
		c2.gridy = gridY++;
		c2.weighty = 10;
		controlsPanel.add(sp, c2);
		
		int tr = truthColor.getRed();
		int tg = truthColor.getGreen();
		int tb = truthColor.getBlue();
		truthBufferColor = new Color(tr, tg, tb, 50);
		
		frame.setVisible(true);
	}
	
	private void setInfo(String message) {
		if (hasGui && mapView != null) {
			mapView.setInfo(message);
		}
	}
	private void clearInfo() {
		if (hasGui && mapView != null) {
			mapView.clearInfo();
		}
	}
		
    private void loadMap() {
    	setInfo("Loading " + currentImageId);
		
		String baseDir = idToDir.get(currentImageId);
		File dir, f;
		int w3 = 0;
		int[][] rs = null;
		int[][] gs = null;
		int[][] bs = null;
		
		// TODO - Change
		// load 3-band file from imageDir/RGB-PanSharpen
		// dir = new File(baseDir, "RGB-PanSharpen");
		// f = new File(dir, "RGB-PanSharpen_" + currentImageId + ".tif");
		// if (!f.exists()) {
		// 	log("Can't find image file: " + f.getAbsolutePath());
		// 	return;
		// }
		dir = new File(baseDir, "images");
		f = new File(dir,currentImageId + "_sat.jpg");
		if (!f.exists()) {
			log("Can't find image file: " + f.getAbsolutePath());
			return;
		}
		try { 
			BufferedImage img = ImageIO.read(f);
			Raster raster = img.getRaster();
			int w = img.getWidth();
			int h = img.getHeight();
			w3 = w;
			rs = new int[w][h];
			gs = new int[w][h];
			bs = new int[w][h];
			int[][] arrs = new int[3][w*h];
			int cnt = 0;
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				int[] samples = raster.getPixel(i, j, new int[3]);
				for (int b = 0; b < 3; b++) {
					if (samples[b] < 0) samples[b] += 65536; // stored as short
					arrs[b][cnt] = Math.max(arrs[b][cnt], samples[b]);
				}
				cnt++;
				rs[i][j] = samples[0];
				gs[i][j] = samples[1];
				bs[i][j] = samples[2];
			}
			for (int b = 0; b < 3; b++) {
				Arrays.sort(arrs[b]);
			}
			int non0 = 0;
			while (arrs[0][non0] == 0) non0++;
			int len = arrs[0].length - non0;
			double[] maxs = new double[3];
			double[] mins = new double[3];
			for (int b = 0; b < 3; b++) {
				maxs[b] = arrs[b][non0 + (int)(0.99 * len)];
				mins[b] = arrs[b][non0 + (int)(0.01 * len)];
			}
			MapData md = new MapData(w, h);
			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
				int r = eq(rs[i][j], mins[0], maxs[0]);
				int g = eq(gs[i][j], mins[1], maxs[1]);
				int b = eq(bs[i][j], mins[2], maxs[2]);
				md.pixels[i][j] = toRGB(r, g, b);
			}
			
			bandTriplets.get(0).mapData = md;
		} 
		catch (Exception e) {
			log("Error reading image from " + f.getAbsolutePath());
			e.printStackTrace();
		}
		
		// TODO - Change
		// load 1-band grayscale file from imageDir/PAN
		// dir = new File(baseDir, "PAN");
		// f = new File(dir, "PAN_" + currentImageId + ".tif");
		// if (!f.exists()) {
		// 	log("Can't find image file: " + f.getAbsolutePath());
		// 	return;
		// }
		// try { 
		// 	BufferedImage img = ImageIO.read(f);
		// 	Raster raster = img.getRaster();
		// 	int w = img.getWidth();
		// 	int h = img.getHeight();
		// 	int[] arr = new int[w*h];
		// 	int cnt = 0;
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
		// 		int sample = raster.getSample(i, j, 0);
		// 		if (sample < 0) sample += 65536; // stored as short
		// 		arr[cnt] = Math.max(arr[cnt], sample);
		// 		cnt++;
		// 		rs[i][j] = sample;
		// 		gs[i][j] = sample;
		// 		bs[i][j] = sample;
		// 	}
		// 	Arrays.sort(arr);
		// 	int non0 = 0;
		// 	while (arr[non0] == 0) non0++;
		// 	int len = arr.length - non0;
		// 	double min = arr[non0 + (int)(0.01 * len)];
		// 	double max = arr[non0 + (int)(0.99 * len)];
		// 	MapData md = new MapData(w, h);
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
		// 		int c = eq(rs[i][j], min, max);
		// 		md.pixels[i][j] = toRGB(c, c, c);
		// 	}
			
		// 	bandTriplets.get(1).mapData = md;
		// } 
		// catch (Exception e) {
		// 	log("Error reading image from " + f.getAbsolutePath());
		// 	e.printStackTrace();
		// }
		
		// TODO - Change
		// load 8-band file from imageDir/MUL into 8 arrays first
		// dir = new File(baseDir, "MUL");
		// f = new File(dir, "MUL_" + currentImageId + ".tif");
		// if (!f.exists()) {
		// 	log("Can't find image file: " + f.getAbsolutePath());
		// 	return;
		// }
		// try {
		// 	BufferedImage img = ImageIO.read(f);
		// 	Raster raster = img.getRaster();
		// 	int w = img.getWidth(); 
		// 	int h = img.getHeight();
		// 	ratio38 = (double)w3 / w;
		// 	double[][][] bandData = new double[8][w][h];
		// 	int max = 0;
		// 	for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
		// 		int[] samples = raster.getPixel(i, j, new int[8]);
		// 		for (int b = 0; b < 8; b++) {
		// 			int v = samples[b];
		// 			if (v < 0) v += 65536; // stored as short
		// 			max = Math.max(max, v);
		// 			bandData[b][i][j] = v;
		// 		}
		// 	}
		// 	if (max > 0) {
		// 		for (int b = 0; b < 8; b++) 
		// 			for (int i = 0; i < w; i++) 
		// 				for (int j = 0; j < h; j++)
		// 					bandData[b][i][j] /= max;
		// 	}
			
		// 	// create all needed combinations
		// 	for (BandTriplet bt: bandTriplets) {
		// 		if (bt.is3band) continue;
		// 		MapData md = new MapData(w, h);
		// 		if (max > 0) {
		// 			for (int i = 0; i < w; i++) for (int j = 0; j < h; j++) {
		// 				int r = (int)(255 * bandData[bt.bands[0]-1][i][j]); 
		// 				int g = (int)(255 * bandData[bt.bands[1]-1][i][j]); 
		// 				int b = (int)(255 * bandData[bt.bands[2]-1][i][j]); 
		// 				md.pixels[i][j] = toRGB(r, g, b);
		// 			}
		// 		}
		// 		bt.mapData = md;
		// 	}
		// }
		// catch (Exception e) {
		// 	log("Error reading image from " + f.getAbsolutePath());
		// 	e.printStackTrace();
		// }
		scale = (double)currentBandTriplet.mapData.W / mapView.getWidth(); 
		x0 = 0; y0 = 0;
		
		currentTrueJunctions = getJunctions(idToTruthRoadSet);
		currentSolutionJunctions = getJunctions(idToSolutionRoadSet);
		
		clearInfo();
	}
	
	private Set<P2> getJunctions(Map<String, RoadSet> idToRS) {
		Set<P2> ret = new HashSet<>();
		// mostly copied from Graph constructor
		// create nodes and connect them with 1-long segments
		Set<P2> allNodes = new HashSet<>();
		Map<Integer, P2> hashToNode = new HashMap<>();
		RoadSet rs = idToRS.get(currentImageId);
		if (rs == null) return ret;
		for (LineString path: rs.roads) {
			P2 prevN = null;
			for (P2 p: path.points) {
				P2 n = new P2(p.x, p.y);
				int hash = n.hashCode();
				if (hashToNode.containsKey(hash)) {
					n = hashToNode.get(hash);
				}
				else {
					allNodes.add(n);
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
		
		// add nodes with more than 2 neighbours
		for (P2 p: allNodes) if (p.edges.size() > 2) ret.add(p);
		return ret;
	}

	private int eq(int c, double min, double max) {
		int v = (int)(255 * (c - min) / (max - min));
		if (v < 0) return 0;
		if (v > 255) return 255;
		return v;
	}
	
	private int toRGB(int r, int g, int b) {
		return (r << 16) | (g << 8) | b;
	}
	
	// TODO - Change
	private void loadBandTriplets() {
		bandTriplets = new Vector<>();
		
		BandTriplet bPanSharp = new BandTriplet();
		bPanSharp.is3band = true;
		bPanSharp.name = "RGB Pan-sharpened";
		bandTriplets.add(bPanSharp);
		currentBandTriplet = bPanSharp;
		
		BandTriplet bPan = new BandTriplet();
		bPan.is3band = true;
		bPan.name = "PAN grayscale";
		bandTriplets.add(bPan);
		
		String line = null;
		int lineNo = 0;
		try {
			LineNumberReader lnr = new LineNumberReader(new FileReader(bandTripletPath));
			while (true) {
				line = lnr.readLine();
				if (line == null) break;
				lineNo++;
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) continue;
				String[] parts = line.split("\t");
				BandTriplet b = new BandTriplet();
				b.is3band = false;
				b.name = parts[1];
				for (int i = 0; i < 3; i++) {
					b.bands[i] = Integer.parseInt(parts[0].substring(i, i+1));
				}
				bandTriplets.add(b);
			}
			lnr.close();
		}
		catch (Exception e) {
			log("Error reading band triplets from " + bandTripletPath);
			log("Line #" + lineNo + " : " + line);
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	private class BandTriplet {
		public String name;
		public int[] bands = new int[3];
		public boolean is3band;
		public MapData mapData;
		
		@Override
		public String toString() {
			if (is3band) return name;
			return bands[0] + "," + bands[1] + "," + bands[2] + " : " + name;
		}
	}
	
	private void SaveCurrentGraphics(){
		try{
			BufferedImage myImage = new BufferedImage(mapView.getWidth(), mapView.getHeight(), BufferedImage.TYPE_INT_RGB);
			Graphics2D myG2 = myImage.createGraphics();
			mapView.paint(myG2);
			ImageIO.write(myImage, "JPEG", new File(saveImageDir + "/" + currentImageId + "_" + currentScore.trim().split(": ")[3].trim() + ".jpg"));
			log("Image Saved =>" + saveImageDir + "/" + currentImageId + "_" + currentScore.trim().split(": ")[3].trim() + ".jpg");
		}
		catch (IOException ie){
			ie.printStackTrace();
		}
	}

	private void repaintMap() {
		if (mapView != null) mapView.repaint();
	}
	
	@SuppressWarnings("serial")
	private class MapView extends JLabel implements MouseListener, MouseMotionListener, MouseWheelListener {
		
		private int mouseX;
		private int mouseY;
		private BufferedImage image;
		private Color invalidColor = new Color(50, 150, 200);
		private int invalidColorI = toRGB(50, 150, 200);
		private String message = null;
		
		public MapView() {
			super();
			this.addMouseListener(this);
			this.addMouseMotionListener(this);
			this.addMouseWheelListener(this);
		}		
		
		public void setInfo(String s) {
			message = s;
			this.paintImmediately(0, 0, this.getWidth(), this.getHeight());
		}
		
		public void clearInfo() {
			setInfo(null);
		}

		@Override
		public void paint(Graphics gr) {
			int W = this.getWidth();
			int H = this.getHeight();
			
			Graphics2D g2 = (Graphics2D) gr;
			g2.setFont(font);
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
			
			if (message != null) {
				g2.setColor(invalidColor);
				g2.fillRect(0, 0, W, H);
				g2.setColor(Color.black);
				g2.drawString(message + " ...", 50, 50);
				return;
			}
						
			if (currentBandTriplet == null || currentBandTriplet.mapData == null) return;
			if (image == null) {
				image = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
			}
			
			MapData mapData = currentBandTriplet.mapData;
			
			
			for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) {
				int c = invalidColorI;
				int mapI = (int)((i - x0) * scale);
				int mapJ = (int)((j - y0) * scale);
				
				if (mapI >= 0 && mapJ >= 0 && mapI < mapData.W && mapJ < mapData.H) {
					c = mapData.pixels[mapI][mapJ];
				}
				image.setRGB(i, j, c);
			}
			g2.drawImage(image, 0, 0, null);
			
			if (showTruthCb.isSelected()) {
				RoadSet truthGraph = idToTruthRoadSet.get(currentImageId);
				if (truthGraph != null) {
					if (showTruthBufferCb.isSelected()) {
						drawGraphPaths(truthGraph, g2, truthBufferColor);
					}
					
					g2.setStroke(strokeWide);
					drawGraph(truthGraph, null, g2, roadBackgroundColor);
					g2.setStroke(strokeThin);
					drawGraph(truthGraph, currentTrueJunctions, g2, truthColor);
				}
			}
			if (showSolutionCb.isSelected()) {
				RoadSet solutionGraph = idToSolutionRoadSet.get(currentImageId);
				if (solutionGraph != null) {
					g2.setStroke(strokeWide);
					drawGraph(solutionGraph, null, g2, roadBackgroundColor);
					g2.setStroke(strokeThin);
					drawGraph(solutionGraph, currentSolutionJunctions, g2, solutionColor);
				}
			}
		}

		private void drawGraph(RoadSet g, Set<P2> junctions, Graphics2D g2, Color color) {
			// graph coordinates are in 3-band space so everything should be scaled if needed
			double r = currentBandTriplet.is3band ? 1 : ratio38;
			
			g2.setColor(color);
			for (LineString road: g.roads) {
				if (road.points.isEmpty()) continue;
				P2 p0 = road.points.get(0);
				P2 p1 = p0;
				for (P2 p2: road.points) {
					if (p2 == p0) continue;
					int x1 = (int)(p1.x / scale / r + x0);
					int y1 = (int)(p1.y / scale / r + y0);
					int x2 = (int)(p2.x / scale / r + x0);
					int y2 = (int)(p2.y / scale / r + y0);
					g2.drawLine(x1, y1, x2, y2);
					p1 = p2;
				}
			}
			
			if (showJunctionsCb.isSelected() && junctions != null) {
				int dIn = 10;
				int dOut = 12;
				for (P2 p: junctions) {
					int x = (int)(p.x / scale / r + x0);
					int y = (int)(p.y / scale / r + y0);
					g2.setColor(roadBackgroundColor);
					g2.fillOval(x-dOut/2, y-dOut/2, dOut, dOut);
					g2.setColor(junctionColor);
					g2.fillOval(x-dIn/2, y-dIn/2, dIn, dIn);					
				}
			}
		}
		
		private void drawGraphPaths(RoadSet g, Graphics2D g2, Color color) {
			// graph coordinates are in 3-band space so everything should be scaled if needed
			double r = currentBandTriplet.is3band ? 1 : ratio38;
			
			g2.setColor(color);
			GeneralPath path = new GeneralPath();
			for (LineString road: g.roads) {
				if (road.points.size() < 2) continue;
				P2 p1 = road.points.get(0);
				double x1 = p1.x / scale / r + x0;
				double y1 = p1.y / scale / r + y0;
				path.moveTo(x1, y1);
				for (int i = 1; i < road.points.size(); i++) {
					P2 p2 = road.points.get(i);
					double x2 = p2.x / scale / r + x0;
					double y2 = p2.y / scale / r + y0;
					path.lineTo(x2, y2);
				}
			}
			float strW = (float)(2 * Metrics.MAX_SNAP_DISTANCE / scale);
			g2.setStroke(new BasicStroke(strW, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
			g2.setColor(color);
			g2.draw(path);
		}

		@Override
		public void mouseClicked(java.awt.event.MouseEvent e) {
			// nothing
		}
		@Override
		public void mouseReleased(java.awt.event.MouseEvent e) {
			repaintMap();
		}
		@Override
		public void mouseEntered(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
		}
		@Override
		public void mouseExited(java.awt.event.MouseEvent e) {
			setCursor(Cursor.getDefaultCursor());
		}

		@Override
		public void mousePressed(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			mouseX = x;
			mouseY = y;
			repaintMap();
		}
		
		@Override
		public void mouseDragged(java.awt.event.MouseEvent e) {
			int x = e.getX();
			int y = e.getY();
			x0 += x - mouseX;
			y0 += y - mouseY;
			mouseX = x;
			mouseY = y;
			repaintMap();
		}

		@Override
		public void mouseMoved(java.awt.event.MouseEvent e) {
			if (currentBandTriplet == null || currentBandTriplet.mapData == null) return;
			int x = e.getX();
			int y = e.getY();
			int i = (int)((x - x0) * scale);
			int j = (int)((y - y0) * scale);
			String info = "NA";
			if (i >= 0 && j >= 0 && i < currentBandTriplet.mapData.W && j < currentBandTriplet.mapData.H) {
				info = i + ", " + j;
			}
			infoLabel.setText(" XY: " + info);
		}


		@Override
		public void mouseWheelMoved(MouseWheelEvent e) {
			mouseX = e.getX();
			mouseY = e.getY();
			double dataX = (mouseX - x0) * scale;
			double dataY = (mouseY - y0) * scale;
			
			double change =  Math.pow(2, 0.5);
			if (e.getWheelRotation() > 0) scale *= change;
			if (e.getWheelRotation() < 0) scale /= change;
			
			x0 = mouseX - dataX / scale;
			y0 = mouseY - dataY / scale;
			
			repaintMap();
		}
	} // class MapView
	

	@Override
	public void actionPerformed(ActionEvent e) {
		// check boxes clicked
		showTruthBufferCb.setEnabled(showTruthCb.isSelected());
		repaintMap();
	}
	
	@Override
	public void itemStateChanged(ItemEvent e) {
		if (e.getStateChange() == ItemEvent.SELECTED) {
			if (e.getSource() == imageSelectorComboBox) {
				// new image selected
				currentImageId = (String) imageSelectorComboBox.getSelectedItem();
				loadMap();
			}
			else if (e.getSource() == viewSelectorComboBox) {
				BandTriplet old = currentBandTriplet;
				// new band triplet selected
				int i = viewSelectorComboBox.getSelectedIndex();
				currentBandTriplet = bandTriplets.get(i);
				// 3 -> 8
				if (old.is3band && !currentBandTriplet.is3band) scale /= ratio38;
				// 8 -> 3
				if (!old.is3band && currentBandTriplet.is3band) scale *= ratio38;				
			}
			repaintMap();
		}
	}	

	@Override
	public void mouseClicked(MouseEvent e) {
		if (e.getSource() != logArea) return;
		try {
			int lineIndex = logArea.getLineOfOffset(logArea.getCaretPosition());
			int start = logArea.getLineStartOffset(lineIndex);
			int end = logArea.getLineEndOffset(lineIndex);
			String line = logArea.getDocument().getText(start, end - start).trim();

			int start1 = logArea.getLineStartOffset(lineIndex+1);
			int end1 = logArea.getLineEndOffset(lineIndex+1);
			String line1 = logArea.getDocument().getText(start1, end1 - start1).trim();
			for (int i = 0; i < imageIds.length; i++) {
				String id = imageIds[i];
				if (id.equals(line) && !id.equals(currentImageId)) {
					currentImageId = id;
					currentScore = line1;
					imageSelectorComboBox.setSelectedIndex(i);
					loadMap();
					repaintMap();
				}
			}
		} 
		catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public void mousePressed(MouseEvent e) {}
	@Override
	public void mouseReleased(MouseEvent e) {}
	@Override
	public void mouseEntered(MouseEvent e) {}
	@Override
	public void mouseExited(MouseEvent e) {}
	
	private void log(String s) {
		if (logArea != null) logArea.append(s + "\n");
		System.out.println(s);
	}
	
	private static Color parseColor(String s) {
		String[] parts = s.split(",");
		int r = Integer.parseInt(parts[0]);
		int g = Integer.parseInt(parts[1]);
		int b = Integer.parseInt(parts[2]);
		int a = parts.length > 3 ? Integer.parseInt(parts[3]) : 255;
		return new Color(r, g, b, a);
	}
	
	private static void exit(String s) {
		System.out.println(s);
		System.exit(1);
	}

	private static String[] parseParamFile(String path) {
		List<String> paramList = new Vector<>();
		try {
			List<String> truthList = new Vector<>();
			List<String> imageDirList = new Vector<>();
			List<String> lines = Utils.readTextLines(path);
			for (String line: lines) {
				if (!line.startsWith("-")) line = "-" + line;
				int pos = line.indexOf("=");
				if (pos == -1) {
					paramList.add(line);
					continue;
				}
				String key = line.substring(0, pos).trim();
				String value = line.substring(pos + 1).trim();
				if (key.equals("-image-dir")) {
					imageDirList.add(value);
				}
				else if (key.equals("-truth")) {
					truthList.add(value);
				}
				else {
					paramList.add(key);
					paramList.add(value);
				}
			} // for lines
			if (!truthList.isEmpty()) {
				paramList.add("-truth");
				String p = "";
				for (String s: truthList) p += s + SEP;
				p = p.substring(0, p.length() - 1);
				paramList.add(p);
			}
			if (!imageDirList.isEmpty()) {
				paramList.add("-image-dir");
				String p = "";
				for (String s: imageDirList) p += s + SEP;
				p = p.substring(0, p.length() - 1);
				paramList.add(p);
			}
		} 
		catch (Exception e) {
			e.printStackTrace();
			exit("Can't parse params file " + path);
		}
		return paramList.toArray(new String[0]);
	}
	
	private static final String SEP = ";";
	
	public static void main(String[] args) throws Exception {
		boolean setDefaults = true;
		for (int i = 0; i < args.length; i++) { // to change settings easily from Eclipse
			if (args[i].equals("-no-defaults")) setDefaults = false;
		}
		
		RoadVisualizer v = new RoadVisualizer();
		v.hasGui = true;
		int w = 1500;
		
		if (setDefaults) {
			v.hasGui = true;
			w = 1500;
			v.truthPaths = null;
			v.solutionPath = null;
			v.imageDirs = null;
			v.bandTripletPath = null;
		}
		else {
			String params;
			// These are just some default settings for local testing, can be ignored.
			
			// sample data
			params = "../data/SpaceNet_Roads_Sample/params.txt";
			
			// training data
			//params = "../data/train/params.txt";
			
			// test data
			//v.hasGui = false;
			//params = "../data/test/params-test.txt";
			
			// validation
//			v.hasGui = false;
//			params = "../submissions/final/params.txt";
			
			args = new String[] {"-params", params};
		}
		
		if (args.length == 2 && args[0].equals("-params")) {
			args = parseParamFile(args[1]);
		}
		
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-no-gui")) v.hasGui = true;
			if (args[i].equals("-debug")) v.isDebug = true;
			if (args[i].equals("-w")) w = Integer.parseInt(args[i+1]);
			if (args[i].equals("-truth")) v.truthPaths = args[i+1].split(SEP);
			if (args[i].equals("-solution")) v.solutionPath = args[i+1];
			if (args[i].startsWith("-image-dir")) v.imageDirs = args[i+1].split(SEP);
			if (args[i].startsWith("-save-img-dir")) v.saveImageDir = args[i+1];
			if (args[i].equals("-band-triplets")) v.bandTripletPath = args[i+1];
			if (args[i].equals("-truth-color")) v.truthColor = parseColor(args[i+1]);
			if (args[i].equals("-solution-color")) v.solutionColor = parseColor(args[i+1]);
		}
		
		if (v.hasGui && (v.imageDirs == null || v.imageDirs.length == 0)) {
			exit("Image folders not set or empty.");
		}
		
		v.setupGUI(w);
		v.run();
	}

}
