package docker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Random;
import java.util.Vector;

public class BaselinePredictor {
	private static final double D = 5 / 0.31;
	private static final double junctionRatio = 0.3;
	private PrintWriter out;
	private int cnt;
	private Random rand;
	private List<Integer> poss;
	
	public static void main(String[] args) {
		// TODO remove, test only
		//args = new String[] {"../data/train/AOI_3_Paris_Roads_Train", "out"};
		
		if (args.length < 2) {
			System.out.println("Usage: docker.BaselinePredictor test_dir[...] out_file");
			System.exit(-1);
		}
		try {
			new BaselinePredictor().run(args);
		} 
		catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void run(String[] args) throws Exception {		
		int n = args.length;
		String outFilePath = args[n-1];
		
		out = new PrintWriter(new BufferedWriter(new FileWriter(outFilePath + ".txt")));
		rand = new Random(0);
		poss = new Vector<>();
		for (double x = D; x < 1300; x += 2*D) poss.add((int)(x));
		
		for (int i = 0; i < n-1; i++) {
			File panDir = new File(args[i], "PAN");
			if (!panDir.exists() || !panDir.isDirectory()) {
				System.out.println("PAN directory not found at " + panDir.getAbsolutePath());
			}
			processDir(panDir);
		}
		out.close();
		System.out.println("Done.");
	}
	
	private void processDir(File dir) throws Exception{
		for (File f: dir.listFiles()) {
			String name = f.getName();
			if (name.startsWith("PAN_") && name.endsWith(".tif")) {
				String id = name.replace("PAN_", "");
				id = id.replace(".tif", "");
				cnt++;
				System.out.println("Processing image " + id + " (" + cnt + ")");
				
				int p1 = poss.get(0);
				int p2 = poss.get(poss.size()-1);
				// id,"LINESTRING (0.00 541.93, 484.20 687.83, 773.90 772.25)" 
				for (int i: poss) {
					StringBuilder sb = new StringBuilder();
					sb.append(id).append(",\"LINESTRING (");
					for (int j: poss) {
						if (j == p1 || j == p2) {
							int j2 = rand.nextDouble() < junctionRatio ? j : j-1;
							sb.append(i).append(" ").append(j2).append(", ");
						}
						else if (rand.nextDouble() < junctionRatio) {
							sb.append(i).append(" ").append(j).append(", ");
						}
					}
					sb.delete(sb.length()-2, sb.length());
					sb.append(")");
					out.println(sb.toString());
				}
				for (int i: poss) {
					StringBuilder sb = new StringBuilder();
					sb.append(id).append(",\"LINESTRING (");
					for (int j: poss) {
						sb.append(j).append(" ").append(i).append(", ");
					}
					sb.delete(sb.length()-2, sb.length());
					sb.append(")");
					out.println(sb.toString());
				}				
			}
		}
	}
}

